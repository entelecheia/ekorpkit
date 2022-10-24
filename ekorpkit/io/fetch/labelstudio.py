import os
import logging
import requests
import pandas as pd
import json
from ekorpkit.io.fetch.base import BaseFetcher
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class LabelStudio(BaseFetcher):
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        super().__init__(**args)
        self._api = self.args.api
        if not self._api.token:
            raise ValueError(
                "api.token is required, set envrionment variable LABEL_STUDIO_USER_TOKEN or config 'api.token' directly"
            )
        if not self._api.server:
            raise ValueError(
                "api.server is required, set envrionment variable LABEL_STUDIO_SERVER or config 'api.server' directly"
            )
        self._jobs = self.args.jobs
        self._request_schema = self.args.request_schema
        self._project_id = self.args.project_id
        self._json_file = self.args.json_file

    @property
    def project_id(self):
        if not self._project_id:
            raise ValueError("project_id is required")
        return self._project_id

    @property
    def headers(self):
        return eKonf.to_dict(self._api.header)

    def get_url(self, job_name, project_id=None):
        if project_id:
            self._project_id = project_id

        url = self._api.server + self._jobs[job_name]
        if "{id}" in url:
            url = url.format(id=self.project_id)
        return url

    def get_request_schema(self, request_name):
        return eKonf.to_dict(self._request_schema[request_name])

    def list_projects(self, verbose=False):
        url = self.get_url("project")
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            response = response.json()
            if verbose:
                for p in response["results"]:
                    print(f"{p['id']}: {p['title']}")
        return response

    def get_project(self, project_id):
        url = self.get_url("get_project", project_id=project_id)
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            print(f"{project_id} is fetched")
        return response.json()

    def delete_project(self, project_id):
        url = self.get_url("get_project", project_id=project_id)
        response = requests.delete(url, headers=self.headers)
        if response.status_code == 204:
            print(f"{project_id} is deleted")
        else:
            print(f"{project_id} is not deleted")

    def create_project(self, title, description=None, label_config=None, **kargs):
        project_list = self.list_projects()
        for p in project_list["results"]:
            if p["title"] == title:
                print(f"{title} already exists with id {p['id']}")
                return p

        url = self.get_url("project")
        req_body = self.get_request_schema("project")
        req_body["title"] = title
        if description:
            req_body["description"] = description
        if label_config:
            req_body["label_config"] = label_config
        req_body.update(kargs)

        response = requests.post(url, headers=self.headers, json=req_body)
        response = response.json()
        if "id" in response:
            self._project_id = response["id"]
            print(f"project {title} is created with id {self._project_id}")
        else:
            print(f"project {title} is not created")
        return response

    def import_data(self, data, project_id=None):
        url = self.get_url("import", project_id=project_id)
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            print(f"data is imported to {project_id}")
        return response.json()

    def import_file(self, file_path, project_id=None):
        url = self.get_url("import", project_id=project_id)
        response = requests.post(
            url, headers=self.headers, files={"file": open(file_path, "rb")}
        )
        if response.status_code == 201:
            print(f"{file_path} is imported to {project_id}")
        return response.json()

    def download_export(self, project_id=None, json_file=None):
        req_body = self.get_request_schema("export")
        url = self.get_url("export", project_id=project_id)

        json_file = json_file or self._json_file
        json_path = os.path.join(self.output_dir, json_file)
        log.info(f"fetching {url} with {req_body} to {json_path}")
        response = requests.get(url, headers=self.headers, params=req_body)
        with open(json_path, "w") as f:
            f.write(response.text)
        log.info(f"{json_path} is downloaded")
        return json_path

    def annotations_to_dataframe(self, json_path, output_file=None):

        df = pd.read_json(json_path)

        def to_list(annotations):
            return [
                [
                    ann["id"],
                    ann["completed_by"],
                    ann["result"][0]["origin"],
                    ann["result"][0]["value"]["choices"][0],
                ]
                for ann in annotations
                if ann["result"]
            ]

        df["annotations"] = df.annotations.apply(to_list)
        df["text"] = df.data.apply(lambda x: x["text"])
        cols = ["id", "annotations", "text"]
        df = df[cols]
        df = df.explode("annotations")
        cols = ["annot_id", "annotator", "origin", "labels"]
        df[cols] = df.annotations.apply(pd.Series)
        df = df.dropna(subset=cols)
        cols = ["annot_id", "annotator"]
        df[cols] = df[cols].astype(int)
        df = df.drop(columns="annotations")
        self._data = df
        if self.verbose:
            print(df.head())
        output_file = output_file or self._output_file
        eKonf.save_data(df, output_file)
        print(f"{output_file} is saved")
        return df

    def dataframe_to_predictions(
        self,
        data,
        json_file="predictions.json",
        type="choices",
        choices_name="sentiment",
        choices_toname="text",
        data_choices="pred_labels",
        data_score="pred_probs",
        model_version="1.0",
    ):
        data = data.to_dict(orient="records")
        annotations = []
        for d in data:
            annotations.append(
                {
                    "data": d,
                    "predictions": [
                        {
                            "model_version": model_version,
                            "result": [
                                {
                                    "from_name": choices_name,
                                    "to_name": choices_toname,
                                    "type": type,
                                    "readonly": False,
                                    "hidden": False,
                                    "value": {
                                        "choices": [d[data_choices]],
                                    },
                                }
                            ],
                            "score": d[data_score],
                        }
                    ],
                }
            )
        json_path = os.path.join(self.output_dir, json_file)
        with open(json_path, "w") as f:
            json.dump(annotations, f, ensure_ascii=False)
        print(f"{json_path} is saved")
        return json_path
