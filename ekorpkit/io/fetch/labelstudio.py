import os
import logging
import requests
import pandas as pd
import json
from omegaconf import DictConfig
from pydantic import BaseModel
from .config import BaseFetcher
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class LabelStudioConfig(BaseModel):
    project_id: str = None
    api: DictConfig = None
    jobs: DictConfig = None
    annotation_file: str = None
    request_schema: DictConfig = None
    label_config: DictConfig = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **args):
        super().__init__(**args)

        if not self.api.token:
            raise ValueError(
                "api.token is required, set envrionment variable LABEL_STUDIO_USER_TOKEN or config 'api.token' directly"
            )
        if not self.api.server:
            raise ValueError(
                "api.server is required, set envrionment variable LABEL_STUDIO_SERVER or config 'api.server' directly"
            )

    @property
    def headers(self):
        return eKonf.to_dict(self.api.header)


class LabelStudio(BaseFetcher):
    model: LabelStudioConfig = None

    def __init__(self, config_name: str = "labelstudio", **args):
        config_group = f"io/fetcher={config_name}"
        super().__init__(config_group=config_group, **args)

    def initialize_configs(self, **kwargs):
        super().initialize_configs(**kwargs)

        self.model = LabelStudioConfig(**self.config.model)

    @property
    def project_id(self):
        return self.model.project_id

    @property
    def headers(self):
        return self.model.headers

    def get_url(self, job_name, project_id=None):
        if project_id:
            self.model.project_id = project_id

        url = self.model.api.server + self.model.jobs[job_name]
        if "{id}" in url:
            url = url.format(id=self.project_id)
        return url

    def get_request_schema(self, request_name):
        return eKonf.to_dict(self.model.request_schema[request_name])

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
            log.info(f"{project_id} is fetched")
        return response.json()

    def delete_project(self, project_id):
        url = self.get_url("get_project", project_id=project_id)
        response = requests.delete(url, headers=self.headers)
        if response.status_code == 204:
            log.info(f"{project_id} is deleted")
        else:
            log.info(f"{project_id} is not deleted")

    def create_project(self, title, description=None, label_config=None, **kargs):
        project_list = self.list_projects()
        for p in project_list["results"]:
            if p["title"] == title:
                log.info(f"{title} already exists with id {p['id']}")
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
            self.model.project_id = response["id"]
            log.info(f"project {title} is created with id {self.model.project_id}")
        else:
            log.info(f"project {title} is not created")
        return response

    def import_data(self, data, project_id=None):
        url = self.get_url("import", project_id=project_id)
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            log.info(f"data is imported to {project_id}")
        return response.json()

    def import_file(self, file_path, project_id=None):
        url = self.get_url("import", project_id=project_id)
        response = requests.post(
            url, headers=self.headers, files={"file": open(file_path, "rb")}
        )
        if response.status_code == 201:
            log.info(f"{file_path} is imported to {project_id}")
        return response.json()

    def export_annotations(self, project_id=None, annotation_file=None):
        req_body = self.get_request_schema("export")
        url = self.get_url("export", project_id=project_id)

        json_path = self.annotation_path(annotation_file)
        log.info(f"fetching {url} with {req_body} to {json_path}")
        response = requests.get(url, headers=self.headers, params=req_body)
        with open(json_path, "w") as f:
            f.write(response.text)
        log.info(f"{json_path} is downloaded")
        return self.annotations_to_dataframe(json_path)

    def annotation_path(self, annotation_file=None):
        if annotation_file is None:
            annotation_file = self.model.annotation_file
        return str(self.batch_dir / f"{self.batch.file_prefix}_{annotation_file}")

    @property
    def export_file(self):
        return f"{self.batch.file_prefix}_export.{self.batch.output_extention}"

    def prediction_path(self, prediction_file=None):
        if prediction_file is None:
            prediction_file = self.model.prediction_file
        return str(self.batch_dir / f"{self.batch.file_prefix}_{prediction_file}")

    def annotations_to_dataframe(self, anotation_path=None, output_file=None):

        if anotation_path is None:
            anotation_path = self.annotation_path()
        if not os.path.exists(anotation_path):
            log.info(f"{anotation_path} does not exist")
            return None
        df = pd.read_json(anotation_path)

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
        self.__data__ = df
        if self.verbose:
            log.info(df.head())
        output_file = output_file or self.export_file
        eKonf.save_data(df, output_file, self.batch_dir)
        self.save_config()
        return df

    def dataframe_to_predictions(
        self,
        data,
        prediction_file="predictions.json",
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

        json_path = self.prediction_path(prediction_file)
        with open(json_path, "w") as f:
            json.dump(annotations, f, ensure_ascii=False)
        log.info(f"{json_path} is saved")
        return json_path
