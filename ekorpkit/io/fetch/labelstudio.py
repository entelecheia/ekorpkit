import os
import logging
import requests
import pandas as pd
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
                "api.token is required, set envrionment variable LABELSTUDIO_TOKEN or config 'api.token' directly"
            )
        if not self._api.server:
            raise ValueError(
                "api.server is required, set envrionment variable LABELSTUDIO_SERVER or config 'api.server' directly"
            )
        self._project_id = self.args.project_id
        if not self._project_id:
            raise ValueError("project_id is required")
        self._query = self.args.query
        self._json_file = self.args.json_file

        if self.auto.load:
            self.fetch()

    def _fetch(self):
        headers = eKonf.to_dict(self._api.header)
        _query = [f"{k}={v}" for k, v in self._query.items() if v]
        file_addr = self._api.server + self._api.path + "?" + "&".join(_query)

        json_path = os.path.join(self.output_dir, self._json_file)
        log.info(f"fetching {file_addr} to {json_path}")
        file_res = requests.get(file_addr, headers=headers)
        with open(json_path, "wb") as f:
            f.write(file_res.content)
        log.info(f"{json_path} is downloaded")
        self._to_dataframe(json_path)

    def _to_dataframe(self, json_path):

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
        eKonf.save_data(df, self.output_file)
