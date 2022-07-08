import logging
import requests
from ekorpkit.io.fetch.base import BaseFetcher
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class LabelStudio(BaseFetcher):
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        super().__init__(**args)
        self._api = self.args.api
        if not self._api.token:
            raise ValueError("token is required")
        self._project_id = self.args.project_id
        if not self._project_id:
            raise ValueError("project_id is required")
        self._query = self.args.query

        if self.auto.load:
            self.fetch()

    def _fetch(self):
        headers = eKonf.to_dict(self._api.header)
        _query = [f"{k}={v}" for k, v in self._query.items() if v]
        file_addr = self._api.server + self._api.path + "?" + "&".join(_query)

        log.info(f"fetching {file_addr} to {self._path.output}")
        file_res = requests.get(file_addr, headers=headers)
        with open(self._path.output.filepath, "wb") as f:
            f.write(file_res.content)
        log.info(f"{self._path.output.filepath} is downloaded")
