import os
import logging
from ekorpkit import eKonf
from pydantic import validator
from .config import BaseDatasetConfig, SPLITS


log = logging.getLogger(__name__)


class Dataset(BaseDatasetConfig):
    use_name_as_subdir: bool = True

    class Config:
        exclude = {"path", "project", "info"}

    def __init__(self, config_name: str = "_default_", **args):
        config_group = f"dataset={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)
        if self.use_name_as_subdir:
            self.data_dir = os.path.join(self.data_dir, self.name)

        self.load_info()
        self.load_column_info()

        if self.auto.build:
            if self.force.build or not eKonf.exists(
                self.data_dir, self.data_files[SPLITS.TRAIN]
            ):
                self.build()
        if self.auto.load:
            self.load()

    @validator("name", pre=True, always=True)
    def _check_name(cls, v):
        if isinstance(v, list):
            return v[0]
        return v

    @property
    def train_data(self):
        if SPLITS.TRAIN not in self.splits:
            return None
        return self.splits[SPLITS.TRAIN]

    @train_data.setter
    def train_data(self, data):
        self.splits[SPLITS.TRAIN.value] = data

    @property
    def dev_data(self):
        if SPLITS.DEV not in self.splits:
            return None
        return self.splits[SPLITS.DEV]

    @dev_data.setter
    def dev_data(self, data):
        self.splits[SPLITS.DEV.value] = data

    @property
    def test_data(self):
        if SPLITS.TEST not in self.splits:
            return None
        return self.splits[SPLITS.TEST]

    @test_data.setter
    def test_data(self, data):
        self.splits[SPLITS.TEST.value] = data
