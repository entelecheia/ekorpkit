import os
import logging
from ekorpkit import eKonf
from .base import BaseSet


log = logging.getLogger(__name__)


class Dataset(BaseSet):
    """Dataset class."""

    def __init__(self, **args):
        super().__init__(**args)
        if isinstance(self.name, list):
            self.name = self.name[0]
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        if use_name_as_subdir:
            self.data_dir = os.path.join(self.data_dir, self.name)

        self.load_info()
        self.load_column_info()

        if self.auto.build:
            if self.force.build or not eKonf.exists(
                self.data_dir, self.data_files[self.SPLITS.TRAIN]
            ):
                self.build()
        if self.auto.load:
            self.load()

    @property
    def train_data(self):
        if self.SPLITS.TRAIN not in self.splits:
            return None
        return self.splits[self.SPLITS.TRAIN]

    @train_data.setter
    def train_data(self, data):
        self.splits[self.SPLITS.TRAIN.value] = data

    @property
    def dev_data(self):
        if self.SPLITS.DEV not in self.splits:
            return None
        return self.splits[self.SPLITS.DEV]

    @dev_data.setter
    def dev_data(self, data):
        self.splits[self.SPLITS.DEV.value] = data

    @property
    def test_data(self):
        if self.SPLITS.TEST not in self.splits:
            return None
        return self.splits[self.SPLITS.TEST]

    @test_data.setter
    def test_data(self, data):
        self.splits[self.SPLITS.TEST.value] = data
