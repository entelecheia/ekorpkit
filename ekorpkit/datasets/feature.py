import os
import logging
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipeline
from .dataset import Dataset


log = logging.getLogger(__name__)


class FeatureSet(Dataset):
    """Feature class."""

    def __init__(self, **args):
        super().__init__(**args)

    @property
    def X_train(self):
        if self.train_data is not None:
            return self.train_data[self.COLUMN.X]
        else:
            return None

    @property
    def X_dev(self):
        if self.dev_data is not None:
            return self.dev_data[self.COLUMN.X]
        else:
            return None

    @property
    def X_test(self):
        if self.test_data is not None:
            return self.test_data[self.COLUMN.X]
        else:
            return None

    @property
    def y_train(self):
        if self.train_data is not None:
            return self.train_data[self.COLUMN.Y]
        else:
            return None

    @y_train.setter
    def y_train(self, data):
        self.train_data[self.COLUMN.Y] = data

    @property
    def y_dev(self):
        if self.dev_data is not None:
            return self.dev_data[self.COLUMN.Y]
        else:
            return None

    @y_dev.setter
    def y_dev(self, data):
        if self.dev_data is not None:
            self.dev_data[self.COLUMN.Y] = data

    @property
    def y_test(self):
        if self.test_data is not None:
            return self.test_data[self.COLUMN.Y]
        else:
            return None

    @y_test.setter
    def y_test(self, data):
        if self.test_data is not None:
            self.test_data[self.COLUMN.Y] = data

    @property
    def X(self):
        return self.data[self.COLUMN.X]

    @property
    def y(self):
        return self.data[self.COLUMN.Y]

    def persist(self):
        if not self._loaded:
            log.info(f"Dataset {self.name} is not loaded")
            return
        if self.summary_info is None:
            self.summarize()
        for split, data in self._splits.items():
            if data is None:
                continue
            data_file = self.data_files[split]
            eKonf.save_data(
                data,
                data_file,
                base_dir=self.data_dir,
                verbose=self.verbose,
            )
        if self.summary_info is not None:
            self.summary_info.save(info={"column_info": self.COLUMN.INFO})
