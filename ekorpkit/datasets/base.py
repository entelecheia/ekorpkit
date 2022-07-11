import os
import logging
import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn import preprocessing
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit import eKonf


DESCRIPTION = "ekorpkit datasets"
LICENSE = "Copyright of the dataset is owned by the authors."


log = logging.getLogger(__name__)


class BaseSet:
    __metaclass__ = ABCMeta

    SPLITS = eKonf.SPLITS

    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.name = self.args.name
        self.verbose = self.args.get("verbose", False)
        self.auto = self.args.auto
        self.force = self.args.force
        self.data_dir = self.args["data_dir"]
        self._collapse_ids = self.args.get("collapse_ids", False)

        self._info_cfg = self.args.get("info", None)
        self._pipeline_cfg = self.args.get("pipeline") or {}
        self._pipeline_ = self._pipeline_cfg.get(eKonf.Keys.PIPELINE, [])
        if self._pipeline_ is None:
            self._pipeline_ = []

        self.filetype = None
        self.data_files = None
        self._info = None
        self._summary_info = None
        self._column = None
        self._splits = {}
        self._data = None
        self._loaded = False
        self._classes = None
        self._le = None

    def load_info(self):
        """Load the info file."""
        self.info_file = os.path.join(self.data_dir, f"info-{self.name}.yaml")
        self._info = (
            eKonf.load(self.info_file)
            if eKonf.exists(self.info_file) and not self.force.build
            else {}
        )
        if self._info:
            log.info(f"Loaded info file: {self.info_file}")
            self.args = eKonf.merge(self.args, self._info)
            self._info = eKonf.to_dict(self._info)
        self.filetype = self.args.get("filetype") or "parquet"
        self.filetype = "." + self.filetype.replace(".", "")
        self.description = self.args.get("description", DESCRIPTION)
        self.license = self.args.get("license", LICENSE)
        if self.verbose:
            log.info(
                f"Intantiating a {self.__class__.__name__} [{self.name}] with a config:"
            )
            eKonf.pprint(self.args)
        self.data_files = self.args.get("data_files", None)
        if self.data_files is None:
            self.data_files = {
                self.SPLITS.TRAIN.value: f"{self.name}-train{self.filetype}",
                self.SPLITS.DEV.value: f"{self.name}-dev{self.filetype}",
                self.SPLITS.TEST.value: f"{self.name}-test{self.filetype}",
            }

    def load_column_info(self):
        self._column_info = self.args.get("column_info")
        if self._column_info is None:
            raise ValueError("Column info can't be None")
        self._column = eKonf.instantiate(self._column_info)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.name}"
        return s

    def __getitem__(self, split="train"):
        if split in self.splits:
            return self.splits[split]
        else:
            return None

    def __len__(self):
        return self.num_rows

    @property
    def num_rows(self) -> int:
        """Number of rows in the corpus (same as :meth:`Corpus.__len__`)."""
        if self.data.index is not None:
            return len(self.data.index)
        return len(self.data)

    @property
    def INFO(self):
        return self._info

    @property
    def COLUMN(self):
        return self._column

    @property
    def ID(self):
        return self.COLUMN.ID

    @property
    def IDs(self):
        return self.COLUMN.IDs

    @property
    def IDs(self):
        return self.COLUMN.IDs

    @property
    def DATA(self):
        return self.COLUMN.DATA

    @property
    def DATATYPEs(self):
        return self.COLUMN.DATATYPEs

    @property
    def data(self):
        dfs = []
        for split, _data in self.splits.items():
            if _data is not None:
                dfs.append(_data)
        df = eKonf.concat_data(dfs)
        return df

    @property
    def splits(self):
        return self._splits

    @property
    def summary_info(self):
        return self._summary_info

    @property
    def classes(self):
        if self._classes is None:
            log.info("LabelEncoder is not fitted")
            return None
        return self._classes.tolist()

    def build(self):
        data = None
        if self._pipeline_ and len(self._pipeline_) > 0:
            data = apply_pipeline(data, self._pipeline_, self._pipeline_cfg)
        if data is not None:
            log.info(f"Dataset {self.name} built with {len(data)} rows")
        else:
            log.info(f"Dataset {self.name} is empty")

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

    def load(self):
        if self._loaded:
            return
        for split, data_file in self.data_files.items():
            if eKonf.exists(self.data_dir, data_file):
                data = eKonf.load_data(
                    data_file,
                    self.data_dir,
                    verbose=self.verbose,
                    concatenate=True,
                )
                data = self.COLUMN.init_info(data)
                data = self.COLUMN.append_split(data, split)
                if self._collapse_ids:
                    data = self.COLUMN.combine_ids(data)
                self._splits[split] = data
                if self.verbose:
                    log.info(f"Data loaded {len(data)} rows")
                    print(data.head(3))
                    print(data.tail(3))
            else:
                log.warning(f"File {data_file} not found.")
                # log.info(f"Dataset {self.name} split {split} is empty")
        self._loaded = True

    def summarize(self):
        if not self._loaded:
            log.info(f"Dataset {self.name} is not loaded")
            return
        summary_info = None
        if self._info_cfg:
            summary_info = eKonf.instantiate(self._info_cfg)
        if summary_info:
            summary_info.load(self.INFO)
        for split, data in self.splits.items():
            if data is None:
                continue
            data_file = self.data_files[split]
            if summary_info:
                stats = {"data_file": data_file}
                summary_info.init_stats(split_name=split, stats=stats)
                summary_info.calculate_stats(data, split)
        self._summary_info = summary_info

    def fit_labelencoder(self, data):
        self._le = preprocessing.LabelEncoder()
        self._le.fit(data)
        self._classes = self._le.classes_
        log.info(f"LabelEncoder classes: {self._classes}")

    def transform_labels(self, data):
        if not self._loaded:
            log.info(f"Dataset {self.name} is not loaded")
            return data
        if data is None:
            log.info(f"Data is None")
            return data
        if self._le is None:
            log.info(f"Label encoder is not fitted")
            self.fit_labelencoder(data)
        _data = self._le.transform(data)
        return _data

    def inverse_transform_labels(self, data):
        if not self._loaded:
            log.info(f"Dataset {self.name} is not loaded")
            return data
        if data is None:
            log.info(f"Data is None")
            return data
        if self._le is None:
            log.info(f"Label encoder is not fitted")
            self.fit_labelencoder(data)
        _data = self._le.inverse_transform(data)
        return _data
