import os
import logging
import pandas as pd
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf
from .base import BaseSet


DESCRIPTION = "ekorpkit datasets"
LICENSE = "Copyright of the dataset is owned by the authors."


log = logging.getLogger(__name__)


class Corpus(BaseSet):
    def __init__(self, **args):
        super().__init__(**args)
        if isinstance(self.name, list):
            self.name = self.name[0]
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        if use_name_as_subdir:
            self.data_dir = os.path.join(self.data_dir, self.name)
        self.metadata_dir = self.args.get("metadata_dir", None)
        if self.metadata_dir is None:
            self.metadata_dir = self.data_dir
        else:
            if use_name_as_subdir:
                self.metadata_dir = os.path.join(self.metadata_dir, self.name)

        self.load_info()
        self.load_column_info()

        self.meta_files = self.args.get("meta_files", None)
        self._metadata = None
        self._metadata_merged = False

        if self.auto.load:
            self.load()
            self.load_metadata()
            self.load_timestamp()
            if self.auto.merge:
                self.merge_metadata()

    @property
    def TEXT(self):
        return self.COLUMN.TEXT

    @property
    def METADATA(self):
        return self.COLUMN.METADATA

    @property
    def metadata(self):
        return self._metadata

    def load_timestamp(self):
        for split, data in self.splits.items():
            if data is None:
                continue
            data, self._metadata = self.COLUMN.to_timestamp(data, self.metadata)
            self._splits[split] = data
            if self.verbose:
                print(data.head())

    def load_metadata(self):
        if not self.meta_files or len(self.meta_files) == 0:
            log.info("No metadata files found")
            return

        dfs = []
        for split, data_file in self.meta_files.items():
            df = eKonf.load_data(data_file, self.metadata_dir, concatenate=True)
            df = self.COLUMN.append_split_to_meta(df, split)
            dfs.append(df)
        self._metadata = pd.concat(dfs)
        if self._collapse_ids:
            self._metadata = self.COLUMN.combine_ids(self._metadata)
        if self.verbose:
            log.info(f"Metadata loaded {len(self._metadata)} rows")
            print(self._metadata.head(3))
            print(self._metadata.tail(3))

    def merge_metadata(self):
        if self._metadata is None or self._metadata_merged:
            return
        for split, data in self.splits.items():
            if data is None:
                continue
            data = self.COLUMN.merge_metadata(data, self._metadata)
            data = self.COLUMN.init_info(data)
            self._splits[split] = data
        self._metadata_merged = True
        log.info(f"Metadata merged to data")
        if self.verbose:
            print(self._data.head(3))
            print(self._data.tail(3))
