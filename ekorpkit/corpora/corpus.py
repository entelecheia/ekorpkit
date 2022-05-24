import logging
import codecs
import pandas as pd
from pathlib import Path
from ekorpkit import eKonf
from ekorpkit.io.file import load_dataframe, get_filepaths

DESCRIPTION = "Corpus for Language Models"
LICENSE = "Copyright of the corpus is owned by the authors."


log = logging.getLogger(__name__)


class Corpus:
    def __init__(self, **args):
        self.args = eKonf.to_dict(args)
        self.name = self.args["name"]
        if isinstance(self.name, list):
            self.name = self.name[0]
        self.verbose = self.args.get("verbose", False)
        self.autoload = self.args.get("autoload", False)
        self.automerge = self.args.get("automerge", False)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self.data_dir = Path(self.args["data_dir"])
        if use_name_as_subdir:
            self.data_dir = self.data_dir / self.name
        self.metadata_dir = self.args.get("metadata_dir", None)
        if self.metadata_dir is None:
            self.metadata_dir = self.data_dir
        else:
            self.metadata_dir = Path(self.metadata_dir)
            if use_name_as_subdir:
                self.metadata_dir = self.metadata_dir / self.name
        self.info_file = self.data_dir / f"info-{self.name}.yaml"
        self.info = eKonf.load(self.info_file) if self.info_file.is_file() else {}
        if self.info:
            if self.verbose:
                log.info(f"Loaded info file: {self.info_file}")
            self.args = eKonf.to_dict(eKonf.update(self.args, self.info))
            self.info = eKonf.to_dict(self.info)

        if self.verbose:
            log.info(f"Intantiating a corpus {self.name} with a config:")
            eKonf.pprint(self.args)

        self.filetype = self.args.get("filetype", "csv")
        self.data_files = self.args.get("data_files", None)
        self.meta_files = self.args.get("meta_files", None)
        if self.data_files is None:
            self.data_files = {
                "train": f"{self.name}*{self.filetype}*",
            }

        self._collapse_ids = self.args.get("collapse_ids", False)
        self.description = self.args.get("description", DESCRIPTION)
        self.license = self.args.get("license", LICENSE)
        self.split_info = self.args.get("splits", None)
        self._column_info = self.args.get("column_info", None)
        if self._column_info is None:
            raise ValueError("Column info can't be None")

        self._column = eKonf.instantiate(self._column_info)

        self._data = None
        self._metadata = None
        self._metadata_merged = False
        self._loaded = False

        if self.autoload:
            self.load()
            self.load_metadata()
            self.load_timestamp()
            if self.automerge:
                self.merge_metadata()

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.name}"
        return s

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
    def TEXT(self):
        return self.COLUMN.TEXT

    @property
    def DATA(self):
        return self.COLUMN.DATA

    @property
    def METADATA(self):
        return self.COLUMN.METADATA

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    def __len__(self):
        return len(self._data)

    @property
    def num_rows(self) -> int:
        """Number of rows in the corpus (same as :meth:`Corpus.__len__`)."""
        if self._data.index is not None:
            return len(self._data.index)
        return len(self._data)

    def load_timestamp(self):
        self._data, self._metadata = self.COLUMN.to_timestamp(
            self._data, self._metadata
        )
        if self.verbose:
            print(self._data.head())

    def load(self):
        dfs = []
        for split, data_file in self.data_files.items():
            filepaths = get_filepaths(data_file, self.data_dir)
            df = pd.concat([load_dataframe(f, verbose=self.verbose) for f in filepaths])
            df = self.COLUMN.combine_texts(df)
            if self._collapse_ids:
                df = self.COLUMN.append_split(df, split)
            dfs.append(df[self.COLUMN.DATA])
        self._data = pd.concat(dfs)
        if self._collapse_ids:
            self.COLUMN.combine_ids(self._data)
        if self.verbose:
            log.info(f"Data loaded {len(self._data)} rows")
            print(self._data.head(3))
            print(self._data.tail(3))
        self._loaded = True

    def load_metadata(self):
        if not self.meta_files or len(self.meta_files) == 0:
            log.info("No metadata files found")
            return

        dfs = []
        for split, data_file in self.meta_files.items():
            filepaths = get_filepaths(data_file, self.metadata_dir)
            df = pd.concat([load_dataframe(f, verbose=self.verbose) for f in filepaths])
            if self._collapse_ids:
                df = self.COLUMN.append_split(df, split)
            dfs.append(df)
        self._metadata = pd.concat(dfs)
        if self._collapse_ids:
            self.COLUMN.combine_ids(self._metadata)
        if self.verbose:
            log.info(f"Metadata loaded {len(self._metadata)} rows")
            print(self._metadata.head(3))
            print(self._metadata.tail(3))

    def merge_metadata(self):
        if self._metadata is None or self._metadata_merged:
            return
        self._data = self.COLUMN.merge_metadata(self._data, self._metadata)
        self._metadata_merged = True
        if self.verbose:
            log.info(f"Metadata merged to data")
            print(self._data.head(3))
            print(self._data.tail(3))
