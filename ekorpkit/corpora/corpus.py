# import os
import codecs
import pandas as pd
from pathlib import Path
from pprint import pprint
from wasabi import msg
from omegaconf import OmegaConf
from ekorpkit.io.file import load_dataframe, get_filepaths

DESCRIPTION = "Corpus for Language Models"
LICENSE = "Copyright of the corpus is owned by the authors."


class Corpus:
    def __init__(self, **args):
        self.args = OmegaConf.create(args)
        self.name = self.args.name
        self.data_dir = Path(self.args.data_dir)
        self.metadata_dir = Path(self.args.get("metadata_dir", None))
        if self.metadata_dir is None:
            self.metadata_dir = self.data_dir
        self.info_file = self.data_dir / f"info-{self.name}.yaml"
        self.info = OmegaConf.load(self.info_file) if self.info_file.is_file() else {}
        if self.info:
            self.args = OmegaConf.merge(self.args, self.info)
        self.verbose = self.args.get("verbose", False)
        self.autoload = self.args.get("autoload", False)

        if self.verbose:
            msg.info(f"Intantiating a corpus {self.name} with a config:")
            pprint(OmegaConf.to_container(self.args))

        self.filetype = self.args.get("filetype", "csv")
        self.data_files = self.args.data_files
        self.meta_files = self.args.get("meta_files", None)
        if self.data_files is None:
            self.data_files = {
                "train": f"{self.name}*{self.filetype}*",
            }

        self.segment_separator = self.args.get("segment_separator", "\n\n")
        self.sentence_separator = self.args.get("sentence_separator", "\n")
        self.segment_separator = codecs.decode(self.segment_separator, "unicode_escape")
        self.sentence_separator = codecs.decode(
            self.sentence_separator, "unicode_escape"
        )
        self.description = self.args.get("description", DESCRIPTION)
        self.license = self.args.get("license", LICENSE)
        self.column_info = self.args.get("column_info", None)
        self.split_info = self.args.get("splits", None)
        if self.column_info is None:
            raise ValueError("Column info can't be None")
        self.column_info = OmegaConf.to_container(self.column_info)
        if self.split_info:
            self.split_info = OmegaConf.to_container(self.split_info)
        self._keys = self.column_info["keys"]
        self._timestamp = self.column_info.get("timestamp", None)
        self.collapse_ids = self.args.get("collapse_ids", True)

        self._id_key = "id"
        self._text_key = "text"
        self._merge_meta_on_key = "merge_meta_on"
        self._timestamp_key = "timestamp"
        for k in [self._id_key, self._text_key, self._merge_meta_on_key]:
            if isinstance(self._keys[k], str):
                self._keys[k] = [self._keys[k]]
            else:
                self._keys[k] = list(self._keys[k])
        if self._merge_meta_on_key in self._keys:
            self._merge_meta_on = self._keys[self._merge_meta_on_key] 
        else:
            self._merge_meta_on = self._id_key
        self._id_keys = self._keys[self._id_key]
        self._id_separator = "_"
        self._data_keys = self.column_info.get("data", None)
        self._meta_kyes = self.column_info.get("meta", None)

        self._data = None
        self._metadata = None

        # TODO: option to add timestamp column to data
        # timestamp column, conversion rules, etc.

        if self.autoload:
            self.load()
            self.load_metadata()
            self.laod_timestamp()

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    @property
    def num_rows(self) -> int:
        """Number of rows in the dataset (same as :meth:`Dataset.__len__`)."""
        if self._data.index is not None:
            return len(self._data.index)
        return len(self._data)

    # def __repr__(self):
    # 	return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}\n}})"
    def laod_timestamp(self):
        if self._timestamp is None:
            return
        _timestamp_col = self._timestamp.get("key", None)
        if _timestamp_col is None:
            return
        _format = self._timestamp.get("format", None)
        _params = self._timestamp.get("params", {})
        if _params is None:
            _params = {}
        if _timestamp_col in self._metadata.columns:
            self._metadata[self._timestamp_key] = pd.to_datetime(self._metadata[_timestamp_col], format=_format, **_params)
            df_dt = self._metadata[self._merge_meta_on + [self._timestamp_key]]
            self._metadata = self._metadata.drop(self._timestamp_key, axis=1)
            self._data = self._data.merge(df_dt, on=self._merge_meta_on, how="left")
            if self.verbose:
                msg.info(f"Timestamp column {self._timestamp_key} added to data")
                print(self._data.head())
            
    def load(self):
        dfs = []
        _text_cols = self._keys[self._text_key]
        _id_cols = self._keys[self._id_key]

        if len(_text_cols) > 1:
            self._data_keys = {
                k: v for k, v in self._data_keys.items() if k not in _text_cols
            }
            self._data_keys[self._text_key] = "str"

        for split, data_file in self.data_files.items():
            filepaths = get_filepaths(data_file, self.data_dir)
            df = pd.concat(
                [
                    load_dataframe(f, filetype=self.filetype, verbose=self.verbose)
                    for f in filepaths
                ]
            )

            df[_text_cols] = df[_text_cols].fillna("")
            if len(_text_cols) > 1:
                df[self._text_key] = df[_text_cols].apply(
                    lambda row: self.segment_separator.join(row.values.astype(str)),
                    axis=1,
                )

            if self.collapse_ids:
                _id_prefix = f"{split}_" if len(self.data_files) > 1 else ""
                if len(_id_cols) > 1 or len(self.data_files) > 1:
                    df[self._id_key] = df[_id_cols].apply(
                        lambda row: _id_prefix
                        + self._id_separator.join(row.values.astype(str)),
                        axis=1,
                    )
            dfs.append(df[list(self._data_keys.keys())])
        self._data = pd.concat(dfs)
        if self.verbose:
            print(f"Data loaded {len(self._data)} rows")
            print(self._data.head(3))
            print(self._data.tail(3))

    def load_metadata(self):
        if self.meta_files is None:
            return
        dfs = []
        _id_cols = self._keys[self._id_key]
        for split, data_file in self.meta_files.items():
            filepaths = get_filepaths(data_file, self.metadata_dir)
            df = pd.concat(
                [
                    load_dataframe(f, filetype=self.filetype, verbose=self.verbose)
                    for f in filepaths
                ]
            )

            if self.collapse_ids:
                _id_prefix = f"{split}_" if len(self.data_files) > 1 else ""
                if len(_id_cols) > 1 or len(self.data_files) > 1:
                    df[self._id_key] = df[_id_cols].apply(
                        lambda row: _id_prefix
                        + self._id_separator.join(row.values.astype(str)),
                        axis=1,
                    )
            dfs.append(df)
        self._metadata = pd.concat(dfs)
        if self.verbose:
            print(f"Metadata loaded {len(self._metadata)} rows")
            print(self._metadata.head(3))
            print(self._metadata.tail(3))

    def merge_metadata(self):
        if self._metadata is None:
            return
        self._data = self._data.merge(
            self._metadata,
            on=self._merge_meta_on,
            how="left",
        )
        if self.verbose:
            print(f"Metadata merged to data")
            print(self._data.head(3))
            print(self._data.tail(3))
