import logging
import codecs
import pandas as pd
from pathlib import Path
from pprint import pprint
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
            self.args = eKonf.to_dict(eKonf.merge(self.args, self.info))
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

        self.collapse_ids = self.args.get("collapse_ids", False)
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

        self._id_key = "id"
        self._text_key = "text"
        self._merge_meta_on_key = "merge_meta_on"
        self._keys = self.column_info["keys"] or {}
        self._timestamp = self.column_info.get("timestamp", None)
        self._timestamp_key = "timestamp"
        for k in [self._id_key, self._text_key, self._merge_meta_on_key]:
            if self._keys.get(k) is None:
                continue
            if isinstance(self._keys[k], str):
                self._keys[k] = [self._keys[k]]
            else:
                self._keys[k] = list(self._keys[k])
        self._merge_meta_on = self._keys.get(self._merge_meta_on_key)
        self._id_keys = self._keys[self._id_key]
        if self._merge_meta_on is None:
            self._merge_meta_on = self._id_keys
        self._id_separator = "_"
        self._data_keys = self.column_info.get("data", None)
        self._meta_kyes = self.column_info.get("meta", None)

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
    def ID(self):
        return self._id_key

    @property
    def IDs(self):
        return self._id_keys

    @property
    def TEXT(self):
        return self._text_key

    @property
    def DATA(self):
        if self._data_keys is None:
            return None
        return list(self._data_keys.keys())

    @property
    def METADATA(self):
        if self._meta_kyes is None:
            return None
        return list(self._meta_kyes.keys())

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

    # def __repr__(self):
    # 	return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}\n}})"
    def load_timestamp(self):
        if self._timestamp is None:
            return
        _timestamp_col = self._timestamp.get("key", None)
        if _timestamp_col is None:
            return
        _format = self._timestamp.get("format", None)
        _params = self._timestamp.get("params", {})
        if _params is None:
            _params = {}
        if self._timestamp_key in self._data.columns:
            self._data[self._timestamp_key] = pd.to_datetime(
                self._data[self._timestamp_key], format=_format, **_params
            )
            if self.verbose:
                log.info(f"Loaded timestamp column {self._timestamp_key}")
        elif _timestamp_col in self._metadata.columns:
            self._metadata[self._timestamp_key] = pd.to_datetime(
                self._metadata[_timestamp_col], format=_format, **_params
            )
            df_dt = self._metadata[self._merge_meta_on + [self._timestamp_key]]
            self._metadata = self._metadata.drop(self._timestamp_key, axis=1)
            self._data = self._data.merge(df_dt, on=self._merge_meta_on, how="left")
            if self.verbose:
                log.info(f"Timestamp column {self._timestamp_key} added to data")
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
            log.info(f"Data loaded {len(self._data)} rows")
            print(self._data.head(3))
            print(self._data.tail(3))
        self._loaded = True

    def load_metadata(self):
        if not self.meta_files or len(self.meta_files) == 0:
            log.info("No metadata files found")
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
            log.info(f"Metadata loaded {len(self._metadata)} rows")
            print(self._metadata.head(3))
            print(self._metadata.tail(3))

    def merge_metadata(self):
        if self._metadata is None or self._metadata_merged:
            return
        self._data = self._data.merge(
            self._metadata,
            on=self._merge_meta_on,
            how="left",
        )
        self._metadata_merged = True
        if self.verbose:
            log.info(f"Metadata merged to data")
            print(self._data.head(3))
            print(self._data.tail(3))
