# import os
import codecs
import pandas as pd
from pathlib import Path
from pprint import pprint
from wasabi import msg
from omegaconf import OmegaConf
from ekorpkit.io.file import load_dataframe, get_filepaths

# from hydra.utils import instantiate
# from omegaconf.dictconfig import DictConfig
# from ekorpkit.utils.func import ordinal, elapsed_timer
# import swifter


DESCRIPTION = "Corpus for Language Models"
LICENSE = "Copyright of the corpus is owned by the authors."


class Corpus:
    def __init__(self, **args):
        self.args = OmegaConf.create(args)
        self.name = self.args.name
        self.data_dir = Path(self.args.data_dir)
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
        self.collapse_ids = self.args.get("collapse_ids", True)

        for k in ["id", "text"]:
            if isinstance(self._keys[k], str):
                self._keys[k] = [self._keys[k]]
            else:
                self._keys[k] = list(self._keys[k])

        self._id_keys = self._keys["id"]
        self._text_key = "text"
        self._id_key = "id"
        self._id_separator = "_"
        self._data_keys = self.column_info.get("data", None)
        self._meta_kyes = self.column_info.get("meta", None)

        self._data = None
        self._metadata = None

        if self.autoload:
            self.load()
            self.load_metadata()

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

    def load(self):
        dfs = []
        _text_keys = self._keys["text"]
        _id_keys = self._keys["id"]

        if len(_text_keys) > 1:
            self._data_keys = {
                k: v for k, v in self._data_keys.items() if k not in _text_keys
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

            df[_text_keys] = df[_text_keys].fillna("")
            if len(_text_keys) > 1:
                df[self._text_key] = df[_text_keys].apply(
                    lambda row: self.segment_separator.join(row.values.astype(str)),
                    axis=1,
                )

            if self.collapse_ids:
                _id_prefix = f"{split}_" if len(self.data_files) > 1 else ""
                if len(_id_keys) > 1 or len(self.data_files) > 1:
                    df[self._id_key] = df[_id_keys].apply(
                        lambda row: _id_prefix
                        + self._id_separator.join(row.values.astype(str)),
                        axis=1,
                    )
            dfs.append(df[list(self._data_keys.keys())])
        self._data = pd.concat(dfs)
        print(self._data.head(3))
        print(self._data.tail(3))

    def load_metadata(self):
        if self.meta_files is None:
            return
        dfs = []
        _id_keys = self._keys["id"]
        for split, data_file in self.meta_files.items():
            filepaths = get_filepaths(data_file, self.data_dir)
            df = pd.concat(
                [
                    load_dataframe(f, filetype=self.filetype, verbose=self.verbose)
                    for f in filepaths
                ]
            )

            if self.collapse_ids:
                _id_prefix = f"{split}_" if len(self.data_files) > 1 else ""
                if len(_id_keys) > 1 or len(self.data_files) > 1:
                    df[self._id_key] = df[_id_keys].apply(
                        lambda row: _id_prefix
                        + self._id_separator.join(row.values.astype(str)),
                        axis=1,
                    )
            dfs.append(df)
        self._metadata = pd.concat(dfs)
        print(self._metadata.head(3))
        print(self._metadata.tail(3))

    def merge_metadata(self):
        if self._metadata is None:
            return
        self._data = self._data.merge(
            self._metadata,
            on=self._id_keys,
            how="left",
            suffixes=("", "_metadata"),
            validate="one_to_one",
        )
