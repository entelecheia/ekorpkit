import pandas as pd
from wasabi import msg
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from .corpus import Corpus


class Corpora:
    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.names = args.name
        if isinstance(self.names, str):
            self.names = [self.names]
        self.data_dir = args.data_dir
        self.metadata_dir = self.args.get("metadata_dir", None)
        if self.metadata_dir is None:
            self.metadata_dir = self.data_dir
        self.data_files = self.args.get("data_files", None)
        self.meta_files = self.args.get("meta_files", None)
        self.corpora = {}
        self._data = None
        self._metadata = None
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        self.verbose = args.get("verbose", False)
        self.column_info = self.args.get("column_info", None)

        self._corpus_key = "corpus"
        self._text_key = "text"
        self._id_key = "id"
        self._id_separator = "_"

        if self.column_info:
            self._keys = self.column_info["keys"]
            if self._keys is not None:
                for k in [self._id_key, self._text_key]:
                    if isinstance(self._keys[k], str):
                        self._keys[k] = [self._keys[k]]
                    else:
                        self._keys[k] = list(self._keys[k])
                self._id_keys = self._keys[self._id_key]
            else:
                self._id_keys = [self._id_key]
            self._data_keys = self.column_info.get("data", None)
            self._meta_kyes = self.column_info.get("meta", None)

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.names:
                print(f"processing {name}")
                if use_name_as_subdir:
                    data_dir = f"{self.data_dir}/{name}"
                    metadata_dir = f"{self.metadata_dir}/{name}"
                else:
                    data_dir = self.data_dir
                    metadata_dir = self.metadata_dir
                args["data_dir"] = data_dir
                args["metadata_dir"] = metadata_dir
                args["name"] = name
                if self.data_files is not None:
                    if name in self.data_files:
                        args["data_files"] = self.data_files[name]
                    elif "train" in self.data_files:
                        args["data_files"] = self.data_files
                if self.meta_files is not None:
                    if name in self.meta_files:
                        args["meta_files"] = self.meta_files[name]
                    elif "train" in self.meta_files:
                        args["meta_files"] = self.meta_files
                corpus = Corpus(**args)
                self.corpora[name] = corpus
            print(f"\n >>> Elapsed time: {elapsed()} <<< ")

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname}\n----------\n"
        for name in self.corpora.keys():
            s += f"{str(name)}\n"
        return s

    def __getitem__(self, name):
        return self.corpora[name]

    @property
    def id_key(self):
        return self._id_key

    @property
    def id_keys(self):
        return self._id_keys

    @property
    def text_key(self):
        return self._text_key

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    def load(self):
        for corpus in self:
            corpus.load()

    def concat_corpora(self, append_corpus_name=True):
        dfs = []
        df_metas = []
        if append_corpus_name:
            if self._corpus_key not in self._id_keys:
                self._id_keys.append(self._corpus_key)

        for name in self.corpora:
            df = self.corpora[name]._data
            if append_corpus_name:
                df[self._corpus_key] = name
            dfs.append(df)
            df_meta = self.corpora[name]._metadata
            if df_meta is not None:
                if append_corpus_name:
                    df_meta[self._corpus_key] = name
                df_metas.append(df_meta)
        self._data = pd.concat(dfs, ignore_index=True)
        if len(df_metas) > 0:
            self._metadata = pd.concat(df_metas, ignore_index=True)
        if self.verbose:
            msg.good(f"concatenated {len(dfs)} corpora")
            print(self._data.head())

    def __iter__(self):
        for corpus in self.corpora.values():
            yield corpus

    def __getitem__(self, name):
        if name not in self.corpora:
            raise KeyError(f"{name} not in corpora")
        return self.corpora[name]

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
