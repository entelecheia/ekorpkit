import pandas as pd
import logging
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from .corpus import Corpus


log = logging.getLogger(__name__)


class Corpora:
    def __init__(self, **args):
        args = eKonf.to_dict(args)
        self.args = args
        self.name = args["name"]
        self.corpora = args.get("corpora", None)
        if self.corpora is None:
            self.corpora = self.name
        if isinstance(self.corpora, str):
            self.corpora = {self.corpora: None}
        elif isinstance(self.corpora, list):
            self.corpora = {name: None for name in self.corpora}
        if isinstance(self.name, list):
            self.name = "-".join(self.name)

        self.data_dir = args["data_dir"]
        self.metadata_dir = self.args.get("metadata_dir", None)
        if self.metadata_dir is None:
            self.metadata_dir = self.data_dir
        self.data_files = self.args.get("data_files", None)
        self.meta_files = self.args.get("meta_files", None)
        autoload = self.args.get("autoload", False)
        automerge = self.args.get("automerge", False)
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        self.verbose = args.get("verbose", False)

        self._column_info = self.args.get("column_info", {})
        self._column = eKonf.instantiate(self._column_info)

        self._data = None
        self._metadata = None
        self._loaded = False

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.corpora:
                log.info(f"processing {name}")
                args["name"] = name
                args["data_dir"] = self.data_dir
                args["metadata_dir"] = self.metadata_dir
                args["autoload"] = autoload
                args["automerge"] = automerge
                args["use_name_as_subdir"] = use_name_as_subdir
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
            log.info(f">>> Elapsed time: {elapsed()} <<< ")

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname}\n----------\n"
        for name in self.corpora.keys():
            s += f"{str(name)}\n"
        return s

    def __getitem__(self, name):
        return self.corpora[name]

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
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    def load(self):
        for _name in self.corpora:
            self.corpora[_name].load()
        self._loaded = True

    def concatenate(self, append_corpus_name=True):
        self.concat_corpora(append_corpus_name=append_corpus_name)

    def concat_corpora(self, append_corpus_name=True):
        if not self._loaded:
            self.load()

        dfs, df_metas = [], []

        for name in self.corpora:
            df = self.corpora[name]._data
            if df is None:
                self.load()
            if append_corpus_name:
                df = self.COLUMN.append_corpus(df, name)
            dfs.append(df)
            df_meta = self.corpora[name]._metadata
            if df_meta is not None:
                if append_corpus_name:
                    df_meta = self.COLUMN.append_corpus(df_meta, name)
                df_metas.append(df_meta)
        self._data = pd.concat(dfs, ignore_index=True)
        if len(df_metas) > 0:
            self._metadata = pd.concat(df_metas, ignore_index=True)
        if self.verbose:
            log.info(f"concatenated {len(dfs)} corpora")
            print(self._data.head())

    def __iter__(self):
        for corpus in self.corpora.values():
            yield corpus

    def __getitem__(self, name):
        if name not in self.corpora:
            raise KeyError(f"{name} not in corpora")
        return self.corpora[name]

    def __len__(self):
        return len(self.corpora)
