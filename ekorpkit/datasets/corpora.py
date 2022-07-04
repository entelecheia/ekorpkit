import pandas as pd
import logging
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from .corpus import Corpus
from .base import BaseSet

log = logging.getLogger(__name__)


class Corpora(BaseSet):
    def __init__(self, **args):
        super().__init__(**args)
        self.corpora = args.get("corpora")
        self.corpora = eKonf.to_dict(self.corpora)
        if self.corpora is None:
            self.corpora = self.name
        if isinstance(self.corpora, str):
            self.corpora = {self.corpora: None}
        elif eKonf.is_list(self.corpora):
            self.corpora = {name: None for name in self.corpora}
        # if eKonf.is_list(self.name):
        #     self.name = "-".join(self.name)
        if self.name is None and eKonf.is_dict(self.corpora):
            self.name = "-".join(self.corpora.keys())

        self.metadata_dir = self.args.get("metadata_dir", None)
        if self.metadata_dir is None:
            self.metadata_dir = self.data_dir
        self.meta_files = self.args.get("meta_files", None)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self.load_column_info()

        self._metadata = None

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.corpora:
                log.info(f"processing {name}")
                args["name"] = name
                args["data_dir"] = self.data_dir
                args["metadata_dir"] = self.metadata_dir
                args["auto"] = self.auto
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
    def TEXT(self):
        return self.COLUMN.TEXT

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

    def concat_corpora(self, append_corpus_name=True):
        self.concatenate(append_name=append_corpus_name)

    def concatenate(self, append_name=True):
        if not self._loaded:
            self.load()

        dfs, df_metas = [], []

        for name in self.corpora:
            df = self.corpora[name].data
            if df is None:
                self.load()
            if append_name:
                df = self.COLUMN.append_corpus(df, name)
            dfs.append(df)
            df_meta = self.corpora[name]._metadata
            if df_meta is not None:
                if append_name:
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
