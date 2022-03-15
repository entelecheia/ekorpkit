# from hydra.utils import instantiate
# from wasabi import msg
from ekorpkit.pipelines.pipe import apply_pipeline
import pandas as pd
from omegaconf import OmegaConf
from ekorpkit.utils.func import elapsed_timer
from .corpus import Corpus


class Corpora:
    def __init__(self, **args):
        args = OmegaConf.create(args)
        self.args = args
        self.names = args.name
        if isinstance(self.names, str):
            self.names = [self.names]
        self.data_dir = args.data_dir
        self.corpora = {}
        self._data = None
        self._metadata = None
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        self.verbose = args.get("verbose", False)
        self.column_info = self.args.get("column_info", None)
        if self.column_info:
            self._keys = self.column_info["keys"]
            if self._keys is not None:
                for k in ["id", "text"]:
                    if isinstance(self._keys[k], str):
                        self._keys[k] = [self._keys[k]]
                    else:
                        self._keys[k] = list(self._keys[k])
                self._id_keys = self._keys["id"]
            else:
                self._id_keys = ["id"]
            self._data_keys = self.column_info.get("data", None)
            self._meta_kyes = self.column_info.get("meta", None)
        self._corpus_key = "corpus"
        self._text_key = "text"
        self._id_key = "id"
        self._id_separator = "_"

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.names:
                print(f"processing {name}")
                if use_name_as_subdir:
                    data_dir = f"{self.data_dir}/{name}"
                else:
                    data_dir = self.data_dir
                args["data_dir"] = data_dir
                args["name"] = name
                corpus = Corpus(**args)
                self.corpora[name] = corpus
            print(f"\n >>> Elapsed time: {elapsed()} <<< ")

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

    def __iter__(self):
        for corpus in self.corpora.values():
            yield corpus

    def __getitem__(self, name):
        if name not in self.corpora:
            raise KeyError(f"{name} not in corpora")
        return self.corpora[name]

    def do_tasks(self, pipeline=None, **kwargs):
        verbose = kwargs.get("verbose", self.verbose)
        merge_metadata = kwargs.get("merge_metadata", False)
        if merge_metadata and self._metadata is not None:
            df = pd.merge(self._metadata, self._data, on=self._id_keys)
        else:
            df = self._data
        # update_args = {"corpus_name": self.name}
        _pipeline_ = pipeline.get("_pipeline_", {})
        df = apply_pipeline(df, _pipeline_, pipeline)
        if verbose and df is not None:
            print(df.head())


def do_corpus_tasks(corpus, pipeline=None, **kwargs):
    # verbose = kwargs.get("verbose", False)
    merge_metadata = kwargs.get("merge_metadata", False)
    if merge_metadata:
        df = pd.merge(corpus._metadata, corpus._data, on=corpus._id_key)
    else:
        df = corpus._data
    update_args = {"corpus_name": corpus.name}
    _pipeline_ = pipeline.get("_pipeline_", {})
    df = apply_pipeline(df, _pipeline_, pipeline, update_args=update_args)
