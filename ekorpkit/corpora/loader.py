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
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        self.column_info = self.args.get("column_info", None)
        if self.column_info:
            self._keys = self.column_info["keys"]
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
        for name in self.corpora:
            df = self.corpora[name]._data
            if append_corpus_name:
                df["corpus"] = name
            dfs.append(df)
        self._data = pd.concat(dfs, ignore_index=True)

    def __iter__(self):
        for corpus in self.corpora.values():
            yield corpus

    def __getitem__(self, name):
        if name not in self.corpora:
            raise KeyError(f"{name} not in corpora")
        return self.corpora[name]


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
