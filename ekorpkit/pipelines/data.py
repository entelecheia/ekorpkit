import os
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.corpora import Corpus, Corpora
from ekorpkit.datasets import Dataset, Datasets
from ekorpkit.io.file import load_dataframe


class Data:
    def __init__(self, data=None, **args):

        self.args = args
        self._data = None
        self._column = None
        self.verbose = args.get("verbose", False)

        if isinstance(data, pd.DataFrame):
            self._data = data
        elif eKonf.is_config(args):
            self._load(**args)

        _data = eKonf.function(args, "concat_dataframes", data=self._data)
        if _data is not None:
            self._data = _data

        self._column_info = self.args.get("column_info")
        if self._column_info is None:
            raise ValueError("Column info can't be None")
        if self._column is None:
            self._column = eKonf.instantiate(self._column_info)

    def _load(self, **args):
        corpus = args.get(eKonf.Keys.CORPUS) or {}
        dataset = args.get(eKonf.Keys.DATASET) or {}
        data_dir = args.get("data_dir")
        data_file = args.get("data_file")

        if corpus.get("name") and dataset.get("name") is None:
            args = corpus
        elif dataset.get("name"):
            args = dataset

        if eKonf.is_instantiatable(args):
            data = eKonf.instantiate(args)
            if isinstance(data, (Corpus)):
                self._data = data.data
                self._column = data.COLUMN
            elif isinstance(data, (Corpora)):
                data.concat_corpora()
                self._data = data.data
                self._column = data.COLUMN
            elif isinstance(data, (Dataset, Datasets)):
                self._data = data.splits
                self._column = data.COLUMN
        elif data_dir and data_file:
            self._data = eKonf.function(args, "load_dataframe")

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}:\n{self.args}"

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
    def data(self):
        return self._data
