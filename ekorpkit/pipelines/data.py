from ekorpkit import eKonf
from ekorpkit.datasets.corpus import Corpus
from ekorpkit.datasets.corpora import Corpora
from ekorpkit.datasets.dataset import Dataset
from ekorpkit.datasets.datasets import Datasets


class Data:
    def __init__(self, data=None, **args):

        args = eKonf.to_config(args)
        self.args = args
        self._data = None
        self._column = None
        self._name = None
        self.verbose = args.get(eKonf.Keys.VERBOSE, False)

        if eKonf.is_dataframe(data):
            self._data = data
        elif eKonf.is_config(args):
            self.load(**args)

        self._column_info = self.args.column_info
        if self._column_info is None:
            raise ValueError("Column info can't be None")
        if self._column is None:
            self._column = eKonf.instantiate(self._column_info)

    def load(self, **args):
        _corpus = args.get(eKonf.Keys.CORPUS) or {}
        _dataset = args.get(eKonf.Keys.DATASET) or {}
        _path = args.get(eKonf.Keys.PATH) or {}

        if _corpus.get(eKonf.Keys.NAME) and _dataset.get(eKonf.Keys.NAME) is None:
            args = _corpus
        elif _dataset.get(eKonf.Keys.NAME):
            args = _dataset

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
            self._name = data.name
        elif _path:
            self._data = eKonf.load_data(**_path[eKonf.Keys.DATA])
            self._name = _path[eKonf.Keys.DATA][eKonf.Keys.FILE]

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}: {self._name}{type(self._data)}"

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
