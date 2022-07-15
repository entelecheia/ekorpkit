import logging
import pandas as pd
from ekorpkit import eKonf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ekorpkit.models.snorkel.labeling import (
    LabelingFunction,
    PandasLFApplier,
    LFAnalysis,
)
from ekorpkit.models.snorkel.labeling.model import LabelModel
from ekorpkit.models.snorkel.analysis import metric_score


log = logging.getLogger(__name__)


class BaseSnorkel:
    ABSTAIN = -1

    def __init__(self, data=None, **args):
        self._data = data
        args = eKonf.to_config(args)
        self.args = args
        self.verbose = args.get("verbose", False)
        self._path = args.path
        self._columns = args.columns
        self.ABSTAIN = args.get("ABSTAIN", -1)
        self._worker_lfs = None
        self._le = None
        self._classes = None
        self._train_data = None
        self._test_data = None
        self.L_train = None
        self.L_test = None
        self.label_model = None
        self.Y_train = None
        self.Y_test = None
        self.applier = None
        self._preds = None

    def load_data(self, data=None, test_size=0.2, random_state=12345, shuffle=True):
        if data is None:
            data = self._data

        if isinstance(data, pd.DataFrame):
            _data = data.copy()
        else:
            if eKonf.is_instantiatable(data):
                data = eKonf.instantiate(data)
            _data = data.data.copy()

        if _data is None:
            raise ValueError("No data to process")

        _columns = self.columns
        le = preprocessing.LabelEncoder()
        _labels = _data[_columns.labels]
        le.fit(_labels)
        _data[_columns.classes] = le.transform(_labels)

        _train_data, _test_data = train_test_split(
            _data, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        log.info(f"Train data: {_train_data.shape}, Test data: {_test_data.shape}")
        if self.verbose:
            print("Train data:", _train_data.shape)
            print(_train_data.head())
            print("Test data:", _test_data.shape)
            print(_test_data.head())

        self._le = le
        self._classes = le.classes_
        self._data = _data
        self._train_data = _train_data
        self._test_data = _test_data

    @property
    def columns(self):
        return self._columns

    @property
    def data(self):
        return self._data

    @property
    def classes(self):
        return self._classes

    @property
    def train_data(self):
        return self._train_data

    @property
    def test_data(self):
        return self._test_data

    def compose_worker_lfs(self):
        labels_by_annotator = self.data.groupby(self.columns.annotator)
        worker_dicts = {}
        for worker_id in labels_by_annotator.groups:
            worker_df = labels_by_annotator.get_group(worker_id)
            worker_dicts[worker_id] = dict(zip(worker_df.id, worker_df.classes))

        log.info(f"Number of workers: {len(worker_dicts)}")

        def worker_lf(x, worker_dict):
            return worker_dict.get(x.id, self.ABSTAIN)

        def make_worker_lf(worker_id):
            worker_dict = worker_dicts[worker_id]
            name = f"worker_{worker_id}"
            return LabelingFunction(
                name, f=worker_lf, resources={"worker_dict": worker_dict}
            )

        worker_lfs = [make_worker_lf(worker_id) for worker_id in worker_dicts]
        self._worker_lfs = worker_lfs
        return worker_lfs

    def apply_worker_lfs(self, worker_lfs=None):
        if worker_lfs is None:
            worker_lfs = self._worker_lfs
        applier = PandasLFApplier(worker_lfs)
        self.applier = applier
        log.info(f"Applying worker lfs to train data")
        self.L_train = applier.apply(self.train_data)
        log.info(f"Applying worker lfs to test data")
        self.L_test = applier.apply(self.test_data)
        self.Y_train = self.train_data[self.columns.classes].values
        self.Y_test = self.test_data[self.columns.classes].values

    def lf_summary(self, worker_lfs=None, L_train=None, Y_train=None):
        L_train = L_train or self.L_train
        Y_train = Y_train or self.Y_train

        if worker_lfs is None:
            worker_lfs = self._worker_lfs
        log.info(
            f"Training set coverage: {100 * LFAnalysis(L_train).label_coverage(): 0.1f}%"
        )
        return LFAnalysis(L_train, worker_lfs).lf_summary(Y_train)

    def fit(
        self,
        L_train=None,
        cardinality=None,
        verbose=False,
        n_epochs=100,
        seed=12345,
        log_freq=20,
        l2=0.1,
        lr=0.01,
        progress_bar=True,
        **args,
    ):
        L_train = L_train or self.L_train
        if cardinality is None:
            cardinality = len(self.classes)
        # Train LabelModel.
        self.label_model = LabelModel(cardinality=cardinality, verbose=verbose)
        self.label_model.fit(
            L_train,
            n_epochs=n_epochs,
            seed=seed,
            log_freq=log_freq,
            l2=l2,
            lr=lr,
            progress_bar=progress_bar,
        )

    def eval(
        self,
        L_train=None,
        Y_train=None,
        L_test=None,
        Y_test=None,
        tie_break_policy="abstain",
        metric="accuracy",
    ):
        L_test = L_test or self.L_test
        Y_test = Y_test or self.Y_test
        L_train = L_train or self.L_train
        Y_train = Y_train or self.Y_train

        preds_train = self.label_model.predict(
            L_train, tie_break_policy=tie_break_policy
        )
        acc = metric_score(Y_train, preds_train, probs=None, metric=metric)
        print(f"LabelModel Accuracy for train: {acc:.3f}")

        preds_test = self.label_model.predict(L_test, tie_break_policy=tie_break_policy)
        acc = metric_score(Y_test, preds_test, probs=None, metric=metric)
        print(f"LabelModel Accuracy for test: {acc:.3f}")

    def predict(self, data=None, columns=None, tie_break_policy="abstain"):
        data = data or self.data.copy()
        columns = columns or self.columns
        L_data = self.applier.apply(data)

        _snorkel_classes = columns["snorkel_classes"]
        _classes = columns["classes"]
        _labels = columns["labels"]

        data[_snorkel_classes] = self.label_model.predict(
            L_data, tie_break_policy=tie_break_policy
        )
        data = data[data.snorkel_classes != self.ABSTAIN]

        data[_labels] = self._le.inverse_transform(data[_snorkel_classes])
        self._preds = data
        if self.verbose:
            print("Data that predictions are different from classes:")
            print(data.query(f"{_classes} != {_snorkel_classes}"))
        return data

    def save_preds(self, data, columns=None, **kwargs):
        if data is None:
            data = self._preds
        if data is None:
            raise ValueError("No predictions to save")
        if isinstance(columns, list):
            data = data[columns].drop_duplicates()

        args = eKonf.to_dict(eKonf.merge(self._path.output, kwargs))
        eKonf.save_data(data, **args)
