import logging
from omegaconf import DictConfig
from ekorpkit import eKonf
from snorkel.labeling import (
    LabelingFunction,
    PandasLFApplier,
    LFAnalysis,
)
from snorkel.labeling.model import LabelModel
from snorkel.analysis import metric_score
from ekorpkit.datasets.config import DataframeConfig
from ekorpkit.config import BaseBatchModel


log = logging.getLogger(__name__)


class BaseSnorkel(BaseBatchModel):
    dataset: DataframeConfig = None
    columns: DictConfig = None
    ABSTAIN = -1
    __worker_lfs__ = None
    __L_train__ = None
    __L_test__ = None
    __Y_train__ = None
    __Y_test__ = None
    __label_model__ = None
    __applier__ = None
    __pred_data__ = None

    def __init__(self, config_name: str = "label.snorkel", **args):
        config_group = f"task={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)

    def initialize_configs(self, **kwargs):
        super().initialize_configs(**kwargs)

        self.dataset = DataframeConfig(**self.config.dataset)

    @property
    def data(self):
        return self.dataset.data

    @property
    def train_data(self):
        return self.dataset.train_data

    @property
    def test_data(self):
        return self.dataset.test_data

    @property
    def le_classes(self):
        return self.dataset.le_classes

    def load_datasets(
        self,
        data=None,
        data_files=None,
        data_dir=None,
        test_split_ratio=0.2,
        seed=None,
        shuffle=None,
        encode_labels=None,
        text_column_name=None,
        label_column_name=None,
    ):
        self.dataset.load_datasets(
            data=data,
            data_files=data_files,
            data_dir=data_dir,
            test_size=test_split_ratio,
            seed=seed,
            shuffle=shuffle,
            encode_labels=encode_labels,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )

        if self.data is None:
            raise ValueError("No data to process")

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
        self.__worker_lfs__ = worker_lfs
        return worker_lfs

    def apply_worker_lfs(self, worker_lfs=None):
        if worker_lfs is None:
            worker_lfs = self.__worker_lfs__
        applier = PandasLFApplier(worker_lfs)
        self.__applier__ = applier
        log.info("Applying worker lfs to train data")
        self.__L_train__ = applier.apply(self.train_data)
        log.info("Applying worker lfs to test data")
        self.__L_test__ = applier.apply(self.test_data)
        self.__Y_train__ = self.train_data[self.columns.classes].values
        self.__Y_test__ = self.test_data[self.columns.classes].values

    def lf_summary(self, worker_lfs=None, L_train=None, Y_train=None):
        L_train = L_train or self.__L_train__
        Y_train = Y_train or self.__Y_train__

        if worker_lfs is None:
            worker_lfs = self.__worker_lfs__
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
        L_train = L_train or self.__L_train__
        if cardinality is None:
            cardinality = len(self.le_classes)
        # Train LabelModel.
        self.__label_model__ = LabelModel(cardinality=cardinality, verbose=verbose)
        self.__label_model__.fit(
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
        L_test = L_test or self.__L_test__
        Y_test = Y_test or self.__Y_test__
        L_train = L_train or self.__L_train__
        Y_train = Y_train or self.__Y_train__

        preds_train = self.__label_model__.predict(
            L_train, tie_break_policy=tie_break_policy
        )
        acc = metric_score(Y_train, preds_train, probs=None, metric=metric)
        print(f"LabelModel Accuracy for train: {acc:.3f}")

        preds_test = self.__label_model__.predict(
            L_test, tie_break_policy=tie_break_policy
        )
        acc = metric_score(Y_test, preds_test, probs=None, metric=metric)
        print(f"LabelModel Accuracy for test: {acc:.3f}")

    def predict(self, data=None, columns=None, tie_break_policy="abstain"):
        data = data or self.data.copy()
        columns = columns or self.columns
        L_data = self.__applier__.apply(data)

        snorkel_classes = columns["snorkel_classes"]
        classes = columns["classes"]
        labels = columns["labels"]

        data[snorkel_classes] = self.__label_model__.predict(
            L_data, tie_break_policy=tie_break_policy
        )
        data = data[data.snorkel_classes != self.ABSTAIN]

        data[labels] = self.dataset.inverse_transform(data[snorkel_classes])
        self.__pred_data__ = data
        if self.verbose:
            print("Data that predictions are different from classes:")
            print(data.query(f"{classes} != {snorkel_classes}"))
        self.save_preds()
        self.save_config()
        return data

    def save_preds(self, data=None, columns=None, **kwargs):
        if data is None:
            data = self.__pred_data__
        if data is None:
            raise ValueError("No predictions to save")
        if isinstance(columns, list):
            data = data[columns].drop_duplicates()

        eKonf.save_data(data, self.batch.output_file, self.batch_dir, **kwargs)
