import numpy as np
import pandas as pd
import logging
import sklearn
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class SimpleTrainer:
    __metaclass__ = ABCMeta
    Keys = eKonf.Keys

    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.verbose = args.get("verbose", False)
        self._model_cfg = eKonf.to_dict(args.config)
        self._dataset = args.get(eKonf.Keys.DATASET)
        self._columns = args.get(eKonf.Keys.COLUMNS)
        self._train_ = args[eKonf.Keys.TRAIN]
        self._predict_ = args[eKonf.Keys.PREDICT]
        self._method_ = self.args.get(eKonf.Keys.METHOD)
        self._eval_cfg = args.model.eval

        self._path = self.args.path
        self._pred_file = args.path.pred.filepath

        self.model = None
        self.dataset = None
        self.splits = {}
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.label_list = None
        self.labels_map = None

        eKonf.methods(self._method_, self)

    @abstractmethod
    def train(self):
        raise NotImplementedError("Must override train")

    @abstractmethod
    def _predict(self, data: list):
        raise NotImplementedError("Must override predict")

    def load_datasets(self):
        if self._dataset is None:
            log.warning("No dataset config found")
            return
        self.dataset = eKonf.instantiate(self._dataset)
        self.splits = self.dataset.splits

        self.train_data = self.dataset.train_data
        self.dev_data = self.dataset.dev_data
        self.test_data = self.dataset.test_data

        if self.verbose:
            print("Train data:")
            print(self.train_data.tail())
        if self.dev_data is not None:
            self._model_cfg["evaluate_during_training"] = True
            if self.verbose:
                print("Eval data:")
                print(self.dev_data.tail())
        else:
            self._model_cfg["evaluate_during_training"] = False
        if self.test_data is not None and self.verbose:
            print("Test data:")
            print(self.test_data.tail())

    def convert_to_train(self):
        return (
            self.rename_columns(self.train_data, self._train_),
            self.rename_columns(self.dev_data, self._train_),
            self.rename_columns(self.test_data, self._train_),
        )

    def rename_columns(self, data, columns):
        if not columns or data is None:
            log.info("No columns or data to rename")
            return data
        renames = {
            name: key
            for key, name in columns.items()
            if name and name != key and name in data.columns
        }
        log.info(f"Renaming columns: {renames}")
        if renames:
            data = data.copy().rename(columns=renames)
        if self.verbose:
            print(data.head())
        return data

    def convert_to_predict(self, data):
        input_col = self._predict_[self.Keys.INPUT]
        data_to_predict = data[input_col].tolist()
        if self.verbose:
            print(data_to_predict[:5])
        return data_to_predict

    def append_predictions(self, data, preds):
        predicted_column = self._predict_[self.Keys.PREDICTED]
        model_outputs_column = self._predict_[self.Keys.MODEL_OUTPUTS]
        data[predicted_column] = preds[self.Keys.PREDICTED]
        data[model_outputs_column] = preds[self.Keys.MODEL_OUTPUTS]
        pred_probs_column = self._predict_.get(self.Keys.PRED_PROBS)
        if pred_probs_column:
            data[pred_probs_column] = preds[self.Keys.PRED_PROBS]
        return data

    def predict(self, data, _predict_={}):
        if _predict_:
            self._predict_ = _predict_
        data_to_predict = self.convert_to_predict(data)
        preds = self._predict(data_to_predict)
        data = self.append_predictions(data, preds)
        return data

    def cross_val_predict(self, cv=5, dev_size=0.2, random_state=1235, shuffle=True):
        if not self.splits:
            self.load_datasets()
        if shuffle:
            data = self.dataset.data.sample(frac=1).reset_index(drop=True)
        else:
            data = self.dataset.data.reset_index(drop=True)

        splits = np.array_split(data, cv)
        pred_dfs = []
        for i, split in enumerate(splits):
            _data = pd.concat(splits[:i] + splits[i + 1 :])
            _train_data, _dev_data = train_test_split(
                _data, test_size=dev_size, random_state=random_state, shuffle=shuffle
            )
            log.info(f"Train data: {_train_data.shape}, Test data: {_dev_data.shape}")
            self.train_data = _train_data
            self.dev_data = _dev_data
            self.test_data = split

            self.train()
            log.info(f"Predicting split {i}")
            pred_df = self.predict(split)
            pred_dfs.append(pred_df)
        return pd.concat(pred_dfs)

    def eval(self):
        if not self.splits:
            self.load_datasets()

        if self.test_data is None:
            log.warning("No test data found")
            return

        data_to_predict = self.convert_to_predict(self.test_data)
        preds = self._predict(data_to_predict)
        self.pred_data = self.append_predictions(self.test_data, preds)
        eKonf.save_data(self.pred_data, self._pred_file)
        if self.verbose:
            print(self.pred_data.head())
        if self._eval_cfg:
            self._eval_cfg.labels = self.labels_list
            eKonf.instantiate(self._eval_cfg, data=self.pred_data)


class SimpleNER(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.ner import NERModel

        args = self.args
        if args.labels is None:
            labels = list(self.train_data[self.Keys.LABELS].unique())

        # Create a NERModel
        model = NERModel(
            args.model_type,
            args.model_uri,
            labels=labels,
            cuda_device=args.cuda_device,
            args=eKonf.to_dict(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_data=self.dev_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions


class SimpleMultiLabel(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.classification import MultiLabelClassificationModel

        args = self.args
        # Create a Model
        model = MultiLabelClassificationModel(
            args.model_type,
            args.model_uri,
            num_labels=args.num_labels,
            args=eKonf.to_dict(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_df=self.dev_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions


class SimpleClassification(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.classification import ClassificationModel

        args = self.args
        if not self.splits:
            self.load_datasets()
        train_data, dev_data, test_data = self.convert_to_train()

        self._model_cfg["labels_list"] = train_data[self.Keys.LABELS].unique().tolist()
        args.num_labels = len(self._model_cfg["labels_list"])

        # Create a NERModel
        model = ClassificationModel(
            args.model_type,
            args.model_uri,
            num_labels=args.num_labels,
            cuda_device=args.cuda_device,
            args=self._model_cfg,
        )
        self.label_list = model.args.labels_list
        self.labels_map = model.args.labels_map

        # Train the model
        model.train_model(
            train_data, eval_df=dev_data, acc=sklearn.metrics.accuracy_score
        )

        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(
            test_data, acc=sklearn.metrics.accuracy_score
        )
        if self.verbose:
            print(f"Evaluation result: {result}")
            print(f"Wrong predictions: {wrong_predictions[:5]}")
            print(f"num_outputs: {len(model_outputs)}")
            print(f"num_wrong_predictions: {len(wrong_predictions)}")

    def load_model(self, model_dir=None):
        from simpletransformers.classification import ClassificationModel

        if model_dir is None:
            model_dir = self.args.config.best_model_dir

        self.model = ClassificationModel(self.args.model_type, model_dir)
        # , args=self._model_cfg
        self.labels_list = self.model.args.labels_list
        self.labels_map = self.model.args.labels_map
        log.info(f"Loaded model from {model_dir}")

    def _predict(self, data: list):
        if self.model is None:
            self.load_model()

        predictions, raw_outputs = self.model.predict(data)
        log.info(f"type of raw_outputs: {type(raw_outputs)}")
        raw_outputs = [output.flatten().tolist() for output in raw_outputs]
        pred_probs = [softmax(output).max() for output in raw_outputs]
        log.info(f"raw_output: {raw_outputs[0]}")
        return {
            self.Keys.PREDICTED.value: predictions,
            self.Keys.PRED_PROBS.value: pred_probs,
            self.Keys.MODEL_OUTPUTS.value: raw_outputs,
        }
