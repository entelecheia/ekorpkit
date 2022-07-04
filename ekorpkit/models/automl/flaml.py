import pandas as pd
import logging
import os
import sklearn
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class AutoML:
    __metaclass__ = ABCMeta
    Keys = eKonf.Keys

    def __init__(self, **args):
        from flaml import AutoML

        self._dataset_cfg = args.pop(eKonf.Keys.DATASET, None)

        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.verbose = args.get("verbose", True)
        self._model_cfg = eKonf.to_dict(args.config)
        self._columns = args.get(eKonf.Keys.COLUMNS)
        self._train_ = args[eKonf.Keys.TRAIN]
        self._predict_ = args[eKonf.Keys.PREDICT]
        self._method_ = self.args.get(eKonf.Keys.METHOD)
        self._eval_cfg = args.model.eval
        self._learning_curve = args.learning_curve.visualize.plot
        self._feature_importance = args.feature_importance.visualize.plot

        self._path = self.args.path
        self._model_file = self._path.model.filepath
        self._log_file = self._path.log.filepath
        self._pred_file = self._path.pred.filepath

        self._automl = AutoML()
        self._dataset = None
        self._X_train = None
        self._X_dev = None
        self._X_test = None
        self._y_train = None
        self._y_dev = None
        self._y_test = None
        self._classes = None
        self.pred_data = None

        eKonf.methods(self._method_, self)

    def fit(self):
        if self.X_train is None:
            self.load_dataset()

        X_train = self.X_train.values
        y_train = self.dataset.transform_labels(self.y_train.values)
        log.info(f"types of X_train: {type(X_train)}, y_train: {type(y_train)}")
        log.info(f"fitting a model with {self._model_cfg}")
        self._automl.fit(X_train=X_train, y_train=y_train, **self._model_cfg)
        # Print the results
        self.show_results()

    def save(self):
        """pickle and save the automl object"""
        import pickle

        with open(self._model_file, "wb") as f:
            pickle.dump(self._automl, f, pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {self._model_file}")

    def load(self):
        import pickle

        with open(self._model_file, "rb") as f:
            self._automl = pickle.load(f)
        log.info(f"Loaded model from {self._model_file}")

    @property
    def dataset(self):
        return self._dataset

    @property
    def classes(self):
        return self._classes

    @property
    def best_estimator(self):
        return self._automl.model.estimator

    def show_results(self):
        """retrieve best config and best learner"""
        print("Best ML leaner:", self._automl.best_estimator)
        print("Best hyperparmeter config:", self._automl.best_config)
        print(
            "Best accuracy on validation data: {0:.4g}".format(
                1 - self._automl.best_loss
            )
        )
        print(
            "Training duration of best run: {0:.4g} s".format(
                self._automl.best_config_train_time
            )
        )

    def get_logs(self, time_budget=240):
        from flaml.data import get_output_from_log

        if not eKonf.exists(self._log_file):
            log.warning(f"Log file {self._log_file} not found")
            return None
        (
            time_history,
            best_valid_loss_history,
            valid_loss_history,
            config_history,
            metric_history,
        ) = get_output_from_log(filename=self._log_file, time_budget=time_budget)
        return {
            "time_history": time_history,
            "best_valid_loss_history": best_valid_loss_history,
            "valid_loss_history": valid_loss_history,
            "config_history": config_history,
            "metric_history": metric_history,
        }

    def _predict(self, X):
        """compute predictions of testing dataset"""
        y_pred = self._automl.predict(X)
        y_pred = self.dataset.inverse_transform_labels(y_pred)
        return {self.Keys.PREDICTED: y_pred}

    def convert_to_X(self, data):
        X_cols = self._predict_[self.Keys.X]
        if X_cols is None:
            X_cols = list(data.columns)
        X = data[X_cols].values
        if self.verbose:
            print(X[:5])
        return X

    def append_predictions(self, data, preds):
        predicted_column = self._predict_[self.Keys.PREDICTED]
        data = data.copy()
        data[predicted_column] = preds[self.Keys.PREDICTED]
        return data

    def predict(self, data, _predict_={}):
        if _predict_:
            self._predict_ = _predict_
        X = self.convert_to_X(data)
        preds = self._predict(X)
        return self.append_predictions(data, preds)

    def eval(self):
        from flaml.ml import sklearn_metric_loss_score

        if self.X_test is None:
            self.load_dataset()

        if self.X_test is None:
            log.warning("No test data found")
            return

        y_preds = self._predict(self.X_test.values)
        pred_data = self.append_predictions(self.X_test, y_preds)
        actual_column = self._eval_cfg._eval_[self.Keys.ACTUAL]
        if actual_column:
            pred_data[actual_column] = self.y_test
        eKonf.save_data(pred_data, self._pred_file)
        if self.verbose:
            print(pred_data.head())

        y_pred = self.dataset.transform_labels(y_preds[self.Keys.PREDICTED])
        y_test = self.dataset.transform_labels(self.y_test)
        print("r2:", 1 - sklearn_metric_loss_score("r2", y_pred, y_test))
        print("mse:", sklearn_metric_loss_score("mse", y_pred, y_test))
        print("mae:", sklearn_metric_loss_score("mae", y_pred, y_test))

        self._eval_cfg.labels = self.classes
        eKonf.instantiate(self._eval_cfg, data=pred_data)
        self.pred_data = pred_data

    def load_dataset(self):
        if self._dataset_cfg is None:
            log.warning("No dataset config found")
            return
        if eKonf.is_instantiatable(self._dataset_cfg):
            self._dataset = eKonf.instantiate(self._dataset_cfg)
        else:
            self._dataset = self._dataset_cfg
        self._dataset.fit_labelencoder(self._dataset.y_train)
        self._classes = self._dataset.classes

        self._X_train = self._dataset.X_train
        self._X_dev = self._dataset.X_dev
        self._X_test = self._dataset.X_test
        self._y_train = self._dataset.y_train
        self._y_dev = self._dataset.y_dev
        self._y_test = self._dataset.y_test

        if self.verbose:
            print("Train data:")
            print(self._X_train.tail())
            if self._X_dev is not None:
                print("Eval data:")
                print(self._X_dev.tail())
            else:
                log.info("No eval data found")
            if self._X_test is not None:
                print("Test data:")
                print(self._X_test.tail())
            else:
                log.info("No test data found")

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_dev(self):
        return self._X_dev

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_train(self):
        return self._y_train

    @property
    def y_dev(self):
        return self._y_dev

    @property
    def y_test(self):
        return self._y_test

    def get_feature_importance(self, estimator=None, n_features=None):
        if estimator is None:
            estimator = self.best_estimator
        if self.X_train is None:
            self.load_dataset()
        _data = {
            "columns": self.X_train.columns.tolist(),
            "importances": estimator.feature_importances_.tolist(),
        }
        data = pd.DataFrame(_data)
        data.sort_values(by="importances", ascending=False, inplace=True)
        if n_features is not None:
            data = data.head(n_features)

        return data

    def plot_feature_importance(self, estimator=None, n_features=None):
        data = self.get_feature_importance(estimator=estimator, n_features=n_features)

        eKonf.instantiate(self._feature_importance, data=data)

    def get_log_data(self):
        train_logs = self.get_logs()

        data = pd.DataFrame(train_logs)
        data["acc_history"] = 1 - data["valid_loss_history"]
        data["best_acc_history"] = 1 - data["best_valid_loss_history"]
        return data

    def plot_learning_curve(self):
        data = self.get_log_data()

        eKonf.instantiate(self._learning_curve, data=data)
