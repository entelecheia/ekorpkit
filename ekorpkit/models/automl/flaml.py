import logging
import os
import sklearn
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf
from ekorpkit.io.file import save_dataframe


log = logging.getLogger(__name__)


class AutoML:
    __metaclass__ = ABCMeta

    def __init__(self, **args):
        from flaml import AutoML

        args = eKonf.to_dict(args)
        self.args = args
        self.name = args["name"]
        self.verbose = args.get("verbose", True)
        self._model_cfg = args["config"]
        self._model_eval = args.get("model", {}).get("eval")
        self._dataset_cfg = args.get(eKonf.Keys.DATASET, None)
        self._to_predict = args["to_predict"]
        self._method_ = self.args.get("_method_")
        self._output_dir = args["output_dir"]
        self._model_file = os.path.join(self._output_dir, args["model_file"])
        self._log_file = args["log_file"]
        self._pred_output_file = args["pred_output_file"]

        os.makedirs(self._output_dir, exist_ok=True)

        self._X_key = "X"
        self._y_pred_key = "y_preds"
        self._y_prob_key = "y_probs"

        self._automl = AutoML()
        self._dataset = None
        self._X_train = None
        self._X_dev = None
        self._X_test = None
        self._y_train = None
        self._y_dev = None
        self._y_test = None

        eKonf.methods(self._method_, self)

    def fit(self):
        if self._X_train is None:
            self.load_dataset()

        self._automl.fit(
            X_train=self._X_train.values,
            y_train=self._y_train.values,
            **self._model_cfg,
        )
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
        y_preds = self._automl.predict(X)
        y_probs = self._automl.predict_proba(X)
        y_probs = y_probs.flatten().tolist()
        return {self._y_pred_key: y_preds, self._y_prob_key: y_probs}

    def convert_to_X(self, data):
        X_cols = self._to_predict[self._X_key]
        if X_cols is None:
            X_cols = list(data.columns)
        X = data[X_cols].values
        if self.verbose:
            print(X[:5])
        return X

    def append_predictions(self, data, preds):
        y_pred_column = self._to_predict[self._y_pred_key]
        y_prob_column = self._to_predict[self._y_prob_key]
        data[y_pred_column] = preds[self._y_pred_key]
        data[y_prob_column] = preds[self._y_prob_key]
        return data

    def predict(self, data, _to_predict={}):
        if _to_predict:
            self._to_predict = _to_predict
        X = self.convert_to_X(data)
        preds = self._predict(X)
        data = self.append_predictions(data, preds)
        return data

    def eval(self):
        from flaml.ml import sklearn_metric_loss_score

        if self._X_test is None:
            self.load_dataset()

        if self._X_test is None:
            log.warning("No test data found")
            return

        y_preds, y_probs = self._predict(self._X_test.values())
        self._pred_data = self.append_predictions(self._X_test, y_preds)
        pred_filepath = os.path.join(self._output_dir, self._pred_output_file)
        save_dataframe(self._pred_data, pred_filepath)
        if self.verbose:
            print(self._pred_data.head())
        if self._model_eval:
            eKonf.instantiate(self._model_eval, data=self._pred_data)
        """ compute different metric values on testing dataset"""
        print(
            "accuracy",
            "=",
            1 - sklearn_metric_loss_score("accuracy", y_preds, self._y_test),
        )
        print(
            "roc_auc",
            "=",
            1 - sklearn_metric_loss_score("roc_auc", y_probs, self._y_test),
        )
        print(
            "log_loss",
            "=",
            sklearn_metric_loss_score("log_loss", y_probs, self._y_test),
        )

    def load_dataset(self):
        if self._dataset_cfg is None:
            log.warning("No dataset config found")
            return
        self._dataset = eKonf.instantiate(self._dataset_cfg)

        self._X_train = self._dataset.X_train
        self._X_dev = self._dataset.X_dev
        self._X_test = self._dataset.X_test
        self._y_train = self._dataset.y_train
        self._y_dev = self._dataset.y_dev
        self._y_test = self._dataset.y_test

        if self.verbose:
            print("Train data:")
            print(self._X_train.info())
            print(self._X_train.tail())
            if self._X_dev is not None:
                print("Eval data:")
                print(self._X_dev.info())
                print(self._X_dev.tail())
            else:
                log.info("No eval data found")
            if self._X_test is not None:
                print("Test data:")
                print(self._X_test.info())
                print(self._X_test.tail())
            else:
                log.info("No test data found")
