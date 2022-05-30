import logging
import os
import sklearn
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf
from ekorpkit.io.file import save_dataframe


log = logging.getLogger(__name__)


class SimpleTrainer:
    __metaclass__ = ABCMeta

    def __init__(self, **args):
        args = eKonf.to_dict(args)
        self.args = args
        self.name = args["name"]
        self.verbose = args.get("verbose", True)
        self._model_cfg = args["config"]
        self._model_eval = args.get("model", {}).get("eval")
        self._dataset = args.get(eKonf.Keys.DATASET, None)
        self._to_predict = args["to_predict"]
        self._to_train = args["to_train"]
        self._method_ = self.args.get("_method_")
        self._pred_output_dir = args["pred_output_dir"]
        self._pred_output_file = args["pred_output_file"]

        self._labels_key = self._to_train.get("labels")

        self._predicted_key = "predicted"
        self._model_outputs_key = "model_outputs"

        os.makedirs(args["output_dir"], exist_ok=True)
        os.makedirs(args["cache_dir"], exist_ok=True)
        os.makedirs(args["pred_output_dir"], exist_ok=True)
        os.makedirs(args["result_dir"], exist_ok=True)

        self.model = None
        self.dataset = None
        self.splits = {}
        self.train_data = None
        self.eval_data = None
        self.test_data = None

        eKonf.methods(self._method_, self)

    @abstractmethod
    def train(self):
        raise NotImplementedError("Must override train")

    @abstractmethod
    def _predict(self, to_predict: list):
        raise NotImplementedError("Must override predict")

    def load_datasets(self):
        if self._dataset is None:
            log.warning("No dataset config found")
            return
        self.dataset = eKonf.instantiate(self._dataset)
        self.splits = self.dataset.splits

        self.train_data = self.splits[self.dataset.SPLITS.TRAIN]
        if self.verbose:
            print("Train data:")
            print(self.train_data.info())
            print(self.train_data.tail())
        if "dev" in self.splits:
            self.eval_data = self.splits[self.dataset.SPLITS.DEV]
            self._model_cfg["evaluate_during_training"] = True
            if self.verbose:
                print("Eval data:")
                print(self.eval_data.info())
                print(self.eval_data.tail())
        else:
            self.eval_data = None
            self._model_cfg["evaluate_during_training"] = False
        self.test_data = self.splits[self.dataset.SPLITS.TEST]
        if self.verbose:
            print("Test data:")
            print(self.test_data.info())
            print(self.test_data.tail())

    def convert_to_predict(self, df):
        input_key = self._to_predict["input"]
        to_predict = df[input_key].tolist()
        if self.verbose:
            print(to_predict[:5])
        return to_predict

    def append_predictions(self, df, preds):
        predicted_column = self._to_predict[self._predicted_key]
        model_outputs_column = self._to_predict[self._model_outputs_key]
        df[predicted_column] = preds[self._predicted_key]
        df[model_outputs_column] = preds[self._model_outputs_key]
        return df

    def predict(self, df, _to_predict={}):
        if _to_predict:
            self._to_predict = _to_predict
        to_predict = self.convert_to_predict(df)
        preds = self._predict(to_predict)
        df = self.append_predictions(df, preds)
        return df

    def eval(self):
        if not self.splits:
            self.load_datasets()

        if self.test_data is None:
            log.warning("No test data found")
            return

        to_predict = self.convert_to_predict(self.test_data)
        preds = self._predict(to_predict)
        self.pred_data = self.append_predictions(self.test_data, preds)
        pred_filepath = os.path.join(self._pred_output_dir, self._pred_output_file)
        save_dataframe(self.pred_data, pred_filepath)
        if self.verbose:
            print(self.pred_data.head())
        if self._model_eval:
            eKonf.instantiate(self._model_eval, data=self.pred_data)


class SimpleNER(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.ner import NERModel

        args = self.args
        if args.labels is None:
            labels = list(self.train_data[self._labels_key].unique())

        # Create a NERModel
        model = NERModel(
            args.model_type,
            args.model_uri,
            labels=labels,
            cuda_device=args.cuda_device,
            args=eKonf.to_dict(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_data=self.eval_data)

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
        model.train_model(self.train_data, eval_df=self.eval_data)

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

        self._model_cfg["labels_list"] = (
            self.train_data[self._labels_key].unique().tolist()
        )
        args["num_labels"] = len(self._model_cfg["labels_list"])

        # Create a NERModel
        model = ClassificationModel(
            args["model_type"],
            args["model_uri"],
            num_labels=args["num_labels"],
            cuda_device=args["cuda_device"],
            args=self._model_cfg,
        )

        # Train the model
        model.train_model(
            self.train_data, eval_df=self.eval_data, acc=sklearn.metrics.accuracy_score
        )

        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(
            self.test_data, acc=sklearn.metrics.accuracy_score
        )
        if self.verbose:
            print(f"Evaluation result: {result}")
            print(f"Wrong predictions: {wrong_predictions[:5]}")
            # print(f"Model outputs: {model_outputs[:5]}")
            print(f"num_outputs: {len(model_outputs)}")
            print(f"num_wrong_predictions: {len(wrong_predictions)}")

    def load_model(self, model_dir=None):
        from simpletransformers.classification import ClassificationModel

        if model_dir is None:
            model_dir = self.args["best_model_dir"]

        self.model = ClassificationModel(
            self.args["model_type"], model_dir, args=self._model_cfg
        )
        log.info(f"Loaded model from {model_dir}")

    def _predict(self, to_predict: list):
        if self.model is None:
            self.load_model()

        predictions, raw_outputs = self.model.predict(to_predict)
        log.info(f"type of raw_outputs: {type(raw_outputs)}")
        raw_outputs= [output.flatten().tolist() for output in raw_outputs]
        log.info(f"raw_output: {raw_outputs[0]}")
        return {
            self._predicted_key: predictions,
            self._model_outputs_key: raw_outputs,
        }
