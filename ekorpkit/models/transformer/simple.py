import logging
import os
import sklearn
from ekorpkit import eKonf
from ekorpkit.io.file import load_dataframe, save_dataframe


log = logging.getLogger(__name__)


class SimpleTrainer:
    def __init__(self, **args):
        args = eKonf.to_dict(args)
        os.makedirs(args["output_dir"], exist_ok=True)
        os.makedirs(args["cache_dir"], exist_ok=True)
        os.makedirs(args["pred_output_dir"], exist_ok=True)
        os.makedirs(args["result_dir"], exist_ok=True)
        self.args = args
        self._dataset = args.get("dataset", None)
        self.model_cfg = args["config"]
        self._to_predict = args["to_predict"]
        self._to_train = args["to_train"]
        self.verbose = args.get("verbose", True)
        self._call = self.args.get("call")

        self.input_key = self._to_predict["input"]
        self.predicted_key = self._to_predict["predicted"]
        self.labels_key = self._to_train["labels"]

        self.model = None
        self.splits = {}
        self.pred_data = {}
        self.to_predict = {}

        eKonf.call(self._call, self)

    def load_datasets(self):
        if self._dataset is None:
            log.warning("No dataset config found")
            return
        dataset = eKonf.instantiate(self._dataset)
        self.splits = dataset.splits

        self.train_data = self.splits["train"]
        if self.verbose:
            print(self.train_data.info())
            print(self.train_data.tail())
        if "dev" in self.splits:
            self.eval_data = self.splits["dev"]
            self.model_cfg["evaluate_during_training"] = True
            if self.verbose:
                print(self.eval_data.info())
                print(self.eval_data.tail())
        else:
            self.eval_data = None
            self.model_cfg["evaluate_during_training"] = False
        self.test_data = self.splits["test"]
        if self.verbose:
            print(self.test_data.info())
            print(self.test_data.tail())

    def convert_to_predict(self, df):
        to_predict = df[self.input_key].tolist()
        if self.verbose:
            print(to_predict[:5])
        return to_predict

    def assign_predictions(self, df, preds):
        df[self.predicted_key] = preds
        return df


class SimpleTrainerNER(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.ner import NERModel

        args = self.args
        if args.labels is None:
            labels = list(self.train_data[self.labels_key].unique())

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


class SimpleTrainerMultiLabel(SimpleTrainer):
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


class SimpleTrainerClassification(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.classification import ClassificationModel

        args = self.args
        if not self.splits:
            self.load_datasets()

        self.model_cfg["labels_list"] = (
            self.train_data[self.labels_key].unique().tolist()
        )
        args["num_labels"] = len(self.model_cfg["labels_list"])

        # Create a NERModel
        model = ClassificationModel(
            args["model_type"],
            args["model_uri"],
            num_labels=args["num_labels"],
            cuda_device=args["cuda_device"],
            args=self.model_cfg,
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
            print(f"Model outputs: {model_outputs[:5]}")
            print(f"num_outputs: {len(model_outputs)}")
            print(f"num_wrong_predictions: {len(wrong_predictions)}")

    def load_model(self, model_dir=None, pred_args=None):
        from simpletransformers.classification import ClassificationModel

        if model_dir is None:
            model_dir = self.args.best_model_dir

        self.model = ClassificationModel(
            self.args.model_type, model_dir, args=self.model_cfg
        )

    def predict(self, to_predict: list):
        if self.model is None:
            self.load_model()

        predictions, raw_outputs = self.model.predict(to_predict)
        return predictions
