import os
import sklearn
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ekorpkit.io.file import load_dataframe, save_dataframe


class SimpleTraner:
    def __init__(self, **args):
        args = OmegaConf.create(args)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(args.pred_output_dir, exist_ok=True)
        os.makedirs(args.result_dir, exist_ok=True)
        self.args = args
        self.dataset_cfg = args.get("dataset_cfg", None)
        self.model_cfg = OmegaConf.to_container(args.config)
        self.prediction_args = OmegaConf.to_container(args.prediction)
        self.verbose = args.get("verbose", True)
        self.model_pipeline = self.args.get("_pipeline_", [])
        if self.model_pipeline is None:
            self.model_pipeline = []

        self.model = None
        self.splits = {}
        self.pred_data = {}
        self.to_predict = {}

    def apply_pipeline(self):
        print(f"Applying pipeline: {self.model_pipeline}")
        for pipe in self.model_pipeline:
            getattr(self, pipe)()

    def load_datasets(self):
        if self.dataset_cfg is None:
            print("No dataset config found")
            return
        dataset = instantiate(self.dataset_cfg, _recursive_=False)
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


class SimpleTrainerNER(SimpleTraner):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.ner import NERModel

        args = self.args
        if args.labels is None:
            labels = list(self.train_data["labels"].unique())

        # Create a NERModel
        model = NERModel(
            args.model_type,
            args.model_uri,
            labels=labels,
            cuda_device=args.cuda_device,
            args=OmegaConf.to_container(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_data=self.eval_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions


class SimpleTrainerMultiLabel(SimpleTraner):
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
            args=OmegaConf.to_container(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_df=self.eval_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions


class SimpleTrainerClassification(SimpleTraner):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.classification import ClassificationModel

        args = self.args
        if not self.splits:
            self.load_datasets()

        self.model_cfg["labels_list"] = self.train_data["labels"].unique().tolist()
        args.num_labels = len(self.model_cfg["labels_list"])

        # Create a NERModel
        model = ClassificationModel(
            args.model_type,
            args.model_uri,
            num_labels=args.num_labels,
            cuda_device=args.cuda_device,
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
        print(result.keys())
        print(len(model_outputs), len(wrong_predictions))
        print(model_outputs[:5])
        print(wrong_predictions[:5])

        # # Check predictions
        # return result, model_outputs, predictions

    def load_model(self, model_dir=None, pred_args=None):
        from simpletransformers.classification import ClassificationModel

        if model_dir is None:
            model_dir = self.args.best_model_dir

        self.model = ClassificationModel(
            self.args.model_type, model_dir, args=self.model_cfg
        )

    def load_pred_data(self):
        data_dir = self.prediction_args["data_dir"]
        data_files = self.prediction_args["data_files"]
        columns_to_keep = self.prediction_args["columns_to_keep"]
        self.pred_keys = self.prediction_args["keys"]
        self.input_text_key = self.pred_keys["input_text"]
        self.prediction_key = self.pred_keys["prediction"]

        if data_files is None:
            print("No data files are provided")
            return

        if isinstance(data_files, str):
            data_files = [data_files]
        for data_file in data_files:
            print(f"Loading {data_file}")
            filepath = os.path.join(data_dir, data_file)
            df = load_dataframe(filepath, verbose=self.verbose)
            if columns_to_keep is not None:
                df = df[columns_to_keep]
            if self.verbose:
                print(df.tail())
            data_file = os.path.basename(data_file)
            self.pred_data[data_file] = df
            to_predict = df[self.input_text_key].tolist()
            if self.verbose:
                print(to_predict[:5])
            self.to_predict[data_file] = to_predict

    def save_predictions(self):
        for data_file, preds in self.predictions.items():
            print(f"Saving predictions for {data_file}")
            df = self.pred_data[data_file]
            df[self.prediction_key] = preds
            filepath = os.path.join(self.args.pred_output_dir, data_file)
            save_dataframe(df, filepath, verbose=self.verbose)

    def predict(self):
        if self.model is None:
            self.load_model()
        if not self.pred_data:
            self.load_pred_data()

        self.predictions = {}
        for data_file, to_predict in self.to_predict.items():
            print(f"Predicting {data_file}")
            predictions, raw_outputs = self.model.predict(to_predict)
            self.predictions[data_file] = predictions
            if self.verbose:
                print(predictions[:5])
                print(raw_outputs[:5])
        self.save_predictions()
