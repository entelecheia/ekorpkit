import pandas as pd
import logging
import sklearn
import numpy as np
from typing import Tuple
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
from abc import abstractmethod
from ekorpkit import eKonf
from ekorpkit.config import BaseBatchModel
from ekorpkit.datasets.config import DataframeConfig
from ekorpkit.base import _Keys as Keys
from .config import (
    ColumnConfig,
    ModelBatchConfig,
    TrainerConfig,
    ModelConfig,
    ClassificationModelConfig,
    ClassificationTrainerConfig,
)

log = logging.getLogger(__name__)


class SimpleTrainer(BaseBatchModel):
    batch: ModelBatchConfig = None
    trainer: TrainerConfig = None
    dataset: DataframeConfig = None
    model: ModelConfig = None
    columns: ColumnConfig = None
    __model_obj__ = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **args):
        super().__init__(**args)

    def initialize_configs(self, **args):
        super().initialize_configs(batch_config_class=ModelBatchConfig, **args)
        if self.secrets.HUGGING_FACE_HUB_TOKEN:
            hf_token = self.secrets.HUGGING_FACE_HUB_TOKEN.get_secret_value()
        else:
            hf_token = None
        if self.secrets.WANDB_API_KEY:
            wandb_token = self.secrets.WANDB_API_KEY.get_secret_value()
        else:
            wandb_token = None

        self.dataset.initialize_config(self.root_dir, self.seed)
        # self.model = ModelConfig(**self.config.model)
        self.model.initialize_config(
            self.name, self.model_dir, self.cache_dir, self.log_dir, hf_token, wandb_token
        )
        self.model.eval.output_dir = str(self.batch_dir)

        if self.trainer.output_dir is None:
            self.trainer.output_dir = self.model.model_output_dir
        if self.trainer.best_model_dir is None:
            self.trainer.best_model_dir = self.model.best_model_output_dir
        if self.trainer.cache_dir is None:
            self.trainer.cache_dir = self.model.cache_dir
        if not self.trainer.wandb_kwargs:
            self.trainer.wandb_kwargs = self.model.wandb_kwargs
        if self.trainer.wandb_project is None:
            self.trainer.wandb_project = self.model.wandb_project

    @abstractmethod
    def train(self):
        raise NotImplementedError("Must override train")

    @abstractmethod
    def predict_data(self, data: list):
        raise NotImplementedError("Must override predict")

    @property
    def raw_datasets(self):
        return self.dataset.datasets

    @property
    def model_obj(self) -> ClassificationModel:
        if self.__model_obj__ is None:
            self.load_model()
        return self.__model_obj__

    def load_model():
        raise NotImplementedError("Must override load_model")

    def load_datasets(
        self,
        data=None,
        data_files=None,
        data_dir=None,
        test_size=0.2,
        dev_size=0.1,
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
            test_size=test_size,
            dev_size=dev_size,
            seed=seed,
            shuffle=shuffle,
            encode_labels=encode_labels,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )

    def convert_to_train(self):
        if self.dataset.dev_data is not None:
            self.trainer.evaluate_during_training = True
        else:
            self.trainer.evaluate_during_training = False
        return (
            self.rename_columns(self.dataset.train_data, self.columns.train),
            self.rename_columns(self.dataset.dev_data, self.columns.train),
            self.rename_columns(self.dataset.test_data, self.columns.train),
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
        if renames:
            log.info(f"Renaming columns: {renames}")
            data = data.copy().rename(columns=renames)
        if self.verbose:
            print(data.head())
        return data

    def convert_to_predict(self, data):
        input_col = self.columns.predict[Keys.INPUT]
        data_to_predict = data[input_col].tolist()
        if self.verbose:
            print(data_to_predict[:5])
        return data_to_predict

    def append_predictions(self, data, preds):
        predicted_column = self.columns.predict[Keys.PREDICTED]
        model_outputs_column = self.columns.predict[Keys.MODEL_OUTPUTS]
        data[predicted_column] = preds[Keys.PREDICTED]
        data[model_outputs_column] = preds[Keys.MODEL_OUTPUTS]
        pred_probs_column = self.columns.predict.get(Keys.PRED_PROBS)
        if pred_probs_column:
            data[pred_probs_column] = preds[Keys.PRED_PROBS]
        return data

    def preds_path(self, preds_file=None):
        if preds_file is None:
            preds_file = self.batch.preds_file
        return str(self.batch_dir / f"{self.batch.file_prefix}_{preds_file}")

    def cv_path(self, cv_file=None):
        if cv_file is None:
            cv_file = self.batch.cv_file
        return str(self.batch_dir / f"{self.batch.file_prefix}_{cv_file}")

    def predict(self, data, preds_file=None, **args):
        if args:
            self.columns.predict = args
        data_to_predict = self.convert_to_predict(data)
        preds = self.predict_data(data_to_predict)
        data = self.append_predictions(data, preds)
        eKonf.save_data(data, self.preds_path(preds_file))
        return data

    def cross_val_predict(
        self,
        cv=5,
        dev_size=0.2,
        random_state=1235,
        shuffle=True,
        cv_file=None,
    ):

        splits = self.dataset.cross_val_datasets(
            cv=cv, dev_size=dev_size, random_state=random_state, shuffle=shuffle
        )
        pred_dfs = []
        for split_no, split in splits:
            self.train()
            log.info(f"Predicting split {split_no}")
            pred_df = self.predict(split)
            pred_dfs.append(pred_df)
        cv_preds = pd.concat(pred_dfs)
        eKonf.save_data(cv_preds, self.cv_path(cv_file))
        return cv_preds


class SimpleClassification(SimpleTrainer):
    model: ClassificationModelConfig = None
    trainer: ClassificationTrainerConfig = None

    def __init__(self, config_name: str = "simple.classification", **args):
        config_group = f"task={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)

    @property
    def labels_list(self):
        return self.model_obj.args.labels_list

    @property
    def labels_map(self):
        return self.model_obj.args.labels_map

    def train(self):
        self.reset()
        self.load_config()

        model_args = self.model
        train_data, dev_data, test_data = self.convert_to_train()

        self.trainer.labels_list = train_data[Keys.LABELS].unique().tolist()
        model_args.num_labels = len(self.trainer.labels_list)

        # Create a NERModel
        model = ClassificationModel(
            model_type=model_args.model_type,
            model_name=model_args.model_name_or_path,
            num_labels=model_args.num_labels,
            use_cuda=model_args.use_cuda,
            cuda_device=model_args.cuda_device,
            args=self.trainer.dict(),
        )

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
        self.save_config()
        self.reset()

    def load_model(self, model_dir=None):

        if model_dir is None:
            model_dir = self.trainer.best_model_dir
            if not eKonf.exists(model_dir):
                model_dir = self.trainer.output_dir

        self.__model_obj__ = ClassificationModel(self.model.model_type, model_dir)
        # , args=self._model_cfg
        log.info(f"Loaded model from {model_dir}")

    def predict_data(self, data: list):
        predictions, raw_outputs = self.model_obj.predict(data)
        log.info(f"type of raw_outputs: {type(raw_outputs)}")
        prob_outputs = [softmax(output.flatten().tolist()) for output in raw_outputs]
        model_outputs = [dict(zip(self.labels_list, output)) for output in prob_outputs]
        pred_probs = [output.max() for output in prob_outputs]
        log.info(f"raw_output: {raw_outputs[0]}")
        return {
            Keys.PREDICTED.value: predictions,
            Keys.PRED_PROBS.value: pred_probs,
            Keys.MODEL_OUTPUTS.value: model_outputs,
        }

    def eval_figure_file(self, figure_file=None):
        if figure_file is None:
            figure_file = self.model.eval.output_file
        return f"{self.batch.file_prefix}_{figure_file}"

    def eval(self):
        if self.dataset.test_data is None:
            log.warning("No test data found")
            return

        data_to_predict = self.convert_to_predict(self.dataset.test_data)
        preds = self.predict_data(data_to_predict)
        pred_data = self.append_predictions(self.dataset.test_data, preds)
        eKonf.save_data(pred_data, self.preds_path())
        if self.verbose:
            print(pred_data.head())
        if self.model.eval:
            self.model.eval.labels = self.labels_list
            self.model.eval.visualize.output_file = self.eval_figure_file()
            eKonf.instantiate(self.model.eval, data=pred_data)

    def create_records_from_preds(
        self,
        cv_preds: pd.DataFrame,
        meta_columns=["id", "split"],
        prediction_agent=None,
    ):
        import argilla as rb

        text_col = self.columns.text
        label_col = self.columns.labels
        pred_col = self.columns.model_outputs
        records = []
        for _, row in cv_preds.iterrows():
            record = rb.TextClassificationRecord(
                inputs={"text": row[text_col]},
                prediction=list(row[pred_col].items()),
                annotation=row[label_col] if label_col in cv_preds.columns else None,
                metadata=row[meta_columns].to_dict(),
                prediction_agent=prediction_agent,
            )
            records.append(record)
        log.info(f"Created {len(records)} records")
        return records

    def find_label_errors(
        self,
        pred_data: pd.DataFrame,
        meta_columns=["id", "split"],
        sort_by: str = "likelihood",
        metadata_key: str = "label_error_candidate",
        num_workers: int = None,
        **kwargs,
    ):
        """Finds potential annotation/label errors in your datasets using [cleanlab](https://github.com/cleanlab/cleanlab)."""
        from argilla.labeling.text_classification import find_label_errors

        num_workers = self.batch.num_workers if num_workers is None else num_workers
        records = self.create_records_from_preds(pred_data, meta_columns=meta_columns)
        records_with_label_errors = find_label_errors(
            records,
            sort_by=sort_by,
            metadata_key=metadata_key,
            n_jobs=num_workers,
            **kwargs,
        )
        return records_with_label_errors
