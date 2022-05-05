import os
import re
import logging
from statistics import mean
from scipy.stats import pearsonr, spearmanr
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1
from .simple import SimpleTrainer


log = logging.getLogger(__name__)


class SimpleT5(SimpleTrainer):
    def __init__(self, **args):
        super().__init__(**args)

    def load_datasets(self):
        super().load_datasets()
        cols = self.train_data.columns
        renames = {
            name: key for key, name in self._to_train.items() if name and name in cols
        }
        if renames:
            self.train_data.rename(columns=renames, inplace=True)
            if self.eval_data is not None:
                self.eval_data.rename(columns=renames, inplace=True)
            if self.test_data is not None:
                self.test_data.rename(columns=renames, inplace=True)
        task_prefix = self._to_train.get("task_prefix")
        if self._to_train["prefix"] is None:
            self.train_data["prefix"] = task_prefix
            if self.eval_data is not None:
                self.eval_data["prefix"] = task_prefix
            if self.test_data is not None:
                self.test_data["prefix"] = task_prefix
        if self.verbose:
            print("Train data for T5:")
            print(self.train_data.head())

    def train(self):
        from simpletransformers.t5 import T5Model

        args = self.args

        if not self.splits:
            self.load_datasets()

        # Create a Model
        model = T5Model(
            args["model_type"],
            args["model_uri"],
            cuda_device=args["cuda_device"],
            args=self._model_cfg,
        )

        # Train the model
        model.train_model(
            self.train_data,
            eval_data=self.eval_data,
        )

        # # Evaluate the model
        result = model.eval_model(self.test_data)
        print(result)

        # # Check predictions
        # # print(predictions[:5])
        # return result, model_outputs, predictions

    def load_model(self, model_dir=None, pred_args=None):
        from simpletransformers.t5 import T5Model

        if model_dir is None:
            model_dir = self.args["best_model_dir"]
        if pred_args is not None:
            self._model_cfg.update(pred_args)

        self.model = T5Model(self.args["model_type"], model_dir, args=self._model_cfg)
        log.info(f"Loaded model from {model_dir}")

    def _predict(self, to_predict: list):
        if self.model is None:
            self.load_model()

        preds = self.model.predict(to_predict)
        return preds

    def convert_to_predict(self, df):
        input_key = self._to_predict["input"]
        task_prefix = self._to_predict["task_prefix"]
        if self._to_predict["prefix"] is None:
            df["prefix"] = task_prefix

        to_predict = [
            prefix + ": " + str(input_text)
            for prefix, input_text in zip(df["prefix"].tolist(), df[input_key].tolist())
        ]

        if self.verbose:
            print(to_predict[:5])
        return to_predict

    def append_predictions(self, df, preds):
        predicted_key = self._to_predict["predicted"]
        if self._model_cfg["num_return_sequences"] > 1:
            preds = [pred[0] for pred in preds]
        df[predicted_key] = preds
        return df


def f1(truths, preds):
    return mean([compute_f1(truth, pred) for truth, pred in zip(truths, preds)])


def exact(truths, preds):
    return mean([compute_exact(truth, pred) for truth, pred in zip(truths, preds)])


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


def convert_BIO_labels(filename):
    result_labels = []
    with open(filename, "r", encoding="utf-8") as file:
        cnt = 0
        for line in file:
            line = re.sub(r"\*(\w+)", r"\1*", line)
            tokens = re.sub(
                r'[!"#$%&\'()+,-.:;<=>?@[\\\]^_`{\|}~⁇]', " ", line.strip()
            ).split()
            seq_label = []
            start_entity = 0
            entity_type = "O"
            for idx, token in enumerate(tokens):
                if token.endswith("*"):
                    start_entity += (
                        1 if (start_entity == 0 or token[:-1] != entity_type) else -1
                    )
                    entity_type = token[:-1]
                else:
                    if start_entity == 0:
                        seq_label.append("O")
                        entity_type = "O"
                    elif start_entity < 0:
                        raise "Something errors"
                    else:
                        if tokens[idx - 1].endswith("*"):
                            seq_label.append("B-" + entity_type.upper())
                        else:
                            seq_label.append("I-" + entity_type.upper())

            result_labels.append(seq_label)
            cnt += 1
    #             if cnt % 100 == 0:
    #                 print('Processed %d sentences' % cnt)
    return result_labels
