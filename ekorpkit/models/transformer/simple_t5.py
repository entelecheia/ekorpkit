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
        task_prefix = self.args.task_prefix._train_
        if task_prefix == self.Keys.CLASSIFICATION:
            train_data, _, _ = self.convert_to_train()
            self.labels_list = train_data[self.Keys.TARGET_TEXT].unique().tolist()
            log.info(f"Label list: {self.labels_list}")

    def convert_to_train(self):
        train_data, dev_data, test_data = super().convert_to_train()
        prefix_col = self._train_[self.Keys.PREFIX]
        task_prefix = self.args.task_prefix._train_
        if task_prefix is not None:
            train_data[prefix_col] = task_prefix
            if dev_data is not None:
                dev_data[prefix_col] = task_prefix
            if test_data is not None:
                test_data[prefix_col] = task_prefix
        return train_data, dev_data, test_data

    def train(self):
        from simpletransformers.t5 import T5Model

        args = self.args

        if not self.splits:
            self.load_datasets()
        train_data, dev_data, test_data = self.convert_to_train()
        if self.verbose:
            print(train_data.head())
        # Create a Model
        model = T5Model(
            args.model_type,
            args.model_uri,
            cuda_device=args.cuda_device,
            args=self._model_cfg,
        )

        # Train the model
        model.train_model(
            train_data,
            eval_data=dev_data,
        )

        # # Evaluate the model
        result = model.eval_model(test_data)
        print(result)

        # # Check predictions
        # # print(predictions[:5])
        # return result, model_outputs, predictions

    def load_model(self, model_dir=None, pred_args=None):
        from simpletransformers.t5 import T5Model

        if model_dir is None:
            model_dir = self.args.config.best_model_dir
        if pred_args is not None:
            self._model_cfg.update(pred_args)

        self.model = T5Model(self.args.model_type, model_dir, args=self._model_cfg)
        log.info(f"Loaded model from {model_dir}")

    def _predict(self, data: list):
        if self.model is None:
            self.load_model()

        preds = self.model.predict(data)
        return preds

    def convert_to_predict(self, data):
        input_col = self._predict_[self.Keys.INPUT]
        prefix_col = self._predict_[self.Keys.PREFIX]
        task_prefix = self.args.task_prefix._predict_
        if task_prefix is not None:
            data[prefix_col] = task_prefix

        data_to_predict = [
            prefix + ": " + str(input_text)
            for prefix, input_text in zip(
                data[prefix_col].tolist(), data[input_col].tolist()
            )
        ]

        if self.verbose:
            print(data_to_predict[:5])
        return data_to_predict

    def append_predictions(self, df, preds):
        predicted_col = self._predict_[self.Keys.PREDICTED]
        if self.args.config.num_return_sequences > 1:
            preds = [pred[0] for pred in preds]
        df[predicted_col] = preds
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
