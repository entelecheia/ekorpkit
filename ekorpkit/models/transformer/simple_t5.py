import re
import os
import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate
from statistics import mean
from scipy.stats import pearsonr, spearmanr
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

# from sklearn.metrics import accuracy_score, f1_score
# from transformers import T5Model


class SimpleTrainerT5:
    def __init__(self, **args):
        args = OmegaConf.create(args)
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(args.pred_output_dir, exist_ok=True)
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
        datasets = instantiate(self.dataset_cfg, _recursive_=False)
        datasets.concat_datasets()
        self.splits = datasets.splits

        self.train_data = self.splits["train"]
        if self.verbose:
            print(self.train_data["prefix"].unique())
            print(self.train_data.info())
            print(self.train_data.tail())
        if "dev" in self.splits:
            self.eval_data = self.splits["dev"]
            self.args["evaluate_during_training"] = True
            if self.verbose:
                print(self.eval_data.info())
                print(self.eval_data.tail())
        else:
            self.eval_data = None
            self.args["evaluate_during_training"] = False
        self.test_data = self.splits["test"]
        if self.verbose:
            print(self.test_data.info())
            print(self.test_data.tail())

    def train(self):
        from simpletransformers.t5 import T5Model

        args = self.args

        if not self.splits:
            self.load_datasets()

        # Create a Model
        model = T5Model(args.model_type, args.model_uri, args=self.model_cfg)

        # Train the model
        model.train_model(self.train_data, eval_data=self.eval_data)

        # # Evaluate the model
        result = model.eval_model(self.test_data)
        print(result)

        # # Check predictions
        # # print(predictions[:5])
        # return result, model_outputs, predictions

    def load_model(self, model_dir=None, pred_args=None):
        from simpletransformers.t5 import T5Model

        if model_dir is None:
            model_dir = self.args.best_model_dir
        if pred_args is not None:
            self.model_cfg.update(pred_args)

        self.model = T5Model(self.args.model_type, model_dir, args=self.model_cfg)

    def load_pred_data(self):
        data_dir = self.prediction_args["data_dir"]
        data_files = self.prediction_args["data_files"]
        self.pred_keys = self.prediction_args["keys"]
        self.input_text_key = self.pred_keys["input_text"]
        self.target_text_key = self.pred_keys["target_text"]
        self.pred_task_prefix = self.prediction_args["task_prefix"]
        if isinstance(data_files, str):
            data_files = [data_files]
        for data_file in data_files:
            print(f"Loading {data_file}")
            if data_file.endswith(".csv"):
                df = pd.read_csv(os.path.join(data_dir, data_file), sep=",")
            elif data_file.endswith(".tsv"):
                df = pd.read_csv(os.path.join(data_dir, data_file), sep="\t")
            else:
                raise ValueError("Data file must be .csv or .tsv")
            if self.pred_keys["prefix"] is None:
                dfs = []
                for prefix in self.pred_task_prefix:
                    print(f"Adding {prefix}")
                    df_copy = df.copy(deep=True)
                    df_copy["prefix"] = prefix
                    dfs.append(df_copy)
                df = pd.concat(dfs, ignore_index=True)
            # print(df.info())
            print(df.tail())
            data_file = os.path.basename(data_file)
            self.pred_data[data_file] = df
            to_predict = [
                prefix + ": " + str(input_text)
                for prefix, input_text in zip(
                    df["prefix"].tolist(), df[self.input_text_key].tolist()
                )
            ]
            print(to_predict[:5])
            self.to_predict[data_file] = to_predict

    def save_predictions(self):
        for data_file, preds in self.predictions.items():
            print(f"Saving predictions for {data_file}")
            df = self.pred_data[data_file]
            if self.target_text_key is not None:
                target_texts = df[self.target_text_key].tolist()
            # Saving the predictions if needed
            pred_output_dir = self.args.pred_output_dir
            pred_text_file = os.path.join(
                pred_output_dir, data_file.split(".")[0] + ".txt"
            )
            with open(pred_text_file, "w") as f:
                for i, text in enumerate(df[self.input_text_key].tolist()):
                    f.write(str(text) + "\n\n")

                    if self.target_text_key is not None:
                        f.write("Truth:\n")
                        f.write(target_texts[i] + "\n\n")

                    f.write("Prediction:\n")
                    if self.model_cfg["num_return_sequences"] > 1:
                        for pred in preds[i]:
                            f.write(str(pred) + "\n")
                    else:
                        f.write(str(preds[i]) + "\n")
                    f.write("_" * 100 + "\n")

            if self.model_cfg["num_return_sequences"] > 1:
                preds = [pred[0] for pred in preds]
            df[self.pred_keys["pred_text"]] = preds
            df.to_csv(os.path.join(self.args.pred_output_dir, data_file), index=False)

    def predict(self):
        if self.model is None:
            self.load_model()
        if not self.pred_data:
            self.load_pred_data()

        self.predictions = {}
        for data_file, to_predict in self.to_predict.items():
            print(f"Predicting {data_file}")
            preds = self.model.predict(to_predict)
            self.predictions[data_file] = preds
            print(preds[0])

        self.save_predictions()


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
