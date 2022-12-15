import logging
import os
import transformers
import datasets
import random
import math
import evaluate
import torch
from itertools import chain
from pydantic import validator
from pathlib import Path
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
    Trainer,
    is_torch_tpu_available,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from .config import LMModelConfig, LMTrainingDatasetConfig, LM_MAPPING
from ekorpkit.tokenizers.config import TokenizerConfig
from ekorpkit.config import BaseBatchModel
from ekorpkit import eKonf

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


class BaseLMTrainer(BaseBatchModel):
    model: LMModelConfig = None
    trainer: TrainingArguments = None
    dataset: LMTrainingDatasetConfig = None
    tokenizer: TokenizerConfig = None
    use_accelerator: bool = False
    last_checkpoint: str = None
    __tokenized_datasets__ = None

    def __init__(self, config_name: str = None, config_group: str = None, **args):
        super().__init__(config_name=config_name, config_group=config_group, **args)
        self._init_env()
        self.autorun()

    @validator("trainer", pre=True)
    def _check_trainer(cls, v):
        return None

    def initialize_configs(self, **args):
        super().initialize_configs(**args)
        hf_token = self.secrets.HUGGING_FACE_HUB_TOKEN.get_secret_value()

        self.dataset.initialize_config(self.root_dir, self.seed)
        self.model.initialize_config(
            self.name, self.model_dir, self.cache_dir, self.log_dir, hf_token
        )
        self.tokenizer.initialize_config(self.model, self.dataset, self.root_dir)

        if self.trainer is None:
            training_args = TrainingArguments(**eKonf.to_dict(self.config.trainer))
        else:
            training_args = self.trainer
        training_args.output_dir = self.model.model_output_dir
        training_args.seed = self.seed
        training_args.hub_token = hf_token
        self.trainer = training_args

    def _init_env(self):
        training_args = self.trainer

        # log_level = training_args.get_process_log_level()
        log_level = self.envs.EKORPKIT_LOG_LEVEL
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f", distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        # Set the verbosity to info of the Transformers logger (on main process only):
        if self.verbose > 1:
            logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        set_seed(training_args.seed)

        self.last_checkpoint = last_checkpoint

    @property
    def model_obj(self):
        if self.model.model_obj is None:
            self.load_model()
        return self.model.model_obj

    @property
    def raw_datasets(self):
        return self.dataset.datasets

    @property
    def tokenizer_obj(self):
        return self.tokenizer.tokenizer_obj

    @property
    def tokenized_datasets(self):
        if self.__tokenized_datasets__ is None:
            self.preprocess_datasets()
        return self.__tokenized_datasets__

    def load_datasets(
        self,
        dataset_name=None,
        dataset_config_name=None,
        text_column_name=None,
        train_file=None,
        validation_file=None,
    ):
        self.dataset.load_datasets(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            text_column_name=text_column_name,
            train_file=train_file,
            validation_file=validation_file,
        )
        self.__tokenized_datasets__ = None

    def load_model(self, model_name=None):
        # Load pretrained model
        self.initialize_configs()
        self.model.load_model(self.tokenizer_obj, model_name)

    def preprocess_datasets(self):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        data_args = self.dataset
        model_args = self.model
        training_args = self.trainer
        raw_datasets = self.raw_datasets
        column_names = raw_datasets["train"].column_names
        text_column_name = self.dataset.text_column_name
        tokenizer = self.tokenizer_obj

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({max_seq_length}). "
                    "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        return_special_tokens_mask = data_args.return_special_tokens_mask
        affix_bos_eos_to_sentences = data_args.affix_bos_eos_to_sentences
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token

        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line
                    if not affix_bos_eos_to_sentences
                    else f"{bos_token} {line} {eos_token}"
                    for line in examples[text_column_name]
                    if len(line) > 0 and not line.isspace()
                ]
                # Tokenize lines
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=return_special_tokens_mask,
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            group_by_shuffling = data_args.group_by_shuffling

            def tokenize_function(examples):
                examples[text_column_name] = [
                    line
                    if not affix_bos_eos_to_sentences
                    else f"{bos_token} {line} {eos_token}"
                    for line in examples[text_column_name]
                    if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    return_special_tokens_mask=return_special_tokens_mask,
                )

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )

            # Main data processing function that will concatenate all texts from our dataset and generate chunks of
            # max_seq_length.
            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                if group_by_shuffling:
                    # Shuffle the order of concatenated examples
                    random.shuffle(concatenated_examples["input_ids"])
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, we could add padding if the model supported it instead of this drop,
                # you can customize this part to your needs.
                if total_length >= max_seq_length:
                    total_length = (total_length // max_seq_length) * max_seq_length
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + max_seq_length]
                        for i in range(0, total_length, max_seq_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                if model_args.model_objective == "clm":
                    result["labels"] = result["input_ids"].copy()
                return result

            # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
            # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
            # might be slower to preprocess.
            #
            # To speed up this part, we use multiprocessing.
            # See the documentation of the map method for more information:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

            with training_args.main_process_first(desc="grouping texts together"):
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )

        self.__tokenized_datasets__ = tokenized_datasets

    def train(self):
        self.reset()
        self.initialize_configs()

        model_args = self.model
        data_args = self.dataset
        training_args = self.trainer

        tokenizer = self.tokenizer_obj
        model = self.model_obj
        tokenized_datasets = self.tokenized_datasets
        last_checkpoint = self.last_checkpoint

        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("do_train requires a train dataset")
            train_dataset = tokenized_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        if "validation" not in tokenized_datasets:
            training_args.do_eval = False
        if training_args.do_eval:
            # if "validation" not in tokenized_datasets:
            #     raise ValueError("do_eval requires a validation dataset")
            eval_dataset = tokenized_datasets["validation"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

            def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    # Depending on the model and config, logits may contain extra tensors,
                    # like past_key_values, but logits always come first
                    logits = logits[0]
                return logits.argmax(dim=-1)

            metric = evaluate.load("accuracy")

            if model_args.model_objective == "mlm":

                def compute_metrics(eval_preds):
                    preds, labels = eval_preds
                    # preds have the same shape as the labels, after the argmax(-1) has been calculated
                    # by preprocess_logits_for_metrics
                    labels = labels.reshape(-1)
                    preds = preds.reshape(-1)
                    mask = labels != -100
                    labels = labels[mask]
                    preds = preds[mask]
                    return metric.compute(predictions=preds, references=labels)

            else:

                def compute_metrics(eval_preds):
                    preds, labels = eval_preds
                    # preds have the same shape as the labels, after the argmax(-1) has been calculated
                    # by preprocess_logits_for_metrics but we need to shift the labels
                    labels = labels[:, 1:].reshape(-1)
                    preds = preds[:, :-1].reshape(-1)
                    return metric.compute(predictions=preds, references=labels)

        # Data collator
        # This one will take care of randomly masking the tokens.
        pad_to_multiple_of_8 = (
            data_args.line_by_line
            and training_args.fp16
            and not data_args.pad_to_max_length
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=data_args.mlm,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

        torch.set_grad_enabled(True)
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            if training_args.do_eval and not is_torch_tpu_available()
            else None,
        )

        # if use_accelerator is True, then prepare trainer for distributed training
        if self.use_accelerator:
            accelerator = Accelerator()
            acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
            logger.info(f"Accelerator state: {acc_state}")
            device = accelerator.device
            logger.info(f"Accelerator device: {device}")
            trainer = accelerator.prepare(trainer)

        self.save_config()

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload
            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate()

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": model_args.task_name,
        }
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs[
                    "dataset"
                ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)

        del trainer
        self.reset()

    def reset(self):
        # self.__tokenized_datasets__ = None
        # self.__model_obj__ = None
        super().reset()
