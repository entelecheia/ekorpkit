import logging
import math
import os
import evaluate
import transformers
import datasets
from itertools import chain
from pathlib import Path
from tokenizers import Tokenizer
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    pipeline,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from accelerate import Accelerator
from .config import ModelArguments, DataTrainingArguments, PreTrainedTokenizerArguments
from ekorpkit.batch import BaseConfig


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


class MlmTrainer(BaseConfig):
    def __init__(self, root_dir=None, **args):
        self._model_args = None
        self._dataset_args = None
        self._training_args = None
        self._tokenizer_args = None
        self._lm_config = None

        self._raw_datasets = None
        self._model = None
        self._tokenizer = None
        # self.trainer = None
        self._tokenized_datasets = None
        self._pipe = None
        self.last_checkpoint = None

        super().__init__(root_dir=root_dir, **args)
        self._init_env()
        self.autorun()

    @property
    def model_config(self):
        if self._model_args is None:
            self._model_args = ModelArguments(**self.config.model)
            self._model_args.cache_dir = str(self.cache_dir)
        return self._model_args

    @property
    def model_path(self):
        return str(self.model_dir / self.model_name)

    @property
    def model_dir(self):
        model_dir = Path(self.model_config.model_dir)
        if not model_dir.is_absolute():
            model_dir = self.output_dir / model_dir / self.name
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        return model_dir

    @property
    def model_name(self):
        model_name = self.model_config.model_name
        if model_name is None:
            model_name = "{}_{}".format(self.name, self.model_config_name)
        return model_name

    @property
    def model_type(self):
        return self.lm_config.model_type

    @property
    def model_config_name(self):
        return (
            self.model_config.config_name
            if self.model_config.config_name
            else self.model_config.model_name_or_path
        )

    @property
    def tokenizer_config(self):
        if self._tokenizer_args is None:
            tk_args = PreTrainedTokenizerArguments(**self.config.tokenizer)
            tk_args.model_max_length = self.dataset_config.max_seq_length
            if self.model_type is not None:
                tk_args.model_type = self.model_type
            self._tokenizer_args = tk_args
        return self._tokenizer_args

    @property
    def training_config(self):
        if self._training_args is None:
            training_args = TrainingArguments(**self.config.training)
            if training_args.output_dir is None:
                training_args.output_dir = self.model_path
            training_args.seed = self.seed
            self._training_args = training_args
        return self._training_args

    def _init_env(self):
        training_args = self.training_config

        log_level = training_args.get_process_log_level()
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

    def prepare_tokenizer(self):
        from tokenizers.processors import BertProcessing

        tk_args = self.tokenizer_config
        if self.pretrained_tokenizer_path is None:
            raise ValueError("tokenizer.path is required")

        tok = Tokenizer.from_file(self.pretrained_tokenizer_path)
        if tk_args.model_type == "bert":
            sep_token = tk_args.sep_token
            cls_token = tk_args.cls_token
            tok.post_processor = BertProcessing(
                sep=(sep_token, tok.token_to_id(sep_token)),
                cls=(cls_token, tok.token_to_id(cls_token)),
            )
        elif tk_args.model_type == "roberta":
            sep_token = tk_args.eos_token
            cls_token = tk_args.bos_token
            tok.post_processor = BertProcessing(
                sep=(sep_token, tok.token_to_id(sep_token)),
                cls=(cls_token, tok.token_to_id(cls_token)),
            )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            truncation=tk_args.truncation,
            max_length=tk_args.model_max_length,
            return_length=tk_args.return_length,
            padding_side=tk_args.padding_side,
            **tk_args.special_tokens_map,
        )
        # Configuration
        config_kwargs = {
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token_id": tokenizer.mask_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            # "torch_dtype": "float16",
        }
        if tk_args.model_type == "bert":
            config_kwargs["sep_token_id"] = tokenizer.sep_token_id
            config_kwargs["cls_token_id"] = tokenizer.cls_token_id
        elif tk_args.model_type == "roberta":
            config_kwargs["eos_token_id"] = tokenizer.eos_token_id
            config_kwargs["bos_token_id"] = tokenizer.bos_token_id
        logger.info(f"Update model config with {config_kwargs}")
        self._lm_config.update(config_kwargs)

        logger.info(f"Is a fast tokenizer? {tokenizer.is_fast}")
        logger.info(f"Vocab size: {tokenizer.vocab_size}")
        tokenizer.save_pretrained(self.model_path)
        self._model_args.tokenizer_name = self.model_path

    @property
    def dataset_config(self):
        if self._dataset_args is None:
            self._dataset_args = DataTrainingArguments(**self.config.dataset)
            self._dataset_args.cache_dir = str(self.cache_dir)
        return self._dataset_args

    def prepare_datasets(self):
        self._raw_datasets = self.dataset_config.load_datasets()

    @property
    def raw_datasets(self):
        if self._raw_datasets is None:
            self.prepare_datasets()
        return self._raw_datasets

    @property
    def tokenizer_name_or_path(self):
        model_args = self.model_config
        if model_args.tokenizer_name:
            return model_args.tokenizer_name
        elif model_args.model_name_or_path:
            return model_args.model_name_or_path

    @property
    def pretrained_tokenizer_dir(self):
        model_dir = Path(self.tokenizer_config.model_dir or "tokenizers")
        if not model_dir.is_absolute():
            model_dir = self.output_dir / model_dir / self.name
        return model_dir

    @property
    def pretrained_tokenizer_path(self):
        if self.tokenizer_config.path is not None:
            return self.tokenizer_config.path
        else:
            return str(self.pretrained_tokenizer_dir / self.tokenizer_config.name)

    def load_lm_config(self):
        # Load model config
        model_args = self.model_config

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, **config_kwargs
            )
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")

        return config

    @property
    def lm_config(self):
        if self._lm_config is None:
            self._lm_config = self.load_lm_config()
        return self._lm_config

    def load_model(self):
        # Load pretrained model
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        model_args = self.model_config
        lm_config = self.lm_config

        if model_args.model_name_or_path:
            model = AutoModelForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=lm_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForMaskedLM.from_config(lm_config)
        self._model = model

    def load_tokenizer(self):
        # Load tokenizer
        model_args = self.model_config

        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name, **tokenizer_kwargs
            )
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path, **tokenizer_kwargs
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(tokenizer))

        self._tokenizer = tokenizer

    def preprocess_datasets(self):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        data_args = self.dataset_config
        training_args = self.training_config

        raw_datasets = self.raw_datasets
        tokenizer = self.tokenizer

        if training_args.do_train:
            column_names = raw_datasets["train"].column_names
        else:
            column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

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

        if data_args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line
                    for line in examples[text_column_name]
                    if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
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
            def tokenize_function(examples):
                return tokenizer(
                    examples[text_column_name], return_special_tokens_mask=True
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

        self._tokenized_datasets = tokenized_datasets

    @property
    def use_accelerator(self):
        return self.config.get("use_accelerator", False)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.pretrained_tokenizer_path is not None:
                self.prepare_tokenizer()
            self.load_tokenizer()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def tokenized_datasets(self):
        if self._tokenized_datasets is None:
            self.preprocess_datasets()
        return self._tokenized_datasets

    def train(self):
        model_args = self.model_config
        data_args = self.dataset_config
        training_args = self.training_config

        tokenizer = self.tokenizer
        model = self.model
        tokenized_datasets = self.tokenized_datasets
        last_checkpoint = self.last_checkpoint

        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = tokenized_datasets["train"]
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        if training_args.do_eval:
            if "validation" not in tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
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

        # Data collator
        # This one will take care of randomly masking the tokens.
        pad_to_multiple_of_8 = (
            data_args.line_by_line
            and training_args.fp16
            and not data_args.pad_to_max_length
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

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

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
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

    def fill_mask(self, text, top_k=5, reload=False, **kwargs):
        """Fill the mask in a text"""
        if reload or self._pipe is None:
            model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            self._pipe = pipeline(
                "fill-mask", model=model, tokenizer=tokenizer, **kwargs
            )
        return self._pipe(text, top_k=top_k)
