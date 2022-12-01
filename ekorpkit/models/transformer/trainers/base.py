import logging
import os
import transformers
import datasets
from omegaconf import DictConfig
from pathlib import Path
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from .config import ModelArguments, DataTrainingArguments
from ekorpkit.tokenizers.config import TokenizerConfig
from ekorpkit.config import BaseBatchModel


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


class BaseTrainer(BaseBatchModel):
    model: ModelArguments = None
    training: DictConfig = None
    dataset: DataTrainingArguments = None
    tokenizer: TokenizerConfig = None
    use_accelerator: bool = False
    training_args: TrainingArguments = None
    last_checkpoint: str = None
    __auto_config__ = None
    __tokenized_datasets__ = None
    __model_obj__ = None
    __tokenizer_obj__ = None

    def __init__(self, config_group: str = "transformer=trainer", **args):
        super().__init__(config_group, **args)
        self._init_env()
        self.autorun()

    def _init_configs(self):
        super()._init_configs()

        if self.dataset.data_dir is None:
            self.dataset.data_dir = str(self.root_dir)
        if self.dataset.seed is None:
            self.dataset.seed = self.seed

        self.model.cache_dir = str(self.cache_dir)
        self.model.use_auth_token = (
            self.secret.hugging_face_hub_token.get_secret_value()
        )

        if self.tokenizer.tokenizer_name is None:
            self.tokenizer.tokenizer_name = self.model_name
        if self.tokenizer.model_type is None:
            self.tokenizer.model_type = self.model_type
        if self.tokenizer.model_dir is None:
            self.tokenizer.model_dir = self.model_dir
        if self.tokenizer.tokenizer_dir is None:
            self.tokenizer.tokenizer_dir = self.tokenizer_dir
        self.tokenizer.model_max_length = self.dataset.max_seq_length
        self.tokenizer.output_dir = self.output_dir
        self.tokenizer.cache_dir = str(self.cache_dir)
        self.tokenizer.use_auth_token = (
            self.secret.hugging_face_hub_token.get_secret_value()
        )

        training_args = TrainingArguments(**self.training)
        if training_args.output_dir is None:
            training_args.output_dir = self.model_path
        training_args.seed = self.seed
        self.training_args = training_args

    def _init_env(self):
        training_args = self.training_args

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

    @property
    def model_path(self):
        return str(self.model_dir / self.model_name)

    @property
    def model_dir(self):
        model_dir = Path(self.model.model_dir or "models")
        if not model_dir.is_absolute():
            model_dir = self.output_dir / model_dir / self.name
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        return model_dir

    @property
    def tokenizer_dir(self):
        tokenizer_dir = Path(self.tokenizer.tokenizer_dir or "tokenizers")
        if not tokenizer_dir.is_absolute():
            tokenizer_dir = self.output_dir / tokenizer_dir / self.name
        return tokenizer_dir

    @property
    def model_name(self):
        model_name = self.model.model_name
        if model_name is None:
            model_name = "{}-{}".format(self.name, self.model_config_name)
            self.model.model_name = model_name
        return model_name

    @property
    def model_type(self):
        return self.auto_config.model_type

    @property
    def model_config_name(self):
        return (
            self.model.config_name
            if self.model.config_name
            else self.model.model_name_or_path
        )

    @property
    def raw_datasets(self):
        return self.dataset.datasets

    @property
    def auto_config(self):
        if self.__auto_config__ is None:
            self.__auto_config__ = self.load_auto_config()
        return self.__auto_config__

    def load_auto_config(self):
        # Load model config
        model_args = self.model

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": model_args.use_auth_token,
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
    def model_obj(self):
        if self.__model_obj__ is None:
            self.load_model()
        return self.__model_obj__

    @property
    def tokenizer_obj(self):
        return self.tokenizer.tokenizer_obj

    @property
    def tokenized_datasets(self):
        if self.__tokenized_datasets__ is None:
            self.preprocess_datasets()
        return self.__tokenized_datasets__

    def load_model(self):
        raise NotImplementedError

    def preprocess_datasets(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
