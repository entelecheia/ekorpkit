import logging
import torch
from pathlib import Path
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from typing import Optional
from ekorpkit.config import BaseBatchConfig
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
)

logger = logging.getLogger(__name__)


class ColumnConfig(BaseModel):
    train: DictConfig = None
    predict: DictConfig = None
    id: str = "id"
    labels: str = "labels"
    text: str = "text"
    predicted: str = "pred_labels"
    pred_probs: str = "pred_probs"
    model_outputs: str = "raw_preds"
    input: str = "text"
    actual: str = "labels"

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True


class ModelBatchConfig(BaseBatchConfig):
    preds_file: str = "preds.parquet"
    cv_file: str = "cv.parquet"


class TrainerConfig(BaseModel):
    best_model_dir: str = None
    cache_dir: str = None
    eval_batch_size: int = 8
    evaluate_during_training: bool = False
    evaluate_during_training_silent: bool = True
    evaluate_during_training_steps: int = 2000
    evaluate_during_training_verbose: bool = False
    evaluate_each_epoch: bool = True
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-5
    local_rank: int = -1
    logging_steps: int = 50
    loss_type: str = None
    manual_seed: int = None
    max_grad_norm: float = 1.0
    max_seq_length: int = 128
    no_cache: bool = False
    no_save: bool = False
    num_train_epochs: int = 1
    optimizer: str = "AdamW"
    output_dir: str = None
    overwrite_output_dir: bool = False
    save_best_model: bool = True
    save_eval_checkpoints: bool = True
    save_model_every_epoch: bool = True
    save_optimizer_and_scheduler: bool = True
    save_steps: int = 2000
    scheduler: str = "linear_schedule_with_warmup"
    silent: bool = False
    skip_special_tokens: bool = True
    tensorboard_dir: str = None
    thread_count: int = None
    tokenizer_name: str = None
    tokenizer_type: str = None
    train_batch_size: int = 8
    train_custom_parameters_only: bool = False
    use_cached_eval_features: bool = False
    use_early_stopping: bool = False
    use_hf_datasets: bool = False
    use_multiprocessing: bool = True
    use_multiprocessing_for_evaluation: bool = True
    wandb_kwargs: dict = None
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0


class ModelConfig(BaseModel):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = Field(
        default=None,
        description=(
            "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        ),
    )
    model_name: Optional[str] = Field(
        default=None,
        description="The model name to use when saving the model and the tokenizer.",
    )
    model_dir: Optional[str] = Field(
        default=None,
        description="The directory where the model and the tokenizer will be saved.",
    )
    model_type: Optional[str] = Field(
        default=None, description="Trasnformer model type."
    )
    config_name: Optional[str] = Field(
        default=None,
        description="Pretrained config name or path if not the same as model_name",
    )
    config_overrides: Optional[str] = Field(
        default=None,
        description=(
            "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        ),
    )
    task_name: Optional[str] = Field(
        default=None,
        description="The tasks to train the model on. Example: 'fill-mask,text-generation,ner'",
    )
    log_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the logs",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    device: Optional[str] = Field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        description="Device (cuda or cpu)",
    )
    model_revision: str = Field(
        default="main",
        description="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    use_auth_token: bool = Field(
        default=False,
        description=(
            "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
            "with private models)."
        ),
    )
    wandb_project: Optional[str] = Field(
        default=None,
        description="The name of the wandb project to use.",
    )
    wandb_group: Optional[str] = Field(
        default=None,
        description="The name of the wandb group to use.",
    )
    ignore_existing_model_output: bool = Field(
        default=False,
        description="If set, the model output directory will be overwritten.",
    )
    use_model_name_or_path: bool = Field(
        default=True,
        description="If set, the model_name_or_path will be used to load the model and the tokenizer.",
    )
    eval: DictConfig = None
    __auto_config__ = None
    __model_obj__ = None

    class Config:
        extra = "allow"
        use_enum_values = True
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "config_overrides can't be used in combination with config_name or model_name_or_path"
            )

    def initialize_config(self, name, model_dir, cache_dir, log_dir, hf_token, wandb_token=None):
        if self.model_name is None:
            self.model_name = "{}-{}".format(name, self.model_config_name)
        if self.model_dir is None:
            self.model_dir = str(model_dir)
        if self.cache_dir is None:
            self.cache_dir = str(cache_dir)
        if self.log_dir is None:
            self.log_dir = str(log_dir)
        self.model_type = self.auto_config.model_type
        self.use_auth_token = hf_token
        if wandb_token:
            self.wandb_project = self.wandb_project.replace("/", "-") or "transformers"
        else:
            self.wandb_project = None

    @property
    def model_config_name(self):
        return self.config_name if self.config_name else self.model_name_or_path

    @property
    def model_output_dir(self):
        return str(Path(self.model_dir) / self.model_name)

    @property
    def best_model_output_dir(self):
        return str(Path(self.model_output_dir) / "best_model")

    @property
    def use_cuda(self):
        return self.device.startswith("cuda")

    @property
    def cuda_device(self):
        if self.device.startswith("cuda:"):
            return int(self.device.split(":")[1])
        return -1

    @property
    def wandb_kwargs(self):
        return {"group": self.wandb_group.replace("/", "-"), "dir": self.log_dir}

    @property
    def model_obj(self):
        if self.__model_obj__ is not None:
            if self.device != self.__model_obj__.device:
                logger.info(
                    f"Moving model to device [{self.device}] from [{self.__model_obj__.device}]"
                )
                self.__model_obj__ = self.__model_obj__.to(self.device)
        return self.__model_obj__

    @property
    def auto_config(self):
        if self.__auto_config__ is None:
            self.__auto_config__ = self.load_auto_config()
        return self.__auto_config__

    def load_auto_config(self):
        config_kwargs = {
            "cache_dir": self.cache_dir,
            "revision": self.model_revision,
            "use_auth_token": self.use_auth_token,
        }
        if self.config_name:
            config = AutoConfig.from_pretrained(self.config_name, **config_kwargs)
        elif self.model_name_or_path:
            config = AutoConfig.from_pretrained(
                self.model_name_or_path, **config_kwargs
            )
        else:
            config = CONFIG_MAPPING[self.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if self.config_overrides is not None:
                logger.info(f"Overriding config: {self.config_overrides}")
                config.update_from_string(self.config_overrides)
                logger.info(f"New config: {config}")

        return config


class ClassificationModelConfig(ModelConfig):
    num_labels: int = None


class ClassificationTrainerConfig(TrainerConfig):
    model_class: str = "ClassificationModel"
    tie_value: int = 1
    labels_list: list = None
    labels_map: dict = None
