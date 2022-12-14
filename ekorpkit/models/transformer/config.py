import logging
import torch
from omegaconf import DictConfig
from pydantic import BaseModel, Field, validator
from typing import Optional
from ekorpkit import eKonf
from ekorpkit.config import BaseBatchConfig


log = logging.getLogger(__name__)


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


class ModelBatchConfig(BaseBatchConfig):
    pred_file: str = "pred.parquet"


class TrainerArgs(BaseModel):
    best_model_dir: str = "outputs/best_model"
    cache_dir: str = "cache_dir/"
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
    output_dir: str = "outputs/"
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
    wandb_kwargs: dict = Field(default_factory=dict)
    wandb_project: str = None
    warmup_ratio: float = 0.06
    warmup_steps: int = 0
    weight_decay: float = 0.0


class SimpleModelConfig(BaseModel):
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
    model_type: Optional[str] = Field(
        default=None, description="Trasnformer model type."
    )
    model_class: Optional[str] = Field(
        default=None,
        description="The tasks to train the model on. Example: 'fill-mask,text-generation,ner'",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="The directory where the model and the tokenizer will be saved.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    log_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the logs",
    )
    device: Optional[str] = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device (cuda or cpu)",
    )
    wandb_project: Optional[str] = Field(
        default=None,
        description="The name of the wandb project to use.",
    )
    wandb_group: Optional[str] = Field(
        default=None,
        description="The name of the wandb group to use.",
    )
    num_labels: int = None
    eval: DictConfig = None

    @validator("wandb_kwargs", pre=True)
    def wandb_kwargs_validator(cls, v):
        if v is None:
            return {}
        return eKonf.to_dict(v)

    class Config:
        extra = "allow"
        use_enum_values = True

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
        return {"group": self.wandb_group, "dir": self.log_dir}


class ClassificationArgs(TrainerArgs):
    model_class: str = "ClassificationModel"
    tie_value: int = 1
    labels_list: list = Field(default_factory=list)
    labels_map: dict = Field(default_factory=dict)
