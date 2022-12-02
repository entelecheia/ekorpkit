import logging
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    XLNetLMHeadModel,
    AutoModelForSeq2SeqLM,
)
from ekorpkit.datasets.config import DatasetConfig


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())

MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class ModelObjectives(str, Enum):
    mlm = "mlm"
    clm = "clm"
    plm = "plm"
    s2s = "s2s"


LM_MAPPING = {
    "mlm": AutoModelForMaskedLM,
    "clm": AutoModelForCausalLM,
    "plm": XLNetLMHeadModel,
    "s2s": AutoModelForSeq2SeqLM,
}


class ModelArguments(BaseModel):
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
        default=None,
        description="If training from scratch, pass a model type from the list: "
        + ", ".join(MODEL_TYPES),
    )
    model_objective: Optional[ModelObjectives] = Field(
        default=ModelObjectives.mlm,
        description="The objective to use when training the model.",
    )
    config_overrides: Optional[str] = Field(
        default=None,
        description=(
            "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        ),
    )
    config_name: Optional[str] = Field(
        default=None,
        description="Pretrained config name or path if not the same as model_name",
    )
    task_name: Optional[str] = Field(
        default=None,
        description="The tasks to train the model on. Example: 'fill-mask,text-generation,ner'",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the pretrained models downloaded from huggingface.co",
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
    ignore_model_path: bool = Field(
        default=False,
        description="If set, the model path will not be used to load the model.",
    )

    class Config:
        extra = "allow"
        use_enum_values = True

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


class DataTrainingArguments(DatasetConfig):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    max_seq_length: Optional[int] = Field(
        default=None,
        description=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        ),
    )
    mlm: bool = Field(
        default=False,
        description="Train with masked-language modeling loss instead of language modeling.",
    )
    mlm_probability: float = Field(
        default=0.15,
        description="Ratio of tokens to mask for masked language modeling loss",
    )
    line_by_line: bool = Field(
        default=False,
        description="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    group_by_shuffling: bool = Field(
        default=False,
        description=(
            "Whether to shuffle the order of the lines in the dataset before grouping them into batches."
            "Only relevant when line_by_line is False."
        ),
    )
    pad_to_max_length: bool = Field(
        default=False,
        description=(
            "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        ),
    )
    return_special_tokens_mask: bool = Field(
        default=True,
        description="Whether to return special tokens mask information (default True).",
    )
    affix_bos_eos_to_sentences: bool = Field(
        default=False,
        description=(
            "Whether to add the beginning and end of sentence tokens to each sentence in the dataset."
            "Only relevant when line_by_line is True."
        ),
    )
    # _raw_datasets = None
