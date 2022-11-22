import logging
from pydantic import BaseModel, Field
from typing import Optional
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
)
from ekorpkit import eKonf


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())

MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class PreTrainedTokenizerArguments(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None
    model_type: Optional[str] = None
    truncation: Optional[bool] = True
    model_max_length: Optional[int] = None
    return_length: Optional[bool] = False
    padding: Optional[bool] = True
    padding_side: Optional[str] = "right"
    unk_token: Optional[str] = "<unk>"
    pad_token: Optional[str] = "<pad>"
    cls_token: Optional[str] = "<cls>"
    sep_token: Optional[str] = "<sep>"
    mask_token: Optional[str] = "<mask>"
    bos_token: Optional[str] = "<s>"
    eos_token: Optional[str] = "</s>"

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if self.name is None and self.path is None:
            raise ValueError("Either filename or path must be set")
        elif self.name and self.path:
            raise ValueError("Only one of filename or path must be set")

    @property
    def special_tokens_map(self):
        tokens = dict(
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            mask_token=self.mask_token,
        )
        if self.model_type == "bert":
            tokens["cls_token"] = self.cls_token
            tokens["sep_token"] = self.sep_token
        elif self.model_type == "roberta":
            tokens["bos_token"] = self.bos_token
            tokens["eos_token"] = self.eos_token
        return tokens


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
    model_type: Optional[str] = Field(
        default=None,
        description="If training from scratch, pass a model type from the list: "
        + ", ".join(MODEL_TYPES),
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
    tokenizer_name: Optional[str] = Field(
        default=None,
        description="Pretrained tokenizer name or path if not the same as model_name",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    use_fast_tokenizer: bool = Field(
        default=True,
        description="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.",
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

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


class DataTrainingArguments(BaseModel):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = Field(
        default=None,
        description="The name of the dataset to use (via the datasets library).",
    )
    dataset_config_name: Optional[str] = Field(
        default=None,
        description="The configuration name of the dataset to use (via the datasets library).",
    )
    train_file: Optional[str] = Field(
        default=None,
        description="The input training data file (or folder).",
    )
    file_extention: str = Field(
        default="txt",
        description="The input file extension",
    )
    validation_file: Optional[str] = Field(
        default=None,
        description="An optional input evaluation data file to evaluate the perplexity on (or folder).",
    )
    overwrite_cache: bool = Field(
        default=False,
        description="Overwrite the cached training and evaluation sets",
    )
    validation_split_percentage: Optional[int] = Field(
        default=5,
        description="The percentage of the train set used as validation set in case there's no validation split",
    )
    max_seq_length: Optional[int] = Field(
        default=None,
        description=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        ),
    )
    preprocessing_num_workers: Optional[int] = Field(
        default=None,
        description="The number of processes to use for the preprocessing.",
    )
    mlm_probability: float = Field(
        default=0.15,
        description="Ratio of tokens to mask for masked language modeling loss",
    )
    line_by_line: bool = Field(
        default=False,
        description="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    pad_to_max_length: bool = Field(
        default=False,
        description=(
            "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        ),
    )
    max_train_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    max_eval_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )

    def __init__(self, **kw):
        super().__init__(**kw)
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                if eKonf.is_file(self.train_file):
                    extension = self.train_file.split(".")[-1]
                    if extension not in ["csv", "json", "txt"]:
                        raise ValueError(
                            "`train_file` should be a csv, a json or a txt file."
                        )
                    self.file_extention = extension
                elif eKonf.is_dir(self.train_file):
                    files = eKonf.get_filepaths(
                        f"*.{self.file_extention}", self.train_file
                    )
                    if len(files) == 0:
                        raise ValueError(
                            f"Could not find any {self.file_extention} file in {self.train_file}"
                        )
                else:
                    raise ValueError(
                        "`train_file` doesn't exist. Please check the path."
                    )

            if self.validation_file is not None:
                if eKonf.is_file(self.validation_file):
                    extension = self.validation_file.split(".")[-1]
                    if extension not in ["csv", "json", "txt"]:
                        raise ValueError(
                            "`validation_file` should be a csv, a json or a txt file."
                        )
                    if extension != self.file_extention:
                        raise ValueError(
                            "`validation_file` should have the same extension as `train_file`."
                        )
                elif eKonf.is_dir(self.validation_file):
                    files = eKonf.get_filepaths(
                        f"*.{self.file_extention}", self.validation_file
                    )
                    if len(files) == 0:
                        raise ValueError(
                            f"Could not find any {self.file_extention} file in {self.validation_file}"
                        )
