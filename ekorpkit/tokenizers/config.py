import logging
from pydantic import BaseModel, Field, validator
from typing import Optional, Union
from enum import Enum
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from ekorpkit.datasets.config import DatasetConfig
from ekorpkit.models.transformer.config import ModelConfig

# from ekorpkit import eKonf


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    UNIGRAM = "unigram"
    BPE = "bpe"
    WORD = "word"
    CHAR = "char"
    WORDPIECE = "wordpiece"


class TrainerType(str, Enum):
    SPM = "spm"
    HF = "huggingface"


class DatasetType(str, Enum):
    DATASET = "dataset"
    TEXT = "text"


class TokenizerConfig(BaseModel):
    tokenizer_name: str = Field(
        None,
        description="Name of the tokenizer to use. If None, the tokenizer will be named after the model.",
    )
    tokenizer_name_or_path: str = Field(
        None,
        description="Path to a pretrained tokenizer or a pretrained model with a tokenizer.",
    )
    pretrained_tokenizer_file: str = Field(
        None,
        description="Path to a pretrained tokenizer file to convert to a transformers tokenizer.",
    )
    tokenizer_dir: Optional[Path] = Field(
        None,
        description="Path to the directory where the pretrained tokenizer will be loaded.",
    )
    model_dir: Optional[Path] = Field(
        None,
        description=(
            "Parent directory to save the tokenizer. "
            "If None, the tokenizer will be saved in the same directory as the model."
        ),
    )
    model_type: str = Field(
        None,
        description="Type of the trasnformer model [bert, roberta, ...]",
    )
    ignore_existing_model_output: bool = Field(
        False,
        description="If set, the model path will not be used to load the tokenizer.",
    )
    truncation: bool = Field(
        True,
        description="Whether to truncate the input sequence to the maximum length.",
    )
    model_max_length: Optional[int] = Field(
        None,
        description="Maximum length of the model input. If None, the maximum length of the dataset will be used.",
    )
    return_length: bool = Field(
        False,
        description="Whether to return the length of the input sequence.",
    )
    unk_token: Optional[str] = Field(
        None,
        description="The unknown token to use. Will be ignored if a model_name_or_path is provided.",
    )
    bos_token: Optional[str] = Field(
        None,
        description="The beginning of sentence token to use. Will be ignored if a model_name_or_path is provided.",
    )
    eos_token: Optional[str] = Field(
        None,
        description="The end of sentence token to use. Will be ignored if a model_name_or_path is provided.",
    )
    pad_token: Optional[str] = Field(
        None,
        description="The padding token to use. Will be ignored if a model_name_or_path is provided.",
    )
    cls_token: Optional[str] = Field(
        None,
        description="The classification token to use. Will be ignored if a model_name_or_path is provided.",
    )
    sep_token: Optional[str] = Field(
        None,
        description="The separator token to use. Will be ignored if a model_name_or_path is provided.",
    )
    mask_token: Optional[str] = Field(
        None,
        description="The mask token to use. Will be ignored if a model_name_or_path is provided.",
    )
    additional_special_tokens: Optional[list] = Field(
        None,
        description="The additional special tokens to use. Will be ignored if a model_name_or_path is provided.",
    )
    add_special_tokens: bool = Field(
        True,
        description="Whether to add special tokens when encoding sequences.",
    )
    padding: str = Field(
        None,
        description="Whether to pad the input sequence to the maximum length.",
    )
    padding_side: str = Field(
        "right",
        description="The side on which to pad.",
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
    special_tokens_map: dict = Field(
        None,
        description="A dictionary mapping special token class names to special tokens (string) or special token "
        "attributes (dictionary).",
    )
    __tokenizer_obj__ = None

    class Config:
        extra = "allow"
        use_enum_values = True

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if (
            self.tokenizer_name is None
            and self.tokenizer_name_or_path is None
            and self.pretrained_tokenizer_file is None
        ):
            raise ValueError(
                "You must provide a tokenizer name, a tokenizer name or path or a pretrained tokenizer file."
            )

    def initialize_config(
        self, model: ModelConfig, dataset: DatasetConfig, root_dir: Union[str, Path]
    ):
        if self.tokenizer_name is None:
            self.tokenizer_name = model.model_name
        if self.model_type is None:
            self.model_type = model.model_type
        if self.model_dir is None:
            self.model_dir = model.model_dir
        self.tokenizer_dir = Path(self.tokenizer_dir or "tokenizers")
        if not self.tokenizer_dir.is_absolute():
            self.tokenizer_dir = Path(root_dir) / self.tokenizer_dir
        self.model_max_length = dataset.max_seq_length
        self.cache_dir = str(model.cache_dir)
        self.use_auth_token = model.use_auth_token
        self.ignore_existing_model_output = model.ignore_existing_model_output

    @validator("model_dir")
    def _check_model_dir(cls, v):
        if isinstance(v, str):
            v = Path(v)
        return v

    @validator("tokenizer_dir")
    def _check_tokenizer_dir(cls, v):
        if isinstance(v, str):
            v = Path(v)
        return v

    @property
    def special_tokens(self):
        if self.special_tokens_map:
            return self.special_tokens_map
        tokens = dict(
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            mask_token=self.mask_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            cls_token=self.cls_token,
            sep_token=self.sep_token,
        )
        # remove None values
        tokens = {k: v for k, v in tokens.items() if v is not None}
        return tokens

    @property
    def special_token_ids(self):
        token_ids = dict(
            unk_token_id=self.tokenizer_obj.unk_token_id,
            pad_token_id=self.tokenizer_obj.pad_token_id,
            mask_token_id=self.tokenizer_obj.mask_token_id,
            bos_token_id=self.tokenizer_obj.bos_token_id,
            eos_token_id=self.tokenizer_obj.eos_token_id,
            cls_token_id=self.tokenizer_obj.cls_token_id,
            sep_token_id=self.tokenizer_obj.sep_token_id,
        )
        return token_ids

    @property
    def vocab_size(self):
        return len(self.tokenizer_obj)

    @property
    def tokenizer_obj(self):
        if self.__tokenizer_obj__ is None:
            if self.pretrained_tokenizer_path is not None:
                self.prepare_tokenizer()
            self.load_tokenizer()
        return self.__tokenizer_obj__

    @property
    def model_output_dir(self):
        if self.tokenizer_name is None:
            return None
        model_path = self.tokenizer_name
        if not Path(model_path).is_absolute() or self.model_dir is not None:
            model_path = str(Path(self.model_dir) / model_path)
        return model_path

    @property
    def pretrained_tokenizer_path(self):
        if self.pretrained_tokenizer_file is not None:
            if (
                Path(self.pretrained_tokenizer_file).is_absolute()
                or self.tokenizer_dir is None
            ):
                return self.pretrained_tokenizer_file
            else:
                return str(self.tokenizer_dir / self.pretrained_tokenizer_file)
        return None

    def prepare_tokenizer(self):
        from tokenizers import processors

        # from tokenizers.processors import BertProcessing

        if self.pretrained_tokenizer_path is None:
            raise ValueError("pretrained_tokenizer_file is required")

        tok = Tokenizer.from_file(self.pretrained_tokenizer_path)
        if self.model_type == "bert":
            sep_token = self.sep_token
            cls_token = self.cls_token
            tok.post_processor = processors.BertProcessing(
                sep=(sep_token, tok.token_to_id(sep_token)),
                cls=(cls_token, tok.token_to_id(cls_token)),
            )
        elif self.model_type == "roberta":
            sep_token = self.eos_token
            cls_token = self.bos_token
            tok.post_processor = processors.BertProcessing(
                sep=(sep_token, tok.token_to_id(sep_token)),
                cls=(cls_token, tok.token_to_id(cls_token)),
            )
        elif self.model_type == "gpt2":
            tok.post_processor = processors.ByteLevel(trim_offsets=False)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            truncation=self.truncation,
            max_length=self.model_max_length,
            return_length=self.return_length,
            padding_side=self.padding_side,
            **self.special_tokens,
        )

        logger.info(f"Is a fast tokenizer? {tokenizer.is_fast}")
        logger.info(f"Vocab size: {tokenizer.vocab_size}")
        tokenizer.save_pretrained(self.model_output_dir)
        logger.info(f"Saved tokenizer to {self.model_output_dir}")
        self.tokenizer_name_or_path = self.model_output_dir

    def load_tokenizer(self):
        # Load tokenizer

        tokenizer_kwargs = {
            "cache_dir": self.cache_dir,
            "use_fast": self.use_fast_tokenizer,
            "revision": self.model_revision,
            "use_auth_token": self.use_auth_token,
        }
        tokenizer = None
        if (
            Path(self.model_output_dir).is_dir()
            and not self.ignore_existing_model_output
        ):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_output_dir, **tokenizer_kwargs
                )
            except OSError:
                logger.warning(
                    f"Couldn't load tokenizer from {self.model_output_dir}. Trying tokenizer_name_or_path."
                )

        if tokenizer is None:
            if self.tokenizer_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path, **tokenizer_kwargs
                )
            else:
                raise ValueError(
                    "You are instantiating a new tokenizer from scratch. This is not supported by this class."
                    "You can do it from the tokenizer trainer class."
                )

        if self.add_special_tokens:
            tokenizer.add_special_tokens(self.special_tokens)
        self.special_tokens_map = tokenizer.special_tokens_map
        self.__tokenizer_obj__ = tokenizer
