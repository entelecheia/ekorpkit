import logging
from pydantic import Field
from pathlib import Path
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
from ..config import ModelConfig


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


class LMModelConfig(ModelConfig):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: Optional[str] = Field(
        default=None,
        description="If training from scratch, pass a model type from the list: "
        + ", ".join(MODEL_TYPES),
    )
    model_objective: Optional[ModelObjectives] = Field(
        default=ModelObjectives.mlm,
        description="The objective to use when training the model.",
    )

    class Config:
        extra = "allow"
        use_enum_values = True
        underscore_attrs_are_private = True

    def __init__(self, **kw):
        super().__init__(**kw)

    def load_model(self, tokenizer_obj, model_name=None):
        # Load pretrained model
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if model_name is not None:
            self.model_name = model_name
            self.ignore_existing_model_output = False

        AutoModelForLM = LM_MAPPING[self.model_objective]
        model = None
        if (
            Path(self.model_output_dir).is_dir()
            and not self.ignore_existing_model_output
        ):
            try:
                model = AutoModelForLM.from_pretrained(
                    self.model_output_dir,
                    from_tf=bool(".ckpt" in self.model_name_or_path),
                )
            except OSError:
                logger.warning(
                    f"Model {self.model_output_dir} not found. Trying model_name_or_path instead."
                )
        if model is None and self.use_model_name_or_path:
            if self.model_name_or_path is not None:

                model = AutoModelForLM.from_pretrained(
                    self.model_name_or_path,
                    from_tf=bool(".ckpt" in self.model_name_or_path),
                    config=self.auto_config,
                    cache_dir=self.cache_dir,
                    revision=self.model_revision,
                    use_auth_token=True if self.use_auth_token else None,
                )
            else:
                auto_config = self.auto_config.to_dict()
                original_config, update_config = {}, {}
                for k, v in tokenizer_obj.special_token_ids.items():
                    if k in auto_config and v != auto_config[k]:
                        original_config[k] = auto_config[k]
                        update_config[k] = v
                if auto_config["vocab_size"] != tokenizer_obj.vocab_size:
                    original_config["vocab_size"] = auto_config["vocab_size"]
                    update_config["vocab_size"] = self.tokenizer.vocab_size
                if update_config:
                    logger.info(
                        f"Overriding original model config {original_config} with {update_config}"
                    )
                    self.auto_config.update(update_config)
                    if self.verbose > 1:
                        logger.info(f"New model config {auto_config}")
                logger.info("Training new model from scratch")
                model = AutoModelForLM.from_config(self.auto_config)

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if tokenizer_obj.vocab_size > embedding_size:
            model.resize_token_embeddings(tokenizer_obj.vocab_size)
            logger.info(
                f"Resized embedding from {embedding_size} to {tokenizer_obj.vocab_size}"
            )

        self.__model_obj__ = model


class LMTrainingDatasetConfig(DatasetConfig):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

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
