import logging
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)
from ekorpkit.models.transformer.trainers.base import BaseLMTrainer


logger = logging.getLogger(__name__)


class MlmTrainer(BaseLMTrainer):
    def __init__(self, config_name: str = "lm.mlm", **args):
        config_group = f"task={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)

    def fill_mask(self, text, top_k=5, reload=False, **kwargs):
        """Fill the mask in a text"""
        if reload or self.__pipe_obj__ is None:
            model = AutoModelForMaskedLM.from_pretrained(self.model_output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_output_dir)

            self.__pipe_obj__ = pipeline(
                "fill-mask", model=model, tokenizer=tokenizer, **kwargs
            )
        return self.__pipe_obj__(text, top_k=top_k)
