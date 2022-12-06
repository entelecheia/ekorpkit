import logging
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)
from ekorpkit.models.transformer.trainers.base import BaseTrainer


logger = logging.getLogger(__name__)


class MlmTrainer(BaseTrainer):
    def __init__(self, config_name: str = "mlm", **args):
        config_group = f"task/nlp/lm={config_name}"
        super().__init__(config_group=config_group, **args)

    def fill_mask(self, text, top_k=5, reload=False, **kwargs):
        """Fill the mask in a text"""
        if reload or self.__pipe_obj__ is None:
            model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            self.__pipe_obj__ = pipeline(
                "fill-mask", model=model, tokenizer=tokenizer, **kwargs
            )
        return self.__pipe_obj__(text, top_k=top_k)
