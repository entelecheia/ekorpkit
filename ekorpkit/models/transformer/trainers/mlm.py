import logging
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    pipeline,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from .base import BaseTrainer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)


class MlmTrainer(BaseTrainer):
    def __init__(self, config_group: str = "transformer=mlm.trainer", **args):
        super().__init__(config_group, **args)

    def fill_mask(self, text, top_k=5, reload=False, **kwargs):
        """Fill the mask in a text"""
        if reload or self.__pipe_obj__ is None:
            model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            self.__pipe_obj__ = pipeline(
                "fill-mask", model=model, tokenizer=tokenizer, **kwargs
            )
        return self.__pipe_obj__(text, top_k=top_k)
