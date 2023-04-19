import logging

from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipeline
from hyfi.config import BaseBatchModel

logger = logging.getLogger(__name__)


class DatasetBuilder(BaseBatchModel):
    name: str
    filetype: str = ".parquet"
    features: dict = {}

    class Config:
        underscore_attrs_are_private = False
