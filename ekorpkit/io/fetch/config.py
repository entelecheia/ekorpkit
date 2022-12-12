import logging
import os
from ekorpkit import eKonf
from ekorpkit.config import BaseBatchModel, BaseBatchConfig


log = logging.getLogger(__name__)


class BaseFatchConfig(BaseBatchConfig):
    limit: int = -1
    force_download: bool = False
    compress: bool = False
    output_extention: str = "parquet"


class BaseFetcher(BaseBatchModel):
    batch: BaseFatchConfig = None
    __data__ = None

    def __init__(self, **args):
        super().__init__(**args)

    def initialize_configs(self, **kwargs):
        super().initialize_configs(batch_config_class=BaseFatchConfig, **kwargs)

    def fetch(self):
        if not self.exsits() or self.batch.force_download:
            self.fetch_data()
        else:
            log.info(f"{self.output_file} already exists. skipping..")
            self.load_data()

    def load_data(self):
        self.__data__ = eKonf.load_data(self.output_file)

    def exsits(self):
        return os.path.exists(self.output_file)

    def fetch_data(self):
        raise NotImplementedError

    @property
    def data(self):
        return self.__data__

    @property
    def output_file(self):
        return str(self.batch.batch_dir / self.batch.output_file)
