import logging

from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipeline
from hyfi.config import BaseBatchModel
from hyfi.utils.func import elapsed_timer
from .config import CorpusFeatures

logger = logging.getLogger(__name__)


class DatasetBuilder(BaseBatchModel):
    name: str
    filetype: str = ".parquet"
    features: CorpusFeatures
    autoload: bool = False

    class Config:
        underscore_attrs_are_private = False

    def __init__(self, config_name: str = "_build_", **args):
        config_group = f"datasets/builder={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)

    def initialize_configs(self, **args):
        super().initialize_configs(**args)

        self._io_ = args.io
        self.fetch_dir = self._io_.get("data_dir", None)
        self.fetch_sources = self._io_.get("data_sources", None)
        if isinstance(self.fetch_sources, str):
            self.fetch_sources = [self.fetch_sources]
        if isinstance(self.fetch_sources, list):
            self.fetch_sources = {"train": self.fetch_sources}
        self.num_workers = self._io_.get("num_workers", None)
        self.force = self._io_.force

        self.fetcher = self._io_.get("fetcher", None)
        self.loader = self._io_.get("loader", None)

        self._info_ = self.args.get("info", None)
        self.summary_info = None

        self._pipeline_ = self.args.get("pipeline", {})
        self.transform_pipeline = self._pipeline_.get("_transform_", [])
        self.process_pipeline = self._pipeline_.get("_preprocess_", [])
        if self.transform_pipeline is None:
            self.transform_pipeline = []
        if self.process_pipeline is None:
            self.process_pipeline = []

        if self.autoload:
            self.build()

    def build(self):
        if self.fetcher:
            if eKonf.is_instantiatable(self.fetcher):
                eKonf.instantiate(self.fetcher)
            _pipeline_ = self.fetcher.get("pipeline", None)
            if _pipeline_:
                eKonf.instantiate(_pipeline_)

        if self._info_:
            self.summary_info = eKonf.instantiate(self._info_)
        if self.summary_info:
            self.summary_info.load(self.args)

        for split_name, split_data_source in self.fetch_sources.items():
            if split_data_source is None:
                continue
            self._process_split(split_name)

        if self.summary_info:
            self.summary_info.save()

        logger.info(
            f"\nCorpus [{self.name}] is built to [{self.data_dir}] from [{self.fetch_dir}]"
        )

    def _process_split(self, split_name):
        _data_path_ = self._io_.data.path[split_name]
        _meta_path_ = self._io_.meta.path[split_name]
        _sample_path_ = self._io_.sample.path[split_name]

        pipe = "save_metadata"
        if pipe in self._pipeline_:
            if pipe not in self.transform_pipeline:
                self.transform_pipeline.append(pipe)
            self._pipeline_[pipe].path.output = _meta_path_
            self._pipeline_[pipe].features = self.features
            self._pipeline_[pipe].split_name = split_name
        pipe = "save_samples"
        if pipe in self._pipeline_:
            if pipe not in self.process_pipeline:
                self.process_pipeline.append(pipe)
            self._pipeline_[pipe].path.output = _sample_path_

        df = None
        if not eKonf.exists(_data_path_.filepath) or self.force.build:
            with elapsed_timer(format_time=True) as elapsed:
                df = eKonf.instantiate(self.loader, split_name=split_name)
                logger.info(f" >> elapsed time to load and parse data: {elapsed()}")

            if df is None:
                raise ValueError("dataframe is None")

            if self.verbose:
                print(df.head())
                print(df.shape)

            if self.transform_pipeline and len(self.transform_pipeline) > 0:
                logger.info(
                    f"\nTransforming dataframe with pipeline: {self.transform_pipeline}"
                )
                df = apply_pipeline(df, self.transform_pipeline, self._pipeline_)

            # Saving data
            columns = self.features.data
            if columns:
                columns = list(columns.keys())
            _data_path_.columns = columns
            eKonf.save_data(df, **_data_path_)

            if self.summary_info and self.force.summarize:
                stats = {
                    "name": split_name,
                    "dataset_name": self.name,
                    "data_file": _data_path_.filename,
                }
                if eKonf.exists(_meta_path_.filepath):
                    stats["meta_file"] = _meta_path_.filename
                self.summary_info.init_stats(df=df, split_name=split_name, stats=stats)

        else:
            logger.info(f"{_data_path_.filepath} already exists")
            if self.force.summarize or self.force.preprocess:
                df = eKonf.load_data(**_data_path_)

        if df is None:
            logger.warning("No datasets found")
            return None

        if self.process_pipeline and len(self.process_pipeline) > 0:
            logger.info(
                f"\nProcessing dataframe with pipeline: {self.process_pipeline}"
            )
            df = apply_pipeline(df, self.process_pipeline, self._pipeline_)

        if self.force.summarize and self.summary_info:
            self.summary_info.calculate_stats(df, split_name)
