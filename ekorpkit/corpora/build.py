import logging
from pathlib import Path
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.pipelines.pipe import apply_pipeline


log = logging.getLogger(__name__)


def build_corpus(**args):
    cfg = args.get(eKonf.Keys.CORPUS, {}).get("builtin", None)
    # print(cfg)
    if cfg:
        DatasetBuilder(**cfg)


def build_t5(**args):
    cfg = args.get(eKonf.Keys.DATASET, {}).get("t5", None)
    # print(cfg)
    if cfg:
        DatasetBuilder(**cfg)


def build_simple(**args):
    cfg = args.get(eKonf.Keys.DATASET, {}).get("simple", None)
    # print(cfg)
    if cfg:
        DatasetBuilder(**cfg)


class DatasetBuilder:
    def __init__(self, **args) -> None:
        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.data_dir = args.data_dir
        self.data_filetype = args.get("filetype", ".parquet")
        self.column_info = self.args.column_info
        self.verbose = self.args.get("verbose", False)
        self.auto = self.args.auto

        self._io_ = args.io
        self.fetch_dir = self._io_.get("data_dir", None)
        self.fetch_sources = self._io_.get("data_sources", None)
        if isinstance(self.fetch_sources, str):
            self.fetch_sources = [self.fetch_sources]
        if isinstance(self.fetch_sources, list):
            self.fetch_sources = {"train": self.fetch_sources}
        self.num_workers = self._io_.get("num_workers", None)
        self.overwrite = self._io_.get("overwrite", False)
        self.calculate_stats = self._io_.get("calculate_stats", False)
        self.preprocess_text = self._io_.get("preprocess_text", False)

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

        if self.auto.load:
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

        log.info(
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
            self._pipeline_[pipe].column_info = self.column_info
            self._pipeline_[pipe].split_name = split_name
        pipe = "save_samples"
        if pipe in self._pipeline_:
            if pipe not in self.process_pipeline:
                self.process_pipeline.append(pipe)
            self._pipeline_[pipe].path.output = _sample_path_
        pipe = "save_dataframe"
        if pipe in self._pipeline_:
            if pipe not in self.process_pipeline:
                self.process_pipeline.append(pipe)
            self._pipeline_[pipe].filepath = _data_path_.filepath
            columns = self.column_info.data
            if columns:
                columns = list(columns.keys())
            self._pipeline_[pipe].columns = columns

        df = None
        if eKonf.exists(_data_path_.filepath) or self.overwrite:
            with elapsed_timer(format_time=True) as elapsed:
                df = eKonf.instantiate(self.loader, split_name=split_name)
                log.info(f" >> elapsed time to load and parse data: {elapsed()}")

            if df is None:
                raise ValueError("dataframe is None")

            if self.verbose:
                print(df.head())
                print(df.shape)

            if self.transform_pipeline and len(self.transform_pipeline) > 0:
                log.info(
                    f"\nTransforming dataframe with pipeline: {self.transform_pipeline}"
                )
                df = apply_pipeline(df, self.transform_pipeline, self._pipeline_)

            if self.summary_info and self.calculate_stats:
                stats = {
                    "name": split_name,
                    "dataset_name": self.name,
                    "data_file": _data_path_.filename,
                }
                if eKonf.exists(_meta_path_.filepath):
                    stats["meta_file"] = _meta_path_.filename
                self.summary_info.init_stats(df=df, split_name=split_name, stats=stats)

        else:
            log.info(f"{_data_path_.filepath} already exists")
            if self.calculate_stats or self.preprocess_text:
                df = eKonf.load_data(**_data_path_)

        if df is None:
            log.warning("No datasets found")
            return None

        if self.process_pipeline and len(self.process_pipeline) > 0:
            log.info(f"\nProcessing dataframe with pipeline: {self.process_pipeline}")
            df = apply_pipeline(df, self.process_pipeline, self._pipeline_)

        if self.calculate_stats and self.summary_info:
            self.summary_info.calculate_stats(df, split_name)
