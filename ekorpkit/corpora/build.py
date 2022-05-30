import logging
from pathlib import Path
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit.io.file import load_dataframe
from hydra.utils import instantiate

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
        self.args = eKonf.to_dict(args)
        self.name = args.get("name", None)
        self.data_dir = args.get("data_dir", None)
        self.data_filetype = args.get("filetype", "parquet")
        self.column_info = self.args.get("column_info", None)
        self.verbose = self.args.get("verbose", False)
        self.autoload = self.args.get("autoload", False)

        self.io_args = args.get("io", None)
        self.fetch_dir = self.io_args.get("data_dir", None)
        self.fetch_sources = self.io_args.get("data_sources", None)
        if isinstance(self.fetch_sources, str):
            self.fetch_sources = [self.fetch_sources]
        if isinstance(self.fetch_sources, list):
            self.fetch_sources = {"train": self.fetch_sources}
        self.num_workers = self.io_args.get("num_workers", None)
        self.overwrite = self.io_args.get("overwrite", False)
        self.calculate_stats = self.io_args.get("calculate_stats", False)
        self.preprocess_text = self.io_args.get("preprocess_text", False)

        self.fetcher = self.io_args.get("fetcher", None)
        self.loader = self.io_args.get("loader", None)

        self.info_args = self.args.get("info", None)
        self.summary_info = None

        self.pipeline_args = self.args.get("pipeline", {})
        self.transform_pipeline = self.pipeline_args.get("_transform_", [])
        self.process_pipeline = self.pipeline_args.get("_preprocess_", [])
        if self.transform_pipeline is None:
            self.transform_pipeline = []
        if self.process_pipeline is None:
            self.process_pipeline = []

        if self.autoload:
            self.build()

    def build(self):
        if self.fetcher:
            if self.fetcher.get("_target_", None):
                eKonf.instantiate(self.fetcher)
            pipeline_args = self.fetcher.get("pipeline", None)
            if pipeline_args:
                eKonf.instantiate(pipeline_args)

        if self.info_args:
            self.summary_info = eKonf.instantiate(self.info_args)
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

        output_dir = Path(self.data_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{self.name}-{split_name}{self.data_filetype}"
        output_meta_file = (
            output_dir / f"meta-{self.name}-{split_name}{self.data_filetype}"
        )
        sample_file_prefix = f"{str(output_dir)}/sample-{self.name}-{split_name}"
        pipe = "save_metadata"
        if pipe in self.pipeline_args:
            if pipe not in self.transform_pipeline:
                self.transform_pipeline.append(pipe)
            self.pipeline_args[pipe]["filepath"] = str(output_meta_file)
            self.pipeline_args[pipe]["column_info"] = self.column_info
            self.pipeline_args[pipe]["split_name"] = split_name
        pipe = "save_samples"
        if pipe in self.pipeline_args:
            if pipe not in self.process_pipeline:
                self.process_pipeline.append(pipe)
            self.pipeline_args[pipe]["sample_file_prefix"] = sample_file_prefix
        pipe = "save_dataframe"
        if pipe in self.pipeline_args:
            if pipe not in self.process_pipeline:
                self.process_pipeline.append(pipe)
            self.pipeline_args[pipe]["filepath"] = str(output_file)
            columns = self.column_info.get("data")
            if columns:
                columns = list(columns.keys())
            self.pipeline_args[pipe]["columns"] = columns

        df = None
        if not output_file.exists() or self.overwrite:
            with elapsed_timer(format_time=True) as elapsed:
                df = instantiate(self.loader, split_name=split_name, _recursive_=False)
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
                df = apply_pipeline(df, self.transform_pipeline, self.pipeline_args)

            if self.summary_info and self.calculate_stats:
                stats = {
                    "name": split_name,
                    "dataset_name": self.name,
                    "data_file": output_file.name,
                }
                if output_meta_file.is_file():
                    stats["meta_file"] = output_meta_file.name
                self.summary_info.init_stats(df=df, split_name=split_name, stats=stats)

        else:
            log.info(f"{output_file} already exists")
            if self.calculate_stats or self.preprocess_text:
                df = load_dataframe(output_file, self.data_filetype)

        if df is None:
            log.warning("No datasets found")
            return None

        if self.process_pipeline and len(self.process_pipeline) > 0:
            log.info(f"\nProcessing dataframe with pipeline: {self.process_pipeline}")
            df = apply_pipeline(df, self.process_pipeline, self.pipeline_args)

        if self.calculate_stats and self.summary_info:
            self.summary_info.calculate_stats(df, split_name)
