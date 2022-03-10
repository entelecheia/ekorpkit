from tabnanny import verbose
import time
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.utils.func import humanbytes, get_modified_time
from ekorpkit.pipelines.stat import summary_stats
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit.io.file import load_dataframe
from pprint import pprint
from hydra.utils import instantiate
from wasabi import msg


def build_corpus(**args):
    cfg = args.get("corpus", {}).get("builtin", None)
    # print(cfg)
    if cfg:
        db = DatasetBuilder(**cfg)
        db.build()


def build_t5(**args):
    cfg = args.get("dataset", {}).get("t5", None)
    # print(cfg)
    if cfg:
        db = DatasetBuilder(**cfg)
        db.build()


def build_simple(**args):
    cfg = args.get("dataset", {}).get("simple", None)
    # print(cfg)
    if cfg:
        db = DatasetBuilder(**cfg)
        db.build()


class DatasetBuilder:
    def __init__(self, **args) -> None:
        self.args = args
        self.name = args.get("name", None)
        self.data_dir = args.get("data_dir", None)
        self.data_filetype = args.get("filetype", "parquet")
        self.column_info = self.args.get("column_info", None)

        self.fetch_args = args.get("fetch", None)
        if isinstance(self.fetch_args, DictConfig):
            self.fetch_args = OmegaConf.to_container(self.fetch_args)
        self.fetch_dir = self.fetch_args.get("data_dir", None)
        self.fetch_sources = self.fetch_args.get("data_sources", None)
        if isinstance(self.fetch_sources, str):
            self.fetch_sources = [self.fetch_sources]
        if isinstance(self.fetch_sources, list):
            self.fetch_sources = {"train": self.fetch_sources}
        self.num_workers = self.fetch_args.get("num_workers", None)
        self.overwrite = self.fetch_args.get("overwrite", False)
        self.calculate_stats = self.fetch_args.get("calculate_stats", False)
        self.preprocess_text = self.fetch_args.get("preprocess_text", False)

        self.downloader = self.fetch_args.get("downloader", None)
        self.loader = self.fetch_args.get("loader", None)

        self.info_args = self.args.get("info", None)
        self._load_info()

        self.stat_args = self.info_args.get("stat", None)
        self.pipeline_args = self.args.get("pipeline", {})
        self.transform_pipeline = self.pipeline_args.get("_transform_", [])
        self.process_pipeline = self.pipeline_args.get("_preprocess_", [])
        if self.transform_pipeline is None:
            self.transform_pipeline = []
        if self.process_pipeline is None:
            self.process_pipeline = []

    def _load_info(self):
        self.info_path = Path(self.data_dir) / self.info_args.get(
            "info_file", f"info-{self.name}.yaml"
        )
        if self.info_path.exists():
            self.info = OmegaConf.to_container(OmegaConf.load(self.info_path))
        else:
            self.info = {}
        for key in self.info_args.info_list:
            if self.args.get(key) is not None:
                self.info[key] = self.args[key]

    def _update_info(self, split_infos):
        self.info["splits"] = split_infos

        files_info = {key: {} for key in self.info_args.update_files_info}
        for i, (split_name, split_info) in enumerate(split_infos.items()):
            for key, val in self.info_args.update_files_info.items():
                if val in split_info:
                    files_info[key][split_name] = split_info[val]
            for key, val in self.info_args.aggregate_info.items():
                if val in split_info:
                    if i == 0:
                        self.info[key] = split_info[val]
                    else:
                        self.info[key] += split_info[val]
        self.info["size_in_human_bytes"] = humanbytes(self.info["size_in_bytes"])

        for key in self.info_args.update_files_info:
            self.info[key] = files_info[key]

        for key, value in self.info_args.modified_info.items():
            vals = [
                get_modified_time(f"{self.data_dir}/{split[value]}")
                for split in self.info["splits"].values()
                if value in split
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                self.info[key] = max(vals)
        self.info["info_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        OmegaConf.save(config=OmegaConf.create(self.info), f=self.info_path)

        pprint(self.info)

    def build(self):
        if self.downloader and self.downloader.get("_target_", None):
            instantiate(self.downloader, _recursive_=False)

        split_infos = {}
        for split_name, split_data_source in self.fetch_sources.items():
            if split_data_source is None:
                continue
            split_info = self._process_split(split_name)
            if split_info:
                split_infos[split_name] = split_info

        if split_infos:
            self._update_info(split_infos)
            print(
                f"\nCorpus [{self.name}] is built to [{self.data_dir}] from [{self.fetch_dir}]"
            )
        else:
            print(f"\nNo splits found for [{self.name}]")

    def _process_split(self, split_name):

        output_dir = Path(self.data_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"{self.name}-{split_name}{self.data_filetype}"
        output_meta_file = (
            output_dir / f"meta-{self.name}-{split_name}{self.data_filetype}"
        )
        sample_file_prefix = f"{str(output_dir)}/sample-{self.name}-{split_name}-"
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

        df = None
        if not output_file.exists() or self.overwrite:
            with elapsed_timer(format_time=True) as elapsed:
                df = instantiate(self.loader, split_name=split_name, _recursive_=False)
                msg.good(f" >> elapsed time to load and parse data: {elapsed()}")

            if df is None:
                raise ValueError("dataframe is None")

            if verbose:
                print(df.head())
                print(df.shape)

            if self.transform_pipeline and len(self.transform_pipeline) > 0:
                print(
                    f"\nTransforming dataframe with pipeline: {self.transform_pipeline}"
                )
                df = apply_pipeline(df, self.transform_pipeline, self.pipeline_args)

            info = {
                "name": split_name,
                "dataset_name": self.name,
                "data_file": output_file.name,
                # 'meta_file': None,
            }
            if output_meta_file.is_file():
                info["meta_file"] = output_meta_file.name

            stat_args = self.info_args.get("stat_before_processing", None)
            if stat_args:
                with elapsed_timer(format_time=True) as elapsed:
                    for k, v in self.stat_args.items():
                        if k not in stat_args:
                            stat_args[k] = v
                    stat_info = summary_stats(df, **stat_args)
                    msg.good(
                        f" >> elapsed time to calculate statistics before processing: {elapsed()}"
                    )
                    if verbose:
                        pprint(stat_info)

                info.update(stat_info)
        else:
            msg.info(f"{output_file} already exists")
            if self.calculate_stats or self.preprocess_text:
                df = load_dataframe(output_file, self.data_filetype)
                info = self.info["splits"][split_name]

        if df is None:
            print("No datasets found")
            return None

        if self.process_pipeline and len(self.process_pipeline) > 0:
            print(f"\nProcessing dataframe with pipeline: {self.process_pipeline}")
            df = apply_pipeline(df, self.process_pipeline, self.pipeline_args)

        if self.calculate_stats:
            with elapsed_timer(format_time=True) as elapsed:
                stat_info = summary_stats(df, **self.stat_args)
                msg.good(f" >> elapsed time to calculate statistics: {elapsed()}")
                if verbose:
                    pprint(stat_info)

            info.update(stat_info)
            return info
