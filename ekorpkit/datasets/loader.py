import os
import pandas as pd
import logging
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.io.file import save_dataframe
from .dataset import Dataset, _SPLITS


log = logging.getLogger(__name__)


class Datasets:
    SPLITS = _SPLITS

    def __init__(self, **args):
        args = eKonf.to_dict(args)
        self.args = args
        self.name = args["name"]
        self.datasets = args.get("datasets", None)
        if self.datasets is None:
            self.datasets = self.name
        if isinstance(self.datasets, str):
            self.datasets = {self.datasets: None}
        elif isinstance(self.datasets, list):
            self.datasets = {name: None for name in self.datasets}
        if isinstance(self.name, list):
            self.name = "-".join(self.name)
        self.info = args.copy()
        self.info["name"] = self.name
        self.info["datasets"] = self.datasets

        self.verbose = args.get("verbose", False)
        self.data_dir = args["data_dir"]
        self.data_files = self.args.get("data_files", None)
        self.filetype = self.args.get("filetype", "csv")
        self._method_ = self.args.get("_method_", None)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self._info_args = self.args.get("info", None)

        self._column_info = self.args.get("column_info", {})
        self._column = eKonf.instantiate(self._column_info)

        self.splits = None
        self._datasets_concatenated = False

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.datasets:
                log.info(f"processing {name}")
                args["name"] = name
                args["data_dir"] = self.data_dir
                args["use_name_as_subdir"] = use_name_as_subdir
                args["verbose"] = self.verbose
                if self.data_files is not None:
                    if name in self.data_files:
                        args["data_files"] = self.data_files[name]
                    elif "train" in self.data_files:
                        args["data_files"] = self.data_files
                dataset = Dataset(**args)
                self.datasets[name] = dataset
                if self.splits is None:
                    self.splits = {split: None for split in dataset.data_files}
            log.info(f">>> Elapsed time: {elapsed()} <<< ")

        eKonf.methods(self._method_, self)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname}\n----------\n"
        for name in self.datasets.keys():
            s += f"{str(name)}\n"
        return s

    def __iter__(self):
        for dataset in self.datasets.values():
            yield dataset

    def __getitem__(self, name):
        if name not in self.datasets:
            raise KeyError(f"{name} not in datasets")
        return self.datasets[name]

    def __len__(self):
        return len(self.datasets)

    @property
    def COLUMN(self):
        return self._column

    @property
    def ID(self):
        return self.COLUMN.ID

    @property
    def IDs(self):
        return self.COLUMN.IDs

    @property
    def DATA(self):
        return self.COLUMN.DATA

    def load(self):
        for _name in self.datasets:
            self.datasets[_name].load()
        self._loaded = True

    def concatenate(self, append_dataset_name=True):
        self.concat_datasets(append_dataset_name=append_dataset_name)

    def concat_datasets(self, append_dataset_name=True):
        dfs = []
        for name in self.datasets:
            df = self.datasets[name][_SPLITS.TRAIN]
            dfs.append(df)
        common_columns = self.COLUMN.common_columns(dfs)

        for split in self.splits:
            dfs = []
            for name in self.datasets:
                df = self.datasets[name][split]
                if common_columns:
                    df = df[common_columns].copy()
                if append_dataset_name:
                    df = self.COLUMN.append_dataset(df, name)
                dfs.append(df)
            self.splits[split] = pd.concat(dfs, ignore_index=True)
        if self.verbose:
            log.info(f"concatenated {len(self.datasets)} dataset(s)")
        self._datasets_concatenated = True

    def persist(self):
        if len(self.datasets) < 2:
            log.warning(f"more than one dataset required to persist")
            return
        if not self._datasets_concatenated:
            log.warning(f"datasets not concatenated yet, calling concatenate()")
            self.concatenate()

        data_dir = f"{self.data_dir}/{self.name}"
        os.makedirs(data_dir, exist_ok=True)

        summary_info = None
        if self._info_args:
            self._info_args["data_dir"] = data_dir
            self._info_args["name"] = self.name
            self._info_args["info_file"] = None
            summary_info = eKonf.instantiate(self._info_args)
        if summary_info:
            summary_info.load(self.info)

        for split, df in self.splits.items():
            data_file = f"{self.name}-{split}.{self.filetype}"
            data_path = f"{data_dir}/{data_file}"
            df = self.COLUMN.reset_id(df)
            save_dataframe(df, data_path)
            if self.verbose:
                log.info(f"saved {data_path}")
            if summary_info:
                stats = {"data_file": data_file}
                summary_info.init_stats(split_name=split, stats=stats)
                summary_info.calculate_stats(df, split)
        if summary_info and df is not None:
            summary_info.save(info={"column_info": self.COLUMN.INFO})
