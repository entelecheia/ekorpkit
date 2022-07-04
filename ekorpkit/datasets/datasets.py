import os
import pandas as pd
import logging
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from .dataset import Dataset
from .base import BaseSet


log = logging.getLogger(__name__)


class Datasets(BaseSet):

    SPLITS = eKonf.SPLITS

    def __init__(self, **args):
        super().__init__(**args)
        self.datasets = args.get("datasets", None)
        self.datasets = eKonf.to_dict(self.datasets)
        if self.datasets is None:
            self.datasets = self.name
        if isinstance(self.datasets, str):
            self.datasets = {self.datasets: None}
        elif eKonf.is_list(self.datasets):
            self.datasets = {name: None for name in self.datasets}
        if self.name is None and eKonf.is_dict(self.datasets):
            self.name = "-".join(self.datasets.keys())

        self._info = args.copy()
        self._info["name"] = self.name
        self._info["datasets"] = self.datasets

        self._method_ = self.args.get("_method_", None)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self.load_column_info()

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

    def load(self):
        for _name in self.datasets:
            self.datasets[_name].load()
        self._loaded = True

    def concat_datasets(self, append_dataset_name=True):
        self.concatenate(append_name=append_dataset_name)

    def concatenate(self, append_name=True):
        if not self._loaded:
            self.load()

        dfs = []
        for name in self.datasets:
            df = self.datasets[name][self.SPLITS.TRAIN]
            dfs.append(df)
        common_columns = self.COLUMN.common_columns(dfs)

        for split in self.splits:
            dfs = []
            for name in self.datasets:
                df = self.datasets[name][split]
                if df is None:
                    continue
                if common_columns:
                    df = df[common_columns].copy()
                if append_name:
                    df = self.COLUMN.append_dataset(df, name)
                dfs.append(df)
            if len(dfs) > 1:
                self._splits[split] = pd.concat(dfs, ignore_index=True)
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

        data_dir = os.path.join(self.data_dir, self.name)

        summary_info = None
        if self._info_cfg:
            self._info_cfg["data_dir"] = data_dir
            self._info_cfg["name"] = self.name
            self._info_cfg["info_file"] = None
            summary_info = eKonf.instantiate(self._info_cfg)
        if summary_info:
            summary_info.load(self._info)

        for split, data in self.splits.items():
            if data is None:
                continue
            data_file = f"{self.name}-{split}{self.filetype}"
            data = self.COLUMN.reset_id(data)
            eKonf.save_data(data, data_file, data_dir)
            if summary_info:
                stats = {"data_file": data_file}
                summary_info.init_stats(split_name=split, stats=stats)
                summary_info.calculate_stats(data, split)
        if summary_info:
            summary_info.save(info={"column_info": self.COLUMN.INFO})
