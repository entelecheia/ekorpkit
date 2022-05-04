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
        self._data_files = self.args.get("data_files", None)
        self.filetype = self.args.get("filetype", "csv")
        self._call_ = self.args.get("_call_", None)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self.info_args = self.args.get("info", None)

        self.column_info = self.args.get("column_info", {})
        self.splits = None
        self._datasets_concatenated = False

        self._id_key = "id"
        self._org_id_key = "org_id"
        self._id_separator = "_"
        self._dataset_key = "dataset"
        self._keys = self.column_info.get("keys", None)
        self._id_keys = self._keys[self._id_key]
        if isinstance(self._id_keys, str):
            self._id_keys = [self._id_keys]
        self._data_keys = self.column_info.get("data", None)

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.datasets:
                log.info(f"processing {name}")
                args["name"] = name
                args["data_dir"] = self.data_dir
                args["use_name_as_subdir"] = use_name_as_subdir
                args["verbose"] = self.verbose
                if self._data_files is not None:
                    if name in self._data_files:
                        args["data_files"] = self._data_files[name]
                    elif "train" in self._data_files:
                        args["data_files"] = self._data_files
                dataset = Dataset(**args)
                self.datasets[name] = dataset
                if self.splits is None:
                    self.splits = {split: None for split in dataset.data_files}
            log.info(f">>> Elapsed time: {elapsed()} <<< ")

        eKonf.call(self._call_, self)

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
    def ID(self):
        return self._id_key

    @property
    def IDs(self):
        return self._id_keys

    @property
    def DATA(self):
        if self._data_keys is None:
            return None
        return list(self._data_keys.keys())

    def load(self):
        for _name in self.datasets:
            self.datasets[_name].load()
        self._loaded = True

    def concatenate(self, append_dataset_name=True):
        self.concat_datasets(append_dataset_name=append_dataset_name)

    def concat_datasets(self, append_dataset_name=True):
        if append_dataset_name:
            if self._dataset_key not in self._id_keys:
                self._id_keys.append(self._dataset_key)

        for split in self.splits:
            dfs = []
            for name in self.datasets:
                df = self.datasets[name][split]
                if self.DATA:
                    df = df[self.DATA].copy()
                if append_dataset_name:
                    df[self._dataset_key] = name
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
        if self.info_args:
            self.info_args["data_dir"] = data_dir
            self.info_args["name"] = self.name
            self.info_args["info_file"] = None
            summary_info = eKonf.instantiate(self.info_args)
        if summary_info:
            summary_info.load(self.info)

        for split, df in self.splits.items():
            data_file = f"{self.name}-{split}.{self.filetype}"
            data_path = f"{data_dir}/{data_file}"
            df.rename({self._id_key: self._org_id_key}, inplace=True)
            df.reset_index().rename({"index": self._id_key}, inplace=True)
            save_dataframe(df, data_path)
            if self.verbose:
                log.info(f"saved {data_path}")
            if summary_info:
                stats = {"data_file": data_file}
                summary_info.init_stats(split_name=split, stats=stats)
                summary_info.calculate_stats(df, split)
        if summary_info and df is not None:
            dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
            self.column_info["data"] = dtypes
            summary_info.save(info={"column_info": self.column_info})
