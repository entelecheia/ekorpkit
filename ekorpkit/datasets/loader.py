import pandas as pd
from wasabi import msg
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from .dataset import Dataset


class Datasets:
    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.names = args.name
        if isinstance(self.names, str):
            self.names = [self.names]
        self.data_dir = args.data_dir
        self.verbose = args.get("verbose", False)
        self.column_info = eKonf.to_dict(self.args.column_info)
        self.split_info = eKonf.to_dict(self.args.splits)
        self.datasets = {}

        self._id_key = "id"
        self._id_separator = "_"
        self._dataset_key = "dataset"
        self._keys = self.column_info["keys"]
        self._id_keys = self._keys[self._id_key]
        if isinstance(self._id_keys, str):
            self._id_keys = [self._id_keys]
        self._data_keys = self.column_info.get("data", None)

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.names:
                print(f"processing {name}")
                data_dir = f"{self.data_dir}/{name}"
                args["data_dir"] = data_dir
                args["name"] = name
                dataset = Dataset(**args)
                self.datasets[name] = dataset
            print(f"\n >>> Elapsed time: {elapsed()} <<< ")

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname}\n----------\n"
        for name in self.corpora.keys():
            s += f"{str(name)}\n"
        return s

    def __getitem__(self, name):
        if name not in self.datasets:
            raise KeyError(f"{name} not in datasets")
        return self.datasets[name]

    @property
    def ID(self):
        return self._id_key

    @property
    def IDs(self):
        return self._id_keys

    @property
    def DATA(self):
        return list(self._data_keys.keys())

    def __getitem__(self, split):
        return self.splits[split]

    def concat_datasets(self, append_dataset_name=True):
        self.splits = {}

        if append_dataset_name:
            if self._dataset_key not in self._id_keys:
                self._id_keys.append(self._dataset_key)

        for split in self.split_info:
            dfs = []
            for name in self.datasets:
                df = self.datasets[name][split]
                if self.DATA:
                    df = df[self.DATA]
                if append_dataset_name:
                    df[self._dataset_key] = name
                dfs.append(df)
            self.splits[split] = pd.concat(dfs, ignore_index=True)
        if self.verbose:
            msg.good(f"concatenated {len(self.datasets)} datasets")
            print(self._data.head())
