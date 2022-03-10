import pandas as pd
from omegaconf import OmegaConf
from ekorpkit.utils.func import elapsed_timer
from .dataset import Dataset


class Datasets:
    def __init__(self, **args):
        args = OmegaConf.create(args)
        self.args = args
        self.names = args.name
        if isinstance(self.names, str):
            self.names = [self.names]
        self.data_dir = args.data_dir
        self.split_info = self.args.splits
        self.datasets = {}
        self._id_key = "id"
        self._id_separator = "_"

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.names:
                print(f"processing {name}")
                data_dir = f"{self.data_dir}/{name}"
                args["data_dir"] = data_dir
                args["name"] = name
                dataset = Dataset(**args)
                self.datasets[name] = dataset
            print(f"\n >>> Elapsed time: {elapsed()} <<< ")

    def concat_datasets(self):
        self.splits = {}
        for split in self.split_info:
            self.splits[split] = pd.concat(
                [ds.splits[split] for ds in self.datasets.values()], ignore_index=True
            )

    def __getitem__(self, name):
        if name not in self.datasets:
            raise KeyError(f"{name} not in datasets")
        return self.datasets[name]
