import pandas as pd
import logging
from pathlib import Path
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit.io.file import load_dataframe, save_dataframe


log = logging.getLogger(__name__)


class FeatureSet:
    """Feature class."""

    SPLITS = eKonf.SPLITS

    def __init__(self, **args):
        self.args = eKonf.to_dict(args)
        self.name = self.args["name"]
        if self.name is None:
            raise Exception("Feature name is required")
        if isinstance(self.name, list):
            self.name = self.name[0]
        self.verbose = self.args.get("verbose", False)
        self.autoload = self.args.get("autoload", False)
        self.autobuild = self.args.get("autobuild", False)
        self.force_rebuild = self.args.get("force_rebuild", False)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self.data_dir = Path(self.args["data_dir"])
        if use_name_as_subdir:
            self.data_dir = self.data_dir / self.name

        self._info_cfg = self.args.get("info", None)

        self.info_file = self.data_dir / f"info-{self.name}.yaml"
        self._info = eKonf.load(self.info_file) if self.info_file.is_file() else {}
        if self._info:
            log.info(f"Loaded info file: {self.info_file}")
            self.args = eKonf.to_dict(eKonf.merge(self.args, self._info))
            self._info = eKonf.to_dict(self._info)

        if self.verbose:
            print(f"Intantiating a dataset {self.name} with a config:")
            eKonf.print(self.args)

        self.filetype = self.args.get("filetype", "parquet").replace(".", "")
        self.data_files = self.args.get("data_files", None)
        if self.data_files is None:
            self.data_files = {
                self.SPLITS.TRAIN.value: f"{self.name}-train.{self.filetype}",
                self.SPLITS.DEV.value: f"{self.name}-dev.{self.filetype}",
                self.SPLITS.TEST.value: f"{self.name}-test.{self.filetype}",
            }

        self.description = self.args.get("description", "")
        self.license = self.args.get("license", "")
        self._column_info = self.args.get("column_info")
        if self._column_info is None:
            raise ValueError("Column info can't be None")

        self._column = eKonf.instantiate(self._column_info)

        self._pipeline_cfg = self.args.get("pipeline", {})
        self._pipeline_ = self._pipeline_cfg.get(eKonf.Keys.PIPELINE, [])
        if self._pipeline_ is None:
            self._pipeline_ = []

        self._splits = {}
        self._loaded = False

        if self.autobuild:
            if self.force_rebuild or not eKonf.exists(
                self.data_dir, self.data_files[self.SPLITS.TRAIN]
            ):
                self.build()
        if self.autoload:
            self.load()

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.name}"
        return s

    def __getitem__(self, split):
        return self.splits[split]

    @property
    def splits(self):
        return self._splits

    @property
    def X_train(self):
        _data = self.splits[self.SPLITS.TRAIN]
        if _data is not None:
            return _data[self.FEATURE.X]
        else:
            return None

    @property
    def X_dev(self):
        if self.SPLITS.DEV not in self.splits:
            return None
        _data = self.splits[self.SPLITS.DEV]
        if _data is not None:
            return _data[self.FEATURE.X]
        else:
            return None

    @property
    def X_test(self):
        if self.SPLITS.TEST not in self.splits:
            return None
        _data = self.splits[self.SPLITS.TEST]
        if _data is not None:
            return _data[self.FEATURE.X]
        else:
            return None

    @property
    def y_train(self):
        _data = self.splits[self.SPLITS.TRAIN]
        if _data is not None:
            return _data[self.FEATURE.Y]
        else:
            return None

    @property
    def y_dev(self):
        if self.SPLITS.DEV not in self.splits:
            return None
        _data = self.splits[self.SPLITS.DEV]
        if _data is not None:
            return _data[self.FEATURE.Y]
        else:
            return None

    @property
    def y_test(self):
        if self.SPLITS.TEST not in self.splits:
            return None
        _data = self.splits[self.SPLITS.TEST]
        if _data is not None:
            return _data[self.FEATURE.Y]
        else:
            return None

    @property
    def X(self):
        return self.data[self.FEATURE.X]

    @property
    def y(self):
        return self.data[self.FEATURE.Y]

    @property
    def data(self):
        dfs = []
        for split, _data in self.splits.items():
            if _data is not None:
                dfs.append(_data)
        df = pd.concat(dfs)
        return df

    @property
    def INFO(self):
        return self._info

    @property
    def FEATURE(self):
        return self._column

    @property
    def ID(self):
        return self.FEATURE.ID

    @property
    def DATATYPEs(self):
        return self.FEATURE.DATATYPEs

    def load(self):
        if self._loaded:
            return
        for split, data_file in self.data_files.items():
            data_file = self.data_dir / data_file
            if eKonf.exists(data_file):
                df = load_dataframe(data_file, dtype=self.DATATYPEs)
                df = self.FEATURE.init_info(df)
                df = self.FEATURE.append_split(df, split)
                self._splits[split] = df
            else:
                log.info(f"Dataset {self.name} split {split} is empty")
        self._loaded = True

    def build(self):
        data = None
        if self._pipeline_ and len(self._pipeline_) > 0:
            data = apply_pipeline(data, self._pipeline_, self._pipeline_cfg)
        if data is not None:
            log.info(f"Dataset {self.name} built with {len(data)} rows")
        else:
            log.info(f"Dataset {self.name} is empty")

    def persist(self):
        summary_info = None
        if self._info_cfg:
            summary_info = eKonf.instantiate(self._info_cfg)
        if summary_info:
            summary_info.load(self.INFO)

        for split, data in self._splits.items():
            if data is None:
                continue
            data_file = self.data_files[split]
            save_dataframe(
                data,
                output_dir=self.data_dir,
                output_file=data_file,
                verbose=self.verbose,
            )
            if summary_info:
                stats = {"data_file": data_file}
                summary_info.init_stats(split_name=split, stats=stats)
                summary_info.calculate_stats(data, split)
        if summary_info and data is not None:
            summary_info.save(info={"column_info": self.FEATURE.INFO})
