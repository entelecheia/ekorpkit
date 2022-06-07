import logging
from pathlib import Path
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit.io.file import load_dataframe


log = logging.getLogger(__name__)


class Dataset:
    """Dataset class."""

    SPLITS = eKonf.SPLITS

    def __init__(self, **args):
        self.args = eKonf.to_dict(args)
        self.name = self.args["name"]
        if isinstance(self.name, list):
            self.name = self.name[0]
        self.verbose = self.args.get("verbose", False)
        self.autoload = self.args.get("autoload", False)
        use_name_as_subdir = args.get("use_name_as_subdir", True)

        self.data_dir = Path(self.args["data_dir"])
        if use_name_as_subdir:
            self.data_dir = self.data_dir / self.name
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

        self.splits = {}
        self._loaded = False

        if self.autoload:
            self.load()

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.name}"
        return s

    def __getitem__(self, split):
        return self.splits[split]

    @property
    def INFO(self):
        return self._info

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

    @property
    def DATATYPEs(self):
        return self.COLUMN.DATATYPEs

    def load(self):
        if self._loaded:
            return
        for split, data_file in self.data_files.items():
            data_file = self.data_dir / data_file
            df = load_dataframe(data_file, dtype=self.DATATYPEs)
            df = self.COLUMN.append_split(df, split)
            if self._pipeline_ and len(self._pipeline_) > 0:
                df = apply_pipeline(df, self._pipeline_, self._pipeline_cfg)
            self.splits[split] = df
        self._loaded = True
