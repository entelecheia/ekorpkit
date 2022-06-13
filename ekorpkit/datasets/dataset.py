import os
import logging
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit.corpora.corpus import BaseSet


log = logging.getLogger(__name__)


class Dataset(BaseSet):
    """Dataset class."""

    SPLITS = eKonf.SPLITS

    def __init__(self, **args):
        super().__init__(**args)
        if isinstance(self.name, list):
            self.name = self.name[0]
        use_name_as_subdir = args.get("use_name_as_subdir", True)
        if use_name_as_subdir:
            self.data_dir = os.path.join(self.data_dir, self.name)

        self.load_info()
        self.load_column_info()

        if self.data_files is None:
            self.data_files = {
                self.SPLITS.TRAIN.value: f"{self.name}-train{self.filetype}",
                self.SPLITS.DEV.value: f"{self.name}-dev{self.filetype}",
                self.SPLITS.TEST.value: f"{self.name}-test{self.filetype}",
            }

        self._pipeline_cfg = self.args.get("pipeline", {})
        self._pipeline_ = self._pipeline_cfg.get(eKonf.Keys.PIPELINE, [])
        if self._pipeline_ is None:
            self._pipeline_ = []

        self._splits = {}
        self.force = self.args.force

        if self.auto.build:
            if self.force.rebuild or not eKonf.exists(
                self.data_dir, self.data_files[self.SPLITS.TRAIN]
            ):
                self.build()
        if self.auto.load:
            self.load()

    @property
    def splits(self):
        return self._splits

    def __getitem__(self, split):
        if split in self.splits:
            return self.splits[split]
        else:
            return None

    @property
    def data(self):
        dfs = []
        for split, _data in self.splits.items():
            if _data is not None:
                dfs.append(_data)
        df = eKonf.concat_data(dfs)
        return df

    @property
    def train_data(self):
        if self.SPLITS.TRAIN not in self.splits:
            return None
        return self.splits[self.SPLITS.TRAIN]

    @property
    def dev_data(self):
        if self.SPLITS.DEV not in self.splits:
            return None
        return self.splits[self.SPLITS.DEV]

    @property
    def test_data(self):
        if self.SPLITS.TEST not in self.splits:
            return None
        return self.splits[self.SPLITS.TEST]

    def load(self):
        if self._loaded:
            return
        for split, data_file in self.data_files.items():
            data_file = os.path.join(self.data_dir, data_file)
            if eKonf.exists(data_file):
                df = eKonf.load_data(
                    data_file, dtype=self.DATATYPEs, verbose=self.verbose
                )
                df = self.COLUMN.init_info(df)
                df = self.COLUMN.append_split(df, split)
                if self._pipeline_ and len(self._pipeline_) > 0:
                    df = apply_pipeline(df, self._pipeline_, self._pipeline_cfg)
                self._splits[split] = df
            else:
                log.warning(f"File {data_file} not found.")
                # log.info(f"Dataset {self.name} split {split} is empty")
        self._loaded = True

    def build(self):
        pass

    def persist(self):
        pass
