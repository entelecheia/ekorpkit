from pathlib import Path
from pprint import pprint
from ekorpkit import eKonf
from wasabi import msg
from ekorpkit.pipelines.pipe import apply_pipeline
from ekorpkit.io.file import load_dataframe


class Dataset:
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        self.name = self.args.name
        self.data_dir = Path(self.args.data_dir)
        self.info_file = self.data_dir / f"info-{self.name}.yaml"
        self.info = eKonf.load(self.info_file) if self.info_file.is_file() else {}
        self.verbose = self.args.get("verbose", False)
        self.autoload = self.args.get("autoload", False)

        if self.info:
            self.args = eKonf.merge(self.args, self.info)

        if self.verbose:
            msg.info(f"Intantiating a dataset {self.name} with a config:")
            pprint(eKonf.to_dict(self.args))

        self.filetype = self.args.filetype
        self.data_files = self.args.data_files
        if self.data_files is None:
            raise ValueError("Column info can't be None")

        self.description = self.args.get("description", "")
        self.license = self.args.get("license", "")
        self.column_info = eKonf.to_dict(self.args.column_info)
        self.split_info = eKonf.to_dict(self.args.splits)
        if self.column_info is None:
            raise ValueError("Column info can't be None")
        self.splits = {}

        self._id_key = "id"
        self._dtype = dict(self.column_info.get("data", None))

        self.pipeline_args = self.args.get("pipeline", {})
        self.transform_pipeline = self.pipeline_args.get("_transform_", [])
        self.process_pipeline = self.pipeline_args.get("_pipeline_", [])
        if self.transform_pipeline is None:
            self.transform_pipeline = []
        if self.process_pipeline is None:
            self.process_pipeline = []

        if self.autoload:
            self.load()

    def load(self):
        for split, data_file in self.data_files.items():
            data_file = self.data_dir / data_file
            df = load_dataframe(data_file, dtype=self._dtype)
            if self.process_pipeline and len(self.process_pipeline) > 0:
                df = apply_pipeline(df, self.process_pipeline, self.pipeline_args)
            self.splits[split] = df
