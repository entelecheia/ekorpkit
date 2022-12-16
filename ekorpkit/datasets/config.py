import os
import logging
import pandas as pd
import numpy as np
from pandas import DataFrame
from random import sample
from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import DictConfig
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Union
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ekorpkit import eKonf
from ekorpkit.config import BaseConfigModel
from ekorpkit.base import _SPLITS as SPLITS
from ekorpkit.info.column import BaseInfo
from ekorpkit.info.stat import SummaryInfo
from ekorpkit.pipelines.pipe import apply_pipeline


log = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    name: str = None
    use_batcher: bool = True
    verbose: bool = False
    _pipeline_: list = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = False
        extra = "allow"

    def __init__(self, **args):
        super().__init__(**args)
        if self._pipeline is None:
            self._pipeline = []


class BaseDatasetConfig(BaseConfigModel):
    info: DictConfig = None
    column_info: DictConfig = None
    data_dir: str = None
    data_files: Dict[str, str] = None
    filetype: str = None
    collapse_ids: bool = False
    pipeline: Optional[PipelineConfig] = None
    description = "ekorpkit datasets"
    license = "Copyright of the dataset is owned by the authors."
    __info__ = None
    __column__: BaseInfo = None
    __splits__ = {}
    __summary_info__: SummaryInfo = None
    __data__ = None
    __loaded__ = False
    __le_classes__ = None
    __le__ = None

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        underscore_attrs_are_private = True
        extra = "allow"
        enum_values_are_str = True

    def __init__(self, **args):
        super().__init__(**args)

    @validator("filetype")
    def _check_filetype(cls, v):
        if v is None:
            return "parquet"
        return v.replace(".", "")

    @validator("data_dir", pre=True)
    def _check_data_dir(cls, v):
        if v is not None:
            return str(v)
        return v

    @property
    def info_file(self):
        return os.path.join(self.data_dir, f"info-{self.name}.yaml")

    def load_info(self):
        """Load the info file."""
        self.__info__ = (
            eKonf.load(self.info_file)
            if eKonf.exists(self.info_file) and not self.force.build
            else {}
        )
        if self.__info__:
            log.info(f"Loaded info file: {self.info_file}")
            self.__info__ = eKonf.to_dict(self.__info__)
        self.description = self.__info__.get("description") or self.description
        self.license = self.__info__.get("license") or self.license
        if self.verbose:
            log.info(f"Intantiating a {self.__class__.__name__} [{self.name}]")
        self.data_files = self.__info__.get("data_files") or self.data_files
        if self.data_files is None:
            self.data_files = {
                SPLITS.TRAIN.value: f"{self.name}-train.{self.filetype}",
                SPLITS.DEV.value: f"{self.name}-dev.{self.filetype}",
                SPLITS.TEST.value: f"{self.name}-test.{self.filetype}",
            }

    def load_column_info(self):
        self.__column__ = eKonf.instantiate(self.column_info)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.name}"
        return s

    def __getitem__(self, split="train"):
        if split in self.splits:
            return self.splits[split]
        else:
            return None

    def __len__(self):
        return self.num_rows

    @property
    def num_rows(self) -> int:
        """Number of rows in the corpus (same as :meth:`Corpus.__len__`)."""
        if self.data.index is not None:
            return len(self.data.index)
        return len(self.data)

    @property
    def INFO(self):
        return self.__info__

    @property
    def COLUMN(self) -> BaseInfo:
        return self.__column__

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

    @property
    def data(self):
        dfs = []
        for _, data in self.splits.items():
            if data is not None:
                dfs.append(data)
        return eKonf.concat_data(dfs)

    @property
    def splits(self):
        return self.__splits__

    @property
    def summary_info(self) -> SummaryInfo:
        return self.__summary_info__

    @property
    def classes(self):
        if self.__le_classes__ is None:
            log.info("LabelEncoder is not fitted")
            return None
        return self.__le_classes__.tolist()

    def build(self):
        data = None
        if self.pipeline._pipeline_ and len(self.pipeline._pipeline_) > 0:
            data = apply_pipeline(data, self.pipeline._pipeline_, self.pipeline)
        if data is not None:
            log.info(f"Dataset {self.name} built with {len(data)} rows")
        else:
            log.info(f"Dataset {self.name} is empty")

    def persist(self):
        if not self.__loaded__:
            log.info(f"Dataset {self.name} is not loaded")
            return
        if self.summary_info is None:
            self.summarize()
        for split, data in self.splits.items():
            if data is None:
                continue
            data_file = self.data_files[split]
            eKonf.save_data(
                data,
                data_file,
                base_dir=self.data_dir,
                verbose=self.verbose,
            )
        if self.summary_info is not None:
            self.summary_info.save(info={"column_info": self.COLUMN.INFO})

    def save_as(self, name):
        if not self.__loaded__:
            log.info(f"Dataset {self.name} is not loaded")
            return
        self.data_dir = self.data_dir.replace(self.name, name)
        self.name = name
        self._config_.data_files = None
        self.info.data_dir = self.data_dir
        self.info.name = self.name
        self.__summary_info__ = None
        self.load_info()
        self.persist()

    def load(self):
        if self.__loaded__:
            return
        for split, data_file in self.data_files.items():
            if eKonf.exists(self.data_dir, data_file):
                data = eKonf.load_data(
                    data_file,
                    self.data_dir,
                    verbose=self.verbose,
                    concatenate=True,
                )
                data = self.COLUMN.init_info(data)
                data = self.COLUMN.append_split(data, split)
                if self.collapse_ids:
                    data = self.COLUMN.combine_ids(data)
                self.splits[split] = data
                if self.verbose:
                    log.info(f"Data loaded {len(data)} rows")
                    print(data.head(3))
                    print(data.tail(3))
            else:
                log.warning(f"File {data_file} not found.")
                # log.info(f"Dataset {self.name} split {split} is empty")
        self.__loaded__ = True

    def summarize(self):
        if not self.__loaded__:
            log.info(f"Dataset {self.name} is not loaded")
            return
        summary_info = None
        if self.info:
            summary_info = eKonf.instantiate(self.info)
        if summary_info:
            summary_info.load(self.INFO)
        for split, data in self.splits.items():
            if data is None:
                continue
            data_file = self.data_files[split]
            if summary_info:
                stats = {"data_file": data_file}
                summary_info.init_stats(split_name=split, stats=stats)
                summary_info.calculate_stats(data, split)
        self.__summary_info__ = summary_info

    def fit_labelencoder(self, data):
        self.__le__ = preprocessing.LabelEncoder()
        self.__le__.fit(data)
        self.__le_classes__ = self.__le__.classes_
        log.info(f"LabelEncoder classes: {self.__le_classes__}")

    def transform_labels(self, data):
        if not self.__loaded__:
            log.info(f"Dataset {self.name} is not loaded")
            return data
        if data is None:
            log.info("Data is None")
            return data
        if self.__le__ is None:
            log.info("Label encoder is not fitted")
            self.fit_labelencoder(data)
        _data = self.__le__.transform(data)
        return _data

    def inverse_transform_labels(self, data):
        if not self.__loaded__:
            log.info(f"Dataset {self.name} is not loaded")
            return data
        if data is None:
            log.info("Data is None")
            return data
        if self.__le__ is None:
            log.info("Label encoder is not fitted")
            self.fit_labelencoder(data)
        _data = self.__le__.inverse_transform(data)
        return _data


class DatasetConfig(BaseModel):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = Field(
        default=None,
        description="The name of the dataset to use (via the datasets library).",
    )
    dataset_config_name: Optional[str] = Field(
        default=None,
        description="The configuration name of the dataset to use (via the datasets library).",
    )
    data_dir: Optional[str] = Field(
        default=None,
        description="The directory where the dataset is located.",
    )
    train_file: Optional[str] = Field(
        default=None,
        description="The input training data file (or folder).",
    )
    file_extension: str = Field(
        default="txt",
        description="The input file extension",
    )
    validation_file: Optional[str] = Field(
        default=None,
        description="An optional input evaluation data file to evaluate the perplexity on (or folder).",
    )
    download_mode: Optional[str] = Field(
        default=None,
        description="Whether to download and prepare the dataset from the hub or only download it.",
    )
    overwrite_cache: bool = Field(
        default=False,
        description="Overwrite the cached training and evaluation sets",
    )
    validation_split_percentage: Optional[int] = Field(
        default=5,
        description="The percentage of the train set used as validation set in case there's no validation split",
    )
    num_workers: Optional[int] = Field(
        default=None,
        description="The number of processes to use for the preprocessing.",
    )
    max_train_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    max_eval_samples: Optional[int] = Field(
        default=None,
        description=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    use_auth_token: bool = Field(
        default=False,
        description=(
            "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
            "with private models)."
        ),
    )
    text_column_name: Optional[str] = Field(
        default="text",
        description="The name of the column in the datasets containing the full text.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the data or not.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="A seed for the shuffle.",
    )
    max_seq_length: Optional[int] = Field(
        default=None,
        description=(
            "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        ),
    )
    raw_datasets: Optional[DatasetDict] = Field(
        default=None,
        description="The raw datasets loaded from the data sources.",
    )

    class Config:
        fields = {"raw_datasets": {"exclude": True}}

    def initialize_config(self, data_dir, seed):
        if self.data_dir is None:
            self.data_dir = str(data_dir)
        if self.seed is None:
            self.seed = seed

    def _check_data_sources(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.data_dir is not None:
                self.data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
            if self.train_file is not None:
                # check if train_file is url or local path
                self.train_file, self.file_extension = check_data_file(
                    self.train_file,
                    self.data_dir,
                    self.file_extension,
                    allowed_file_extensions=["csv", "json", "txt", "parquet"],
                )

            if self.validation_file is not None:
                # check if validation_file is url or local path
                self.validation_file, _ = check_data_file(
                    self.validation_file,
                    self.data_dir,
                    self.file_extension,
                    allowed_file_extensions=["csv", "json", "txt", "parquet"],
                )

    @property
    def dataset_kwargs(self):
        kwargs = dict(
            path=self.dataset_name,
            cache_dir=self.cache_dir,
            use_auth_token=True if self.use_auth_token else None,
        )
        if self.download_mode is not None:
            kwargs["download_mode"] = self.download_mode
        return kwargs

    @property
    def data_source(self):
        dataset_kwargs = self.dataset_kwargs
        dataset_kwargs["name"] = self.dataset_config_name
        return dataset_kwargs

    @property
    def train_data_source(self):
        if self.dataset_name is not None:
            dataset_kwargs = self.data_source
            if self.validation_split_percentage:
                dataset_kwargs["split"] = f"train[{self.validation_split_percentage}%:]"
        elif self.train_file is not None:
            dataset_kwargs = self.dataset_kwargs
            if self.file_extension == "txt":
                dataset_kwargs["path"] = "text"
            else:
                dataset_kwargs["path"] = self.file_extension
            if self.train_file.startswith("http") or eKonf.is_file(self.train_file):
                dataset_kwargs["data_files"] = self.train_file
            else:
                dataset_kwargs["data_dir"] = self.train_file
            if self.validation_file is None and self.validation_split_percentage:
                dataset_kwargs["split"] = f"train[{self.validation_split_percentage}%:]"
            else:
                dataset_kwargs["split"] = "train"
        else:
            raise ValueError("Trainer: training requires a train_file.")
        return dataset_kwargs

    @property
    def validation_data_source(self):
        dataset_kwargs = {}
        if self.dataset_name is not None:
            dataset_kwargs = self.data_source
            if self.validation_split_percentage:
                dataset_kwargs["split"] = f"train[:{self.validation_split_percentage}%]"
        elif self.validation_file is not None:
            dataset_kwargs = self.dataset_kwargs
            if self.file_extension == "txt":
                dataset_kwargs["path"] = "text"
            else:
                dataset_kwargs["path"] = self.file_extension
            if self.validation_file.startswith("http") or eKonf.is_file(
                self.validation_file
            ):
                dataset_kwargs["data_files"] = self.validation_file
            else:
                dataset_kwargs["data_dir"] = self.validation_file
            dataset_kwargs["split"] = "validation"
        elif self.validation_split_percentage:
            dataset_kwargs = self.train_data_source
            dataset_kwargs["split"] = f"train[:{self.validation_split_percentage}%]"
        return dataset_kwargs

    def load_datasets(
        self,
        dataset_name=None,
        dataset_config_name=None,
        text_column_name=None,
        train_file=None,
        validation_file=None,
    ):
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at
        # https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the datasets Hub
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column.
        # You can easily tweak this behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if dataset_name is not None:
            self.dataset_name = dataset_name
            self.dataset_config_name = dataset_config_name
            self.text_column_name = text_column_name
        elif train_file is not None:
            self.train_file = train_file
            self.validation_file = validation_file
            self.dataset_name = None
            self.dataset_config_name = None
            self.text_column_name = "text"

        self._check_data_sources()
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            logging.info(f"Loading dataset {self.dataset_name}")
            raw_datasets = load_dataset(**self.data_source)
            if (
                self.validation_split_percentage
                and "validation" not in raw_datasets.keys()
            ):
                raw_datasets["validation"] = load_dataset(**self.validation_data_source)
                raw_datasets["train"] = load_dataset(**self.train_data_source)
        else:
            raw_datasets = DatasetDict()
            if self.validation_split_percentage or self.validation_file is not None:
                logging.info(f"Loading validation dataset {self.validation_file}")
                raw_datasets["validation"] = load_dataset(**self.validation_data_source)
            logging.info(f"Loading training dataset {self.train_file}")
            raw_datasets["train"] = load_dataset(**self.train_data_source)

        column_names = raw_datasets["train"].column_names
        text_column_name = self.text_column_name or "text"
        self.text_column_name = (
            text_column_name if text_column_name in column_names else column_names[0]
        )

        if self.shuffle:
            log.info("Shuffling the dataset with seed %s", self.seed)
            raw_datasets = raw_datasets.shuffle(seed=self.seed)

        # See more about loading any type of standard or custom dataset
        # (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        self.raw_datasets = raw_datasets
        return raw_datasets

    @property
    def datasets(self) -> DatasetDict:
        if self.raw_datasets is None:
            self.raw_datasets = self.load_datasets()
        return self.raw_datasets

    def sample_dataset(self, dataset=None, sample_frac=0.1, split="train"):
        if dataset is None:
            dataset = self.datasets[split]
        return dataset.select(range(int(len(dataset) * sample_frac)))

    def batch_iterator(self, batch_size=1000, split="train", text_column_name=None):
        dataset = self.datasets[split]
        if text_column_name is None:
            text_column_name = self.text_column_name
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size][text_column_name]

    def export_sentence_chunks(
        self,
        output_dir,
        overwrite=False,
        split="train",
        chunk_size=10_000,
        filename_fmt="sent_chunk_{:04d}.txt",
        sent_tokenize=None,
    ):
        """
        Make a sentence per line files, chuncsize sentences per file
        """
        dataset = self.datasets[split]
        num_files = len(list(glob(f"{output_dir}/*.txt")))
        if num_files > 0 and not overwrite:
            log.info("Exported files already exist, skipping")
            return

        log.info(f"Writing sentence chunks to {output_dir}")

        # loop over the chunks
        num_sentences = 0
        for chunk_id, data_chunk in enumerate(batch_chunks(dataset, chunk_size)):
            # new file for each chunk
            filename = filename_fmt.format(chunk_id)
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w") as f:
                for line in data_chunk:
                    line = line.strip()
                    # tokenize into sentences
                    if sent_tokenize is None:
                        sentences = line.split("\n")
                    else:
                        sentences = sent_tokenize(line)
                    # do not save empty items such as
                    if sentences != []:
                        f.writelines(s + "\n" for s in sentences)
                        num_sentences += len(sentences)
        log.info(f"Saved {num_sentences} sentences to {output_dir}")

    def export_sample(
        self, input_dir, output_filepath, sample_frac=0.1, overwrite=False
    ):
        """
        Use the set of files containing a sentence per line,
        sample num_files out of those and save as one text file
        """
        if os.path.exists(output_filepath) and not overwrite:
            log.info("Output file already exists, skipping")
            return

        sentence_files = list(list(glob(f"{input_dir}/*.txt")))
        sample_size = int(len(sentence_files) * sample_frac)

        # sample num_files
        if sample_size <= len(sentence_files):
            sentence_files = sample(sentence_files, sample_size)
        else:
            log.info(
                f"Sample size {sample_size} is larger than number of files {len(sentence_files)}. ",
                "Using all files",
            )

        filenames = [os.path.basename(f) for f in sentence_files]
        log.info(f"Sampled files: {filenames}")

        # read all the lines from sampled files and save to a list
        all_lines = []
        for fp in sentence_files:
            with open(fp) as f:
                lines = f.read().splitlines()
            all_lines.extend(lines)
        log.info(f"Number of lines sampled: {len(all_lines):,}")

        num_sentences = 0
        with open(output_filepath, "w") as f:
            for sentence in tqdm(all_lines):
                # remove newlines
                sentence = sentence.strip()
                # do not save empty items such as
                if sentence != []:
                    f.writelines(sentence + "\n")
                    num_sentences += 1
        log.info(f"Saved {num_sentences} sentences to {output_filepath}")


def batch_chunks(dataset, batch_size, text_column="text"):
    """Yield successive batch-sized chunks from dataset."""
    for i in tqdm(range(0, len(dataset), batch_size)):
        end_i = min(len(dataset), i + batch_size)
        yield dataset[i:end_i][text_column]


def check_data_file(
    data_file,
    data_dir=None,
    file_extension=None,
    allowed_file_extensions=["csv", "parquet"],
):

    if data_file.startswith("http"):
        file_extension = data_file.split(".")[-1]
        if file_extension not in allowed_file_extensions:
            raise ValueError("`data_file` should be a csv or a parquet file.")
    else:
        if not Path(data_file).is_absolute() and data_dir is not None:
            data_file = os.path.join(data_dir, data_file)

        if eKonf.is_file(data_file):
            file_extension = data_file.split(".")[-1]
            if file_extension not in allowed_file_extensions:
                raise ValueError("`data_file` should be a csv or a parquet file.")
        elif eKonf.is_dir(data_file):
            files = eKonf.get_filepaths(f"*.{file_extension}", data_file)
            if len(files) == 0:
                raise ValueError(
                    f"Could not find any {file_extension} file in {data_file}"
                )
        else:
            raise ValueError("`data_file` doesn't exist. Please check the path.")
    return data_file, file_extension


class DataframeConfig(BaseModel):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = Field(
        default=None,
        description="The name of the dataset to use.",
    )
    data_dir: Optional[str] = Field(
        default=None,
        description="The directory where the dataset is located.",
    )
    data_files: Union[str, Dict[str, str]] = Field(
        default=None,
        description="The input data files for the splits from 'train', 'validation', 'test'.",
    )
    file_extension: str = Field(
        default="parquet",
        description="The input file extension",
    )
    overwrite_cache: bool = Field(
        default=False,
        description="Overwrite the cached training and evaluation sets",
    )
    num_workers: Optional[int] = Field(
        default=None,
        description="The number of processes to use for the preprocessing.",
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    data_columns: Optional[List[str]] = Field(
        default=None,
        description="The columns of the data to use.",
    )
    text_column_name: Optional[str] = Field(
        default="text",
        description="The name of the column in the datasets containing the full text.",
    )
    label_column_name: Optional[str] = Field(
        default="labels",
        description="The name of the column in the datasets containing the labels.",
    )
    class_column_name: Optional[str] = Field(
        default="classes",
        description="The name of the encoded class column in the datasets containing the labels.",
    )
    test_size: Optional[float] = Field(
        default=0.2,
        description="The proportion of the dataset to include in the test split.",
    )
    dev_size: Optional[float] = Field(
        default=0.2,
        description="The proportion of the dataset to include in the dev split.",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the data or not.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="A seed for the shuffle.",
    )
    encode_labels: bool = Field(
        default=False,
        description="Whether to encode the labels or not.",
    )
    __raw_datasets__ = None
    __le__ = None
    __le_classes__ = None

    class Config:
        extra = "allow"
        underscore_attrs_are_private = True

    @validator("data_dir", pre=True)
    def check_data_dir(cls, v):
        if v is not None:
            return str(v)
        return v

    @validator("data_files", pre=True)
    def check_data_files(cls, v):
        if v is not None:
            if isinstance(v, Path):
                return str(v)
            elif isinstance(v, DictConfig):
                return eKonf.to_dict(v)
        return v

    def initialize_config(self, data_dir, seed):
        if self.data_dir is None:
            self.data_dir = str(data_dir)
        if self.seed is None:
            self.seed = seed

    def _check_data_sources(self):
        if self.data_files is None:
            raise ValueError("Need a data files dictionary to load a dataset.")
        else:
            if self.data_dir is not None:
                self.data_dir = os.path.abspath(os.path.expanduser(self.data_dir))
            if isinstance(self.data_files, str):
                data_file, self.file_extension = check_data_file(
                    self.data_files, self.data_dir, self.file_extension
                )
                self.data_files = {"train": data_file}
            else:
                for split, data_file in self.data_files.items():
                    self.data_files[split], self.file_extension = check_data_file(
                        data_file, self.data_dir, self.file_extension
                    )

    def load_datasets(
        self,
        data=None,
        data_files=None,
        data_dir=None,
        test_size=None,
        dev_size=None,
        shuffle=None,
        seed=None,
        encode_labels=None,
        text_column_name=None,
        label_column_name=None,
        class_column_name=None,
    ) -> dict:

        raw_datasets = {}
        if data is not None:
            if isinstance(data, DataFrame):
                raw_datasets["train"] = data
            elif isinstance(data, dict):
                raw_datasets = data
        else:
            if data_files is not None:
                self.data_file = data_files
            if data_dir is not None:
                self.data_dir = data_dir

            self._check_data_sources()
            for split, data_file in self.data_files.items():
                raw_datasets[split] = eKonf.load_data(data_file)

        if text_column_name is not None:
            self.text_column_name = text_column_name
        column_names = raw_datasets["train"].columns
        text_column_name = self.text_column_name or "text"
        self.text_column_name = (
            text_column_name if text_column_name in column_names else column_names[0]
        )

        # encode labels
        if encode_labels is not None:
            self.encode_labels = encode_labels
        if label_column_name is not None:
            self.label_column_name = label_column_name
        if class_column_name is not None:
            self.class_column_name = class_column_name
        if self.encode_labels:
            le = preprocessing.LabelEncoder()
            label_series = raw_datasets["train"][self.label_column_name]
            le.fit(label_series)
            for split in raw_datasets:
                label_series = raw_datasets["train"][self.label_column_name]
                raw_datasets[split][self.class_column_name] = le.transform(label_series)
            self.__le__ = le
            self.__le_classes__ = le.classes_

        if test_size is not None:
            self.test_size = test_size
        if dev_size is not None:
            self.dev_size = dev_size
        if shuffle is not None:
            self.shuffle = shuffle
        if seed is not None:
            self.seed = seed

        # split the data into train, dev and test
        if (
            self.test_size is not None
            and self.test_size > 0
            and raw_datasets.get("test") is None
        ):
            log.info(
                "Splitting the dataframe into train and test with ratio %s",
                self.test_size,
            )
            train, test = train_test_split(
                raw_datasets["train"],
                test_size=self.test_size,
                random_state=self.seed,
                shuffle=self.shuffle,
            )
            raw_datasets["train"] = train
            raw_datasets["test"] = test

        if (
            self.dev_size is not None
            and self.dev_size > 0
            and raw_datasets.get("dev") is None
        ):
            log.info(
                "Splitting the dataframe into train and dev with ratio %s",
                self.dev_size,
            )
            train, dev = train_test_split(
                raw_datasets["train"],
                test_size=self.dev_size,
                random_state=self.seed,
                shuffle=self.shuffle,
            )
            raw_datasets["train"] = train
            raw_datasets["dev"] = dev

        if self.shuffle:
            log.info("Shuffling the dataframe with seed %s", self.seed)
            for split, data in raw_datasets.items():
                raw_datasets[split] = data.sample(
                    frac=1, random_state=self.seed
                ).reset_index(drop=True)

        log.info(f"Train data: {raw_datasets['train'].shape}")
        if raw_datasets.get("test") is not None:
            log.info(f"Test data: {raw_datasets['test'].shape}")
        if raw_datasets.get("dev") is not None:
            log.info(f"Dev data: {raw_datasets['dev'].shape}")

        self.__raw_datasets__ = raw_datasets
        return raw_datasets

    @property
    def datasets(self) -> dict:
        if self.__raw_datasets__ is None:
            self.load_datasets()
        return self.__raw_datasets__

    @property
    def data(self) -> DataFrame:
        dfs = []
        for _, data in self.datasets.items():
            if data is not None:
                dfs.append(data)
        return eKonf.concat_data(dfs)

    @property
    def train_data(self) -> DataFrame:
        return self.datasets["train"]

    @property
    def test_data(self) -> DataFrame:
        if "test" in self.datasets:
            return self.datasets["test"]
        return None

    @property
    def dev_data(self) -> DataFrame:
        if "dev" in self.datasets:
            return self.datasets["dev"]
        return None

    @property
    def le_classes(self):
        return self.__le_classes__

    @property
    def le(self):
        return self.__le__

    def inverse_transform(self, y):
        if self.encode_labels:
            return self.le.inverse_transform(y)
        return y

    def cross_val_datasets(self, cv=5, dev_size=0.2, random_state=1235, shuffle=True):
        if not self.datasets:
            self.load_datasets()
        if shuffle:
            data = self.data.sample(frac=1).reset_index(drop=True)
        else:
            data = self.data.reset_index(drop=True)

        splits = np.array_split(data, cv)
        for split_no, split in enumerate(splits):
            _data = pd.concat(splits[:split_no] + splits[split_no + 1 :])
            if dev_size is not None and dev_size > 0:
                _train_data, _dev_data = train_test_split(
                    _data,
                    test_size=dev_size,
                    random_state=random_state,
                    shuffle=shuffle,
                )
                log.info(
                    f"Train data: {_train_data.shape}, Test data: {_dev_data.shape}"
                )
            else:
                _train_data, _dev_data = _data, None
                log.info(f"Train data: {_train_data.shape}")
            self.datasets["train"] = _train_data
            self.datasets["dev"] = _dev_data
            self.datasets["test"] = split
            yield split_no, split
