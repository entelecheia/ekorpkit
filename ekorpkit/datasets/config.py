import logging
from pydantic import BaseModel, Field
from typing import Optional
from datasets import load_dataset, DatasetDict
from ekorpkit import eKonf


logger = logging.getLogger(__name__)


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
    train_file: Optional[str] = Field(
        default=None,
        description="The input training data file (or folder).",
    )
    file_extention: str = Field(
        default="txt",
        description="The input file extension",
    )
    validation_file: Optional[str] = Field(
        default=None,
        description="An optional input evaluation data file to evaluate the perplexity on (or folder).",
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
        default=None,
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

    def __init__(self, **kw):
        super().__init__(**kw)
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                if eKonf.is_file(self.train_file):
                    extension = self.train_file.split(".")[-1]
                    if extension not in ["csv", "json", "txt", "parquet"]:
                        raise ValueError(
                            "`train_file` should be a csv, a json, a txt or a parquet file."
                        )
                    self.file_extention = extension
                elif eKonf.is_dir(self.train_file):
                    files = eKonf.get_filepaths(
                        f"*.{self.file_extention}", self.train_file
                    )
                    if len(files) == 0:
                        raise ValueError(
                            f"Could not find any {self.file_extention} file in {self.train_file}"
                        )
                else:
                    raise ValueError(
                        "`train_file` doesn't exist. Please check the path."
                    )

            if self.validation_file is not None:
                if eKonf.is_file(self.validation_file):
                    extension = self.validation_file.split(".")[-1]
                    if extension not in ["csv", "json", "txt", "parquet"]:
                        raise ValueError(
                            "`validation_file` should be a csv, a json, a txt or a parquet file."
                        )
                    if extension != self.file_extention:
                        raise ValueError(
                            "`validation_file` should have the same extension as `train_file`."
                        )
                elif eKonf.is_dir(self.validation_file):
                    files = eKonf.get_filepaths(
                        f"*.{self.file_extention}", self.validation_file
                    )
                    if len(files) == 0:
                        raise ValueError(
                            f"Could not find any {self.file_extention} file in {self.validation_file}"
                        )

    @property
    def dataset_kwargs(self):
        return dict(
            path=self.dataset_name,
            cache_dir=self.cache_dir,
            use_auth_token=True if self.use_auth_token else None,
        )

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
            if self.file_extention == "txt":
                dataset_kwargs["path"] = "text"
            else:
                dataset_kwargs["path"] = self.file_extention
            if eKonf.is_file(self.train_file):
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
            if self.file_extention == "txt":
                dataset_kwargs["path"] = "text"
            else:
                dataset_kwargs["path"] = self.file_extention
            if eKonf.is_file(self.validation_file):
                dataset_kwargs["data_files"] = self.validation_file
            else:
                dataset_kwargs["data_dir"] = self.validation_file
            dataset_kwargs["split"] = "validation"
        elif self.validation_split_percentage:
            dataset_kwargs = self.train_data_source
            dataset_kwargs["split"] = f"train[:{self.validation_split_percentage}%]"
        return dataset_kwargs

    def load_datasets(self):
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at
        # https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the datasets Hub
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column.
        # You can easily tweak this behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if self.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(**self.data_source)
            if (
                self.validation_split_percentage
                and "validation" not in raw_datasets.keys()
            ):
                raw_datasets["validation"] = load_dataset(**self.validation_data_source)
                raw_datasets["train"] = load_dataset(**self.train_data_source)
        else:
            raw_datasets = DatasetDict()
            if self.validation_split_percentage:
                raw_datasets["validation"] = load_dataset(**self.validation_data_source)
            raw_datasets["train"] = load_dataset(**self.train_data_source)

        column_names = raw_datasets["train"].column_names
        self.text_column_name = "text" if "text" in column_names else column_names[0]

        if self.shuffle:
            logger.info("Shuffling the dataset with seed %s", self.seed)
            raw_datasets = raw_datasets.shuffle(seed=self.seed)

        # See more about loading any type of standard or custom dataset
        # (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.
        return raw_datasets
