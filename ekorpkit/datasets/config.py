import os
import logging
from random import sample
from glob import glob
from pathlib import Path
from tqdm.auto import tqdm
from pydantic import BaseModel, Field
from typing import Optional
from datasets import load_dataset, DatasetDict
from ekorpkit import eKonf


log = logging.getLogger(__name__)


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
    file_extention: str = Field(
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
    raw_datasets: Optional[DatasetDict] = Field(
        default=None,
        description="The raw datasets loaded from the data sources.",
    )

    class Config:
        fields = {"raw_datasets": {"exclude": True}}

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
                if self.train_file.startswith("http"):
                    extension = self.train_file.split(".")[-1]
                    if extension not in ["csv", "json", "txt", "parquet"]:
                        raise ValueError(
                            "`train_file` should be a csv, a json, a txt or a parquet file."
                        )
                    self.file_extention = extension
                else:
                    if (
                        not Path(self.train_file).is_absolute()
                        and self.data_dir is not None
                    ):
                        self.train_file = os.path.join(self.data_dir, self.train_file)

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
                # check if validation_file is url or local path
                if self.validation_file.startswith("http"):
                    self.validation_file = self.validation_file
                else:
                    if (
                        not Path(self.validation_file).is_absolute()
                        and self.data_dir is not None
                    ):
                        self.validation_file = os.path.join(
                            self.data_dir, self.validation_file
                        )

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
            if self.file_extention == "txt":
                dataset_kwargs["path"] = "text"
            else:
                dataset_kwargs["path"] = self.file_extention
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
            if self.file_extention == "txt":
                dataset_kwargs["path"] = "text"
            else:
                dataset_kwargs["path"] = self.file_extention
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
