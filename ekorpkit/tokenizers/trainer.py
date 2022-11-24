import os
import logging
import sentencepiece as spm
from random import sample
from pathlib import Path
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers import (
    Regex,
    normalizers,
    pre_tokenizers,
)
from .trainers.spm import train_spm
from .trainers.hf import train_hf_tokenizer
from .config import ModelType, TrainerType, DatasetType
from ekorpkit.batch import BaseConfig
from ekorpkit.datasets.config import DatasetConfig
from ekorpkit import eKonf
from ekorpkit.utils.func import change_directory


log = logging.getLogger(__name__)


class TokenizerTrainer(BaseConfig):
    def __init__(self, root_dir=None, **args):
        self._dataset_config = None
        self._training_config = None
        super().__init__(root_dir=root_dir, **args)
        self.autorun()

    def train(self):
        """
        Trains the tokenizer.
        """
        if self.trainer_type == TrainerType.HF:
            self.train_huggingface()
        else:
            self.train_spm()

    @property
    def training_config(self):
        if self._training_config is None:
            self._training_config = self.config.training
        return self._training_config

    @property
    def model_name(self):
        model_name = self.model_config.get("model_name")
        if model_name is None:
            model_name = "{}_{}_{}_vocab_{}".format(
                self.name, self.model_type, self.trainer_type, self.vocab_size
            )
        return model_name

    @property
    def vocab_size(self):
        return self.model_config.vocab_size

    @property
    def trainer_type(self) -> TrainerType:
        return self.training_config.trainer_type

    @property
    def model_type(self) -> ModelType:
        return self.model_config.model_type

    @property
    def dataset_type(self) -> DatasetType:
        return self.training_config.dataset_type

    @property
    def model_filename(self):
        if self.trainer_type == TrainerType.SPM:
            model_name = f"{self.model_name}.model"
        elif self.trainer_type == TrainerType.HF:
            model_name = f"{self.model_name}.json"
        return model_name

    @property
    def model_path(self):
        return str(self.model_dir / self.model_filename)

    @property
    def model_dir(self):
        model_dir = Path(self.model_config.model_dir or "tokenizers")
        if not model_dir.is_absolute():
            model_dir = self.output_dir / model_dir / self.name
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        return model_dir

    def prepare_trainer(self):
        """
        Prepares the tokenizer and trainer with unknown & special tokens.
        """
        model_type = self.model_type
        unk_token = self.model_config.unk_token
        special_tokens = list(self.model_config.special_tokens)
        vocab_size = self.vocab_size
        lowercase = self.model_config.lowercase
        whitespace_token = self.model_config.whitespace_token
        add_prefix_space = self.model_config.add_prefix_space

        if model_type == ModelType.BPE:
            tokenizer = Tokenizer(BPE(unk_token=unk_token))
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
            )
            pre_tokenizers_ = [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Punctuation(),
                pre_tokenizers.UnicodeScripts(),
            ]
        elif model_type == ModelType.UNIGRAM:
            tokenizer = Tokenizer(Unigram())
            trainer = UnigramTrainer(
                vocab_size=vocab_size,
                unk_token=unk_token,
                special_tokens=special_tokens,
            )
            pre_tokenizers_ = [
                pre_tokenizers.Metaspace(
                    replacement=whitespace_token, add_prefix_space=add_prefix_space
                ),
                pre_tokenizers.Punctuation(),
                pre_tokenizers.UnicodeScripts(),
                pre_tokenizers.Digits(individual_digits=True),
            ]
        else:
            tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
            trainer = WordLevelTrainer(
                vocab_size=vocab_size, special_tokens=special_tokens
            )
            pre_tokenizers_ = [pre_tokenizers.Whitespace()]
        normalizers_ = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Nmt(),
            normalizers.NFKC(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.StripAccents(),
        ]
        if lowercase:
            normalizers_ += [normalizers.Lowercase()]

        tokenizer.normalizer = normalizers.Sequence(normalizers_)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizers_)

        return tokenizer, trainer

    def train_huggingface(self):
        """
        Takes the files and trains the tokenizer.
        """
        model_path = self.model_path
        input_files = [str(self.sample_filepath)]

        tokenizer, trainer = self.prepare_trainer()
        tokenizer.train(
            input_files,
            trainer=trainer,
        )
        tokenizer.save(model_path)
        log.info(f"saved model to {model_path}")

    def train_spm(self):
        input = str(self.sample_filepath)
        model_name = self.model_filename
        output_dir = self.model_dir
        log.info(f"Training SentencePiece model {model_name}")
        model_args = {
            k: v
            for k, v in self.model_config.items()
            if k not in ["model_name", "model_dir"]
        }
        # change context work dir to output_dir
        # so that the model is saved to the correct location
        with change_directory(output_dir):
            spm.SentencePieceTrainer.train(
                input=input,
                model_prefix=self.model_name,
                model_type=self.model_type,
                **model_args,
            )
        log.info(f"Saved SentencePiece model to {output_dir}")

    @property
    def dataset_config(self):
        if self._dataset_config is None:
            cfg = DatasetConfig(**self.config.dataset)
            cfg.cache_dir = str(self.cache_dir)
            if cfg.seed is None:
                cfg.seed = self.seed
            self._dataset_config = cfg
        return self._dataset_config

    def prepare_datasets(self):
        self._raw_datasets = self.dataset_config.load_datasets()

    @property
    def raw_datasets(self):
        if self._raw_datasets is None:
            self.prepare_datasets()
        return self._raw_datasets

    @property
    def sample_filepath(self):
        fp = self.batch_dir / self.export_config.sample_name
        if not fp.exists():
            fp.parent.mkdir(parents=True, exist_ok=True)
        return fp

    def export_sample(self):
        """
        Use the set of files containing a sentence per line,
        sample num_files out of those and save as one text file
        """
        overwrite = self.export_config.overwrite_sample
        if self.sample_filepath.exists() and not overwrite:
            log.info("Sample file already exists, skipping")
            return

        sentence_files = list(self.export_dir.glob("*.txt"))
        sample_frac = self.export_config.sample_frac
        sample_size = int(len(sentence_files) * sample_frac)

        # sample num_files
        if sample_size <= len(sentence_files):
            sentence_files = sample(sentence_files, sample_size)
        else:
            log.info(
                f"Sample size {sample_size} is larger than number of files {len(sentence_files)}"
            )

        filenames = [os.path.basename(f) for f in sentence_files]
        log.info(f"sampled files: {filenames}")

        # read all the lines from sampled files and save to a list
        all_lines = []
        for fp in sentence_files:
            with open(fp) as f:
                lines = f.read().splitlines()
            all_lines.extend(lines)
        log.info(f"number of lines sampled: {len(all_lines):,}")

        num_sentences = 0
        with open(self.sample_filepath, "w") as f:
            for sentence in tqdm(all_lines):
                # remove newlines
                sentence = sentence.strip()
                # do not save empty items such as
                if sentence != []:
                    f.writelines(sentence + "\n")
                    num_sentences += 1
        log.info(f"Saved {num_sentences} sentences to {self.sample_filepath}")

    @property
    def export_config(self):
        cfg = self.config.export
        return cfg

    @property
    def export_dir(self):
        dir_ = self.batch_dir / self.export_config.export_name
        if not dir_.exists():
            dir_.mkdir(parents=True)
        return dir_

    def export_sentence_chunks(
        self,
        sent_tokenize=None,
    ):
        """
        Make a sentence per line files, chuncsize sentences per file
        """
        dataset = self.raw_datasets["train"]
        output_dir = self.export_dir
        overwrite = self.export_config.overwrite_chunks
        if len(list(output_dir.glob("*.txt"))) > 0 and not overwrite:
            log.info("Exported files already exist, skipping")
            return

        log.info(f"Writing sentence chunks to {output_dir}")
        chunk_size = self.export_config.chunk_size
        filename_fmt = self.export_config.filename_fmt

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


def batch_chunks(dataset, batch_size, text_column="text"):
    """Yield successive batch-sized chunks from dataset."""
    for i in tqdm(range(0, len(dataset), batch_size)):
        end_i = min(len(dataset), i + batch_size)
        yield dataset[i:end_i][text_column]


def train_tokenizer(
    model_prefix,
    input_files,
    input_dir=None,
    output_dir="tokenizers",
    vocab_size=30000,
    model_type: ModelType = ModelType.UNIGRAM,
    trainer_type: TrainerType = TrainerType.SPM,
    character_coverage=1.0,
    num_workers=1,
    train_extremely_large_corpus=False,
    project_dir=None,
    verbose=False,
    **kwargs,
):
    if model_prefix is None:
        raise ValueError("model_prefix must be specified")
    if kwargs:
        kwargs = eKonf.to_dict(kwargs)
        log.info(f"Additional kwargs: {kwargs}")

    if project_dir is not None:
        log.info(f"Using project_dir {project_dir}")
        output_dir = os.path.join(project_dir, output_dir)
        if input_dir is not None:
            input_dir = os.path.join(project_dir, input_dir)
        else:
            input_dir = project_dir

    input_files = eKonf.get_filepaths(input_files, input_dir)

    if trainer_type == TrainerType.SPM:
        model_path = train_spm(
            model_prefix=model_prefix,
            input=input_files,
            output_dir=output_dir,
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            num_threads=num_workers,
            train_extremely_large_corpus=train_extremely_large_corpus,
            **kwargs,
        )
    elif trainer_type == TrainerType.HF:
        model_path = train_hf_tokenizer(
            model_prefix=model_prefix,
            input_files=input_files,
            output_dir=output_dir,
            vocab_size=vocab_size,
            model_type=model_type,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")

    if verbose:
        print(f"saved model to {model_path}")
