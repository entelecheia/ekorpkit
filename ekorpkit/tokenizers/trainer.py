import logging
import sentencepiece as spm
import random
import pandas as pd
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers import (
    Regex,
    normalizers,
    pre_tokenizers,
)
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
        self._tokenizer = None
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
    def use_sample(self):
        return self.training_config.use_sample

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
                # pre_tokenizers.Punctuation(),
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

    @property
    def input_files(self):
        if self.dataset_type == DatasetType.TEXT:
            if self.use_sample:
                if not self.sample_filepath.exists():
                    self.export_sample()
                input_files = [str(self.sample_filepath)]
            else:
                if self.num_exported_files == 0:
                    self.export_sentence_chunks()
                input_files = eKonf.get_filepaths("*.txt", self.export_dir)
            return input_files
        else:
            return []

    def train_huggingface(self):
        """
        Takes the files and trains the tokenizer.
        """
        model_path = self.model_path

        tokenizer, trainer = self.prepare_trainer()
        if self.dataset_type == DatasetType.TEXT:
            tokenizer.train(
                files=self.input_files,
                trainer=trainer,
            )
        else:
            split = self.training_config.dataset_split
            tokenizer.train_from_iterator(
                self.dataset_config.batch_iterator(split=split),
                lengths=len(self.raw_datasets[split]),
                trainer=trainer,
            )
        tokenizer.save(model_path)
        log.info(f"saved model to {model_path}")

    def train_spm(self):
        input = self.input_files
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
                **model_args,
            )
        log.info(f"Saved SentencePiece model to {output_dir}")

    @property
    def dataset_config(self):
        if self._dataset_config is None:
            if self.config.dataset.data_dir is None:
                self.config.dataset.data_dir = self.data_dir
            cfg = DatasetConfig(**self.config.dataset)
            cfg.cache_dir = str(self.cache_dir)
            if cfg.seed is None:
                cfg.seed = self.seed
            self._dataset_config = cfg
        return self._dataset_config

    @property
    def raw_datasets(self):
        return self.dataset_config.raw_datasets

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
        if self.num_exported_files == 0:
            self.export_sentence_chunks()

        overwrite = self.export_config.overwrite_sample
        sample_frac = self.export_config.sample_frac

        self.dataset_config.export_sample(
            input_dir=self.export_dir,
            output_filepath=self.sample_filepath,
            sample_frac=sample_frac,
            overwrite=overwrite,
        )

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

    @property
    def num_exported_files(self):
        return len(list(self.export_dir.glob("*.txt")))

    def export_sentence_chunks(
        self,
        sent_tokenize=None,
    ):
        """
        Make a sentence per line files, chuncsize sentences per file
        """
        output_dir = self.export_dir
        overwrite = self.export_config.overwrite_chunks
        chunk_size = self.export_config.chunk_size
        filename_fmt = self.export_config.filename_fmt

        self.dataset_config.export_sentence_chunks(
            output_dir=output_dir,
            overwrite=overwrite,
            chunk_size=chunk_size,
            filename_fmt=filename_fmt,
            sent_tokenize=sent_tokenize,
        )

    def load_tokenizer(self):
        if self.trainer_type == TrainerType.SPM:
            return spm.SentencePieceProcessor(model_file=self.model_path)
        else:
            return Tokenizer.from_file(self.model_path)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    def tokenize(self, text):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            return self.tokenizer.encode(text, out_type=str)
        return self.tokenizer.encode(text).tokens


def compare_tokens(tokenizers, sentences):
    def tokenize(tokenizer, text):
        """
        Tokenizes the text using the tokenizer.
        """
        if isinstance(tokenizer, spm.SentencePieceProcessor):
            return tokenizer.encode(text, out_type=str)
        return tokenizer.encode(text).tokens

    tokens = {}
    text = random.choice(sentences).strip()
    print(f"Text: {text}")
    # tokenize the texts with the tokenizers
    for name, tokenizer in tokenizers.items():
        tokens[name] = tokenize(tokenizer, text)

    max_len = max(len(tokens[name]) for name in tokenizers.keys())
    diffs = {name: max_len - len(tokens[name]) for name in tokenizers.keys()}

    padded_tokens = {
        name: tokens[name] + [""] * diffs[name] for name in tokenizers.keys()
    }

    df = pd.DataFrame(padded_tokens)
    return df
