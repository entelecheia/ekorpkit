import logging
import random
import pandas as pd
import sentencepiece as spm
from pathlib import Path
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers import (
    Regex,
    normalizers,
    pre_tokenizers,
)
from .config import ModelType, TrainerType, DatasetType
from ekorpkit.config import BaseBatchModel
from ekorpkit.datasets.config import DatasetConfig
from ekorpkit import eKonf
from ekorpkit.utils.func import change_directory


log = logging.getLogger(__name__)


class TokenizerTrainer(BaseBatchModel):
    dataset: DatasetConfig = None
    _export_: DictConfig = None
    _train_: DictConfig = None
    __tokenizer_obj__ = None

    class Config:
        underscore_attrs_are_private = False

    def __init__(self, config_group: str = "tokenizer=trainer", **args):
        super().__init__(config_group, **args)
        self._init_dataset()
        self.autorun()

    def _init_dataset(self):
        if self.dataset.data_dir is None:
            self.dataset.data_dir = str(self.root_dir)
        if self.dataset.seed is None:
            self.dataset.seed = self.seed

    def train(self):
        """
        Trains the tokenizer.
        """
        if self.trainer_type == TrainerType.HF:
            self.train_huggingface()
        else:
            self.train_spm()

    @property
    def model_name(self):
        model_name = self.model.get("model_name")
        if model_name is None:
            model_name = "{}_{}_{}_vocab_{}".format(
                self.name, self.model_type, self.trainer_type, self.vocab_size
            )
        return model_name

    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def trainer_type(self) -> TrainerType:
        return self._train_.trainer_type

    @property
    def model_type(self) -> ModelType:
        return self.model.model_type

    @property
    def dataset_type(self) -> DatasetType:
        return self._train_.dataset_type

    @property
    def use_sample(self):
        return self._train_.use_sample

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
        model_dir = Path(self.model.model_dir or "tokenizers")
        if not model_dir.is_absolute():
            model_dir = self.root_dir / model_dir / self.name
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        return model_dir

    def prepare_trainer(self):
        """
        Prepares the tokenizer and trainer with unknown & special tokens.
        """
        model_type = self.model_type
        unk_token = self.model.unk_token
        special_tokens = list(self.model.special_tokens)
        vocab_size = self.vocab_size
        lowercase = self.model.lowercase
        whitespace_token = self.model.whitespace_token
        add_prefix_space = self.model.add_prefix_space

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
                pre_tokenizers.Digits(individual_digits=False),
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
            split = self._train_.dataset_split
            tokenizer.train_from_iterator(
                self.dataset.batch_iterator(split=split),
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
            k: v for k, v in self.model.items() if k not in ["model_name", "model_dir"]
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
    def raw_datasets(self):
        return self.dataset.datasets

    @property
    def sample_filepath(self):
        fp = self.batch_dir / self._export_.sample_name
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

        overwrite = self._export_.overwrite_sample
        sample_frac = self._export_.sample_frac

        self.dataset.export_sample(
            input_dir=self.export_dir,
            output_filepath=self.sample_filepath,
            sample_frac=sample_frac,
            overwrite=overwrite,
        )

    @property
    def export_dir(self):
        dir_ = self.batch_dir / self._export_.export_name
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
        overwrite = self._export_.overwrite_chunks
        chunk_size = self._export_.chunk_size
        filename_fmt = self._export_.filename_fmt

        self.dataset.export_sentence_chunks(
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
    def tokenizer_obj(self):
        if self.__tokenizer_obj__ is None:
            self.__tokenizer_obj__ = self.load_tokenizer()
        return self.__tokenizer_obj__

    def tokenize(self, text):
        if isinstance(self.tokenizer_obj, spm.SentencePieceProcessor):
            return self.tokenizer_obj.encode(text, out_type=str)
        return self.tokenizer_obj.encode(text).tokens


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
