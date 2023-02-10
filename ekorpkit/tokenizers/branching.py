import os
import logging
from typing import Iterator, List, Optional, Union
from tokenizers import (
    Regex,
    decoders,
    normalizers,
    pre_tokenizers,
)
from .models.branching import (
    BranchingEntropy,
    BranchingEntropyTrainer,
)
from .base import BaseTokenizer
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class BranchingEntropyTokenizer(BaseTokenizer):
    """SentencePiece Unigram Tokenizer

    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        vocab: Optional[str] = None,
        branching_threshold=0.0,
        cohesion_threshold=0.0,
        whitespace_token="▁",
        whitespace_token_as_prefix=True,
        add_prefix_space: bool = True,
        lowercase: bool = True,
        verbose=False,
        **kwargs,
    ):
        if vocab is not None:
            # Let Unigram(..) fail if only one of them is None
            tokenizer = BranchingEntropy(
                vocab=vocab,
                branching_threshold=branching_threshold,
                cohesion_threshold=cohesion_threshold,
                whitespace_token=whitespace_token,
                whitespace_token_as_prefix=whitespace_token_as_prefix,
                verbose=verbose,
            )
        else:
            tokenizer = BranchingEntropy()

        normalizers_ = [
            normalizers.Nmt(),
            normalizers.NFKC(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.StripAccents(),
        ]
        if lowercase:
            normalizers_ += [normalizers.Lowercase()]

        tokenizer.normalizer = normalizers.Sequence(normalizers_)
        pre_tokenizers_ = [
            pre_tokenizers.Metaspace(
                replacement=whitespace_token, add_prefix_space=add_prefix_space
            ),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Digits(individual_digits=True),
        ]
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizers_)
        tokenizer.decoder = decoders.Metaspace(
            replacement=whitespace_token, add_prefix_space=add_prefix_space
        )

        parameters = {
            "model": "BranchingEntropy",
            "branching_threshold": branching_threshold,
            "whitespace_token": whitespace_token,
            "cohesion_threshold": cohesion_threshold,
            "whitespace_token_as_prefix": whitespace_token_as_prefix,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        whitespace_token="▁",
        max_sentencepiece_length=20,
        min_frequency: int = 2,
        show_progress=True,
        verbose=False,
    ):
        """
        Train the model using the given files
        """

        files = eKonf.get_filepaths(files)
        num_files = len(files)
        # create iterator for files
        iterator = (open(f, "r", encoding="utf-8") for f in files)

        self.train_from_iterator(
            iterator,
            whitespace_token=whitespace_token,
            max_sentencepiece_length=max_sentencepiece_length,
            min_frequency=min_frequency,
            show_progress=show_progress,
            verbose=verbose,
            length=num_files,
        )

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        whitespace_token="▁",
        max_sentencepiece_length=20,
        min_frequency: int = 2,
        show_progress=True,
        verbose=False,
        length: Optional[int] = None,
    ):
        """
        Train the model using the given iterator
        """

        trainer = BranchingEntropyTrainer(
            whitespace_token=whitespace_token,
            max_sentencepiece_length=max_sentencepiece_length,
            min_frequency=min_frequency,
            show_progress=show_progress,
            verbose=verbose,
        )

        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )

    @staticmethod
    def load(folder, prefix=None, **kwargs):
        """Load a model from the given folder

        Args:
            folder (str): Path to the folder containing the vocab and merges files
            prefix (:obj:`str`, `optional`):
                An optional prefix, used to prefix each file name
        """
        if prefix is not None:
            folder = os.path.join(folder, prefix)
        vocab = os.path.join(folder, "vocab.json.zst")
        config = os.path.join(folder, "config.json")

        return BranchingEntropyTokenizer.from_file(vocab, config, **kwargs)

    @staticmethod
    def from_file(vocab_filename: str, config_filename: str, **kwargs):
        vocab = BranchingEntropy.read_file(vocab_filename)
        config = BaseTokenizer.read_config(config_filename)
        config.update(kwargs)

        return BranchingEntropyTokenizer(vocab, **config)

    def find_local_entropy(self, word):
        return self._tokenizer.find_local_entropy(word)

    def show_local_entropy(self, word):
        return self._tokenizer.show_local_entropy(word)

    def plot_local_entropy(
        self,
        word,
        figsize=(12, 5),
    ):
        return self._tokenizer.plot_local_entropy(word, figsize=figsize)

    def tokenize(
        self,
        sequence,
        flatten=True,
        branching_threshold=None,
        cohesion_threshold=None,
        **kwargs,
    ):
        return self._tokenizer.tokenize(
            sequence,
            flatten=flatten,
            branching_threshold=branching_threshold,
            cohesion_threshold=cohesion_threshold,
            **kwargs,
        )

    def naive_segment(self, text):
        return self._tokenizer.naive_segment(text)
