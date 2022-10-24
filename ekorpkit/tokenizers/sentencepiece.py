import os
from typing import Iterator, List, Optional, Union
from tokenizers import (
    AddedToken,
    Regex,
    decoders,
    normalizers,
    pre_tokenizers,
)
from .models.unigram import Unigram, UnigramTrainer
from .base import BaseTokenizer


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """SentencePiece Unigram Tokenizer

    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """

    def __init__(
        self,
        vocab: Optional[str] = None,
        replacement: str = "▁",
        add_prefix_space: bool = True,
        lowercase: bool = True,
        **kwargs,
    ):
        if vocab is not None:
            # Let Unigram(..) fail if only one of them is None
            tokenizer = Unigram(vocab)
        else:
            tokenizer = Unigram()

        normalizers_ = [
            normalizers.Nmt(),
            normalizers.NFKC(),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if lowercase:
            normalizers_ += [normalizers.Lowercase()]

        tokenizer.normalizer = normalizers.Sequence(normalizers_)
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement=replacement, add_prefix_space=add_prefix_space
        )

        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens: Optional[List[Union[str, AddedToken]]] = None,
        initial_alphabet: Optional[List[str]] = None,
        shrinking_factor=0.75,
        unk_token: Optional[str] = None,
        max_piece_length=16,
        n_sub_iterations=5,
        max_rounds=5,
        delta=0.01,
        min_frequency: int = 2,
    ):
        """
        Train the model using the given files

        Args:
            files (:obj:`List[str]`):
                A list of path to the files that we should use for training
            vocab_size (:obj:`int`):
                The size of the final vocabulary, including all tokens and alphabet.
            show_progress (:obj:`bool`):
                Whether to show progress bars while training.
            special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
                A list of special tokens the model should know of.
            initial_alphabet (:obj:`List[str]`, `optional`):
                A list of characters to include in the initial alphabet, even
                if not seen in the training dataset.
                If the strings contain more than one character, only the first one
                is kept.
            unk_token (:obj:`str`, `optional`):
                The unknown token to be used by the model.
        """

        if special_tokens is None:
            special_tokens = []

        if initial_alphabet is None:
            initial_alphabet = []

        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
            initial_alphabet=initial_alphabet,
            shrinking_factor=shrinking_factor,
            unk_token=unk_token,
            max_piece_length=max_piece_length,
            n_sub_iterations=n_sub_iterations,
            max_rounds=max_rounds,
            delta=delta,
            min_frequency=min_frequency,
        )

        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
        special_tokens: Optional[List[Union[str, AddedToken]]] = None,
        initial_alphabet: Optional[List[str]] = None,
        shrinking_factor=0.75,
        unk_token: Optional[str] = None,
        max_piece_length=16,
        n_sub_iterations=5,
        max_rounds=5,
        delta=0.01,
        min_frequency: int = 2,
        length: Optional[int] = None,
    ):
        """
        Train the model using the given iterator

        Args:
            iterator (:obj:`Union[Iterator[str], Iterator[Iterator[str]]]`):
                Any iterator over strings or list of strings
            vocab_size (:obj:`int`):
                The size of the final vocabulary, including all tokens and alphabet.
            show_progress (:obj:`bool`):
                Whether to show progress bars while training.
            special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
                A list of special tokens the model should know of.
            initial_alphabet (:obj:`List[str]`, `optional`):
                A list of characters to include in the initial alphabet, even
                if not seen in the training dataset.
                If the strings contain more than one character, only the first one
                is kept.
            unk_token (:obj:`str`, `optional`):
                The unknown token to be used by the model.
            length (:obj:`int`, `optional`):
                The total number of sequences in the iterator. This is used to
                provide meaningful progress tracking
        """

        if special_tokens is None:
            special_tokens = []

        if initial_alphabet is None:
            initial_alphabet = []

        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=show_progress,
            initial_alphabet=initial_alphabet,
            shrinking_factor=shrinking_factor,
            unk_token=unk_token,
            max_piece_length=max_piece_length,
            n_sub_iterations=n_sub_iterations,
            max_rounds=max_rounds,
            delta=delta,
            min_frequency=min_frequency,
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
        vocab = os.path.join(folder, "vocab.json")
        config = os.path.join(folder, "config.json")

        return SentencePieceUnigramTokenizer.from_file(vocab, config, **kwargs)

    @staticmethod
    def from_file(vocab_filename: str, config_filename: str, **kwargs):
        vocab = Unigram.read_file(vocab_filename)
        config = BaseTokenizer.read_config(config_filename)
        config.update(kwargs)
        return SentencePieceUnigramTokenizer(vocab, **config)
