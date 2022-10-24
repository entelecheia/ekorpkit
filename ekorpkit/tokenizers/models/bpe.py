import json
import os
import logging
from typing import Iterator, List, Optional, Union
from .base import Model
from ..trainers.bpe import BpeTrainer


log = logging.getLogger(__name__)


class BPE(Model):
    """
    An implementation of the BPE (Byte-Pair Encoding) algorithm

    Args:
        vocab (:obj:`Dict[str, int]`, `optional`):
            A dictionnary of string keys and their ids :obj:`{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`, `optional`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`) :obj:`[("a", "b"),...]`

        cache_capacity (:obj:`int`, `optional`):
            The number of words that the BPE cache can contain. The cache allows
            to speed-up the process by keeping the result of the merge operations
            for a number of words.

        dropout (:obj:`float`, `optional`):
            A float between 0 and 1 that represents the BPE dropout to use.

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

        continuing_subword_prefix (:obj:`str`, `optional`):
            The prefix to attach to subword units that don't represent a beginning of word.

        end_of_word_suffix (:obj:`str`, `optional`):
            The suffix to attach to subword units that represent an end of word.

        fuse_unk (:obj:`bool`, `optional`):
            Whether to fuse any subsequent unknown tokens into a single one
    """

    def __init__(
        self,
        vocab=None,
        merges=None,
        cache_capacity=None,
        dropout=None,
        unk_token="</u>",
        continuing_subword_prefix=None,
        end_of_word_suffix="</w>",
        fuse_unk=None,
    ):
        super().__init__(vocab=vocab, merges=merges)
        self.cache_capacity = cache_capacity
        self.dropout = dropout
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.fuse_unk = fuse_unk

    def initialize_vocab(self, vocab, merges, **kwargs):
        self.vocab = {}
        self.merges = {}
        self.token2id = {}
        self.id2token = {}
        if vocab:
            self.vocab = vocab
            self.merges = merges
            self.token2id = {token: i for i, token in enumerate(vocab)}
            self.id2token = {v: k for k, v in self.token2id.items()}

    @classmethod
    def from_file(cls, vocab, merges, **kwargs):
        """
        Instantiate a BPE model from the given files.

        This method is roughly equivalent to doing::

           vocab, merges = BPE.read_file(vocab_filename, merges_filename)
           bpe = BPE(vocab, merges)

        If you don't need to keep the :obj:`vocab, merges` values lying around,
        this method is more optimized than manually calling
        :meth:`~tokenizers.models.BPE.read_file` to initialize a :class:`~tokenizers.models.BPE`

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

            merges (:obj:`str`):
                The path to a :obj:`merges.txt` file

        Returns:
            :class:`~tokenizers.models.BPE`: An instance of BPE loaded from these files
        """
        vocab, merges = cls.read_file(vocab, merges)
        return cls(vocab, merges, **kwargs)

    @staticmethod
    def read_file(vocab, merges):
        """
        Read a :obj:`vocab.json` and a :obj:`merges.txt` files

        This method provides a way to read and parse the content of these files,
        returning the relevant data structures. If you want to instantiate some BPE models
        from memory, this method gives you the expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

            merges (:obj:`str`):
                The path to a :obj:`merges.txt` file

        Returns:
            A :obj:`Tuple` with the vocab and the merges:
                The vocabulary and merges loaded into memory
        """
        with open(vocab, "r") as f:
            vocab = json.load(f)
        with open(merges, "r") as f:
            merges = [tuple(line.rstrip().split()) for line in f]
            merges = {merge: merge[0] + merge[1] for merge in merges}
        return vocab, merges

    def save(self, folder, prefix=None, pretty: bool = False):
        """
        Save the current model

        Save the current model in the given folder, using the given prefix for the various
        files that will get created.
        Any file with the same name that already exists in this folder will be overwritten.

        Args:
            folder (:obj:`str`):
                The path to the target folder in which to save the various files

            prefix (:obj:`str`, `optional`):
                An optional prefix, used to prefix each file name

        Returns:
            :obj:`List[str]`: The list of saved files
        """
        if prefix is not None:
            folder = os.path.join(folder, prefix)
        vocab_filename = os.path.join(folder, "vocab.json")
        merges_filename = os.path.join(folder, "merges.txt")
        if not os.path.exists(folder):
            os.makedirs(folder)
        indent = 2 if pretty else None
        json.dump(self.vocab, open(vocab_filename, "w"), indent=indent)
        with open(merges_filename, "w") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}" + os.linesep)
        print(f"Model files saved in {folder}")
        return [vocab_filename, merges_filename]

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        if ids is None:
            raise ValueError("None input is not valid. Should be a list of integers.")

        tokens = [self.id_to_token(id) for id in ids]
        # if skip_special_tokens:
        #     tokens = [token for token in tokens if token not in self.special_tokens]

        return "".join(tokens).replace(self.end_of_word_suffix, " ").strip()

    def tokenize(self, sequence):
        """
        Tokenize a sequence

        Args:
            sequence (:obj:`str`):
                A sequence to tokenize

        Returns:
            A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
        """
        pre_tokenized_text = self.pre_tokenize(sequence)
        splits = [
            [char for char in word] + [self.end_of_word_suffix]
            for word in pre_tokenized_text
        ]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        trainer: BpeTrainer = None,
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer.normalizer = self.normalizer
        trainer.pre_tokenizer = self.pre_tokenizer
        vocab, merges = trainer.train(iterator, length=length)
        self.initialize_vocab(vocab, merges)
