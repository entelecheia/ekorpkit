import os
import json
import logging
import numpy as np
from scipy.special import digamma
from typing import Iterator, Optional, Union
from .base import Model
from ..trainers.unigram import UnigramTrainer
from ..utils.trie import Trie


log = logging.getLogger(__name__)


class Unigram(Model):
    """
    An implementation of the Unigram algorithm
    Args:
        vocab (:obj:`List[Tuple[str, float]]`, `optional`):
            A list of vocabulary items and their relative score [("am", -0.2442),...]
    """

    def __init__(self, vocab=None):
        super().__init__(vocab)

    def initialize_vocab(self, vocab, **kwargs):
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        self.trie: Trie = None
        self.max_piece_length = None
        if vocab:
            self.vocab = vocab
            self.token2id = {token: i for i, token in enumerate(vocab)}
            self.id2token = {v: k for k, v in self.token2id.items()}
            self.trie, self.max_piece_length = self.initialize_trie(vocab)

    def initialize_trie(self, tokens):
        trie = Trie()
        norm = sum(list(tokens.values()))
        logsum = digamma(norm)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, digamma(val) - logsum)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def generalized_forward_step(self, text, trie: Trie, nbest_size=1):
        N = len(text)
        d = [-np.inf] * (N + 1)
        p = [None] * (N + 1)
        d[0] = 0
        for i in range(1, N + 1):
            d_queue = []
            p_queue = []
            for j in range(max(i - self.max_piece_length, 0), i):
                final_token = text[j:i]
                final_value = trie.get_value(final_token)
                if final_value:
                    curr_d = d[j] + final_value
                    curr_p = len(final_token)
                    d[i] = max(d[i], curr_d)
                    d_queue.append(curr_d)
                    p_queue.append(curr_p)
            ids = np.argsort(d_queue)[-nbest_size:]
            p[i] = [p_queue[z] for z in ids]
        return p

    def generalized_backward_step(self, text, p):
        idx = len(p)
        tokenization = []
        while idx > 1:
            back_steps = np.random.choice(p[idx - 1])
            next_idx = idx - back_steps
            tok = text[next_idx - 1 : idx - 1]
            tokenization.append(tok)
            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def tokenize(self, sequence, nbest_size=1):
        """
        Tokenize a sequence
        Args:
            sequence (:obj:`str`):
                A sequence to tokenize
            nbest_size (:obj:`int`, `optional`, defaults to 1):
        Returns:
            A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
        """
        sequence = self.pre_tokenize(sequence)
        text = "".join(sequence)
        if self.trie is None:
            raise ValueError("Trainer has not yet been fit. Cannot tokenize.")
        p = self.generalized_forward_step(text, self.trie, nbest_size)
        tokenization = self.generalized_backward_step(text, p)
        return tokenization

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        trainer: UnigramTrainer = None,
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer.normalizer = self.normalizer
        trainer.pre_tokenizer = self.pre_tokenizer
        vocab = trainer.train(iterator, length=length)
        self.initialize_vocab(vocab)

    @classmethod
    def from_file(cls, vocab, **kwargs):
        """
        Instantiate a Unigram model from the given files.

        This method is roughly equivalent to doing::

           vocab, merges = Unigram.read_file(vocab_filename, merges_filename)
           bpe = Unigram(vocab, merges)

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            :class:`~tokenizers.models.Unigram`: An instance of Unigram loaded from these files
        """
        vocab = cls.read_file(vocab)
        return cls(vocab, **kwargs)

    @staticmethod
    def read_file(vocab):
        """
        Read a :obj:`vocab.json` file

        This method provides a way to read and parse the content of these files,
        returning the relevant data structures. If you want to instantiate some BPE models
        from memory, this method gives you the expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            A :obj:`Tuple` with the vocab and the merges:
                The vocabulary and merges loaded into memory
        """
        with open(vocab, "r") as f:
            vocab = json.load(f)
        return vocab

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
        if not os.path.exists(folder):
            os.makedirs(folder)
        indent = 2 if pretty else None
        json.dump(self.vocab, open(vocab_filename, "w"), indent=indent)
        return [vocab_filename]
