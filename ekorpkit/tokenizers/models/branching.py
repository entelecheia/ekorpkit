import os
import ekorpkit.io.zjson as json
import logging
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Iterator, Optional, Union
from enum import Enum
from ekorpkit.visualize.base import get_plot_font
from .base import Model
from ..trainers.branching import BranchingEntropyTrainer
from ..utils.trie import Trie, entropy


log = logging.getLogger(__name__)


class BranchingDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


class BranchingEntropy(Model):
    def __init__(
        self,
        vocab=None,
        branching_threshold=0.2,
        whitespace_token="▁",
        whitespace_token_as_prefix=True,
        direction: BranchingDirection = BranchingDirection.FORWARD,
        verbose=False,
        **kwargs,
    ):
        self.branching_threshold = branching_threshold
        self.whitespace_token = whitespace_token
        self.whitespace_token_as_prefix = whitespace_token_as_prefix
        self.direction = direction
        self.verbose = verbose
        super().__init__(vocab)

    def initialize_vocab(self, vocab, **kwargs):
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        self.fwd_trie = None
        self.bwd_trie = None
        self.max_piece_length = None
        if vocab:
            self.vocab = vocab
            if (
                self.direction == BranchingDirection.FORWARD
                or self.direction == BranchingDirection.BOTH
            ):
                self.fwd_trie, self.max_piece_length = self.initialize_trie(
                    vocab, "forward"
                )
            if (
                self.direction == BranchingDirection.BACKWARD
                or self.direction == BranchingDirection.BOTH
            ):
                self.bwd_trie, self.max_piece_length = self.initialize_trie(
                    vocab, "backward"
                )

    def initialize_trie(self, tokens, direction="forward"):
        trie = Trie(direction=direction)

        maxlen = 0
        for tok, val in tqdm(tokens.items(), desc=f"Building {direction} trie"):
            trie.add(tok, val)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def get_entropy(self, word, direction="forward"):
        if direction == "forward":
            _trie = self.fwd_trie
        else:
            _trie = self.bwd_trie
        return entropy(_trie, word)

    def score(self, word, direction="forward"):
        return self.get_entropy(word, direction=direction)

    def find_local_entropy(self, word, direction="forward"):
        # get the local entropy and the difference in entropy

        entropies = []
        for i in range(1, len(word) + 1):
            if direction == "forward":
                if word.startswith(self.whitespace_token):
                    subword = word[:i]
                else:
                    subword = self.whitespace_token + word[:i]
            else:
                if word.endswith(self.whitespace_token):
                    subword = word[-i:]
                else:
                    subword = word[-i:] + self.whitespace_token
            _score = self.get_entropy(subword, direction=direction)
            entropies.append(_score)
            if self.verbose:
                print(subword, _score)
        if direction == "backward":
            entropies = entropies[::-1]

        # # get diffs
        if direction == "forward":
            diffs = [0.0] + [
                (entropies[i] - entropies[i - 1]) for i in range(1, len(entropies))
            ]
        else:
            diffs = [
                (entropies[i + 1] - entropies[i]) for i in range(0, len(entropies) - 1)
            ] + [0.0]

        return list(zip(word, entropies, diffs))

    # plot entropies
    def plot_local_entropy(self, word, direction="forward", figsize=(12, 5)):
        get_plot_font()

        results = self.find_local_entropy(word, direction=direction)
        chars, entropies, diffs = zip(*results)
        plt.figure(figsize=figsize)
        plt.plot(entropies, label="entropy", marker="o")
        plt.xticks(range(len(chars)), chars)
        plt.legend(loc="upper left")

        # plot diffs on the right y-axis
        # plt.twinx()
        # plt.plot(diffs, label="diffs", color="red", linestyle="--", marker="o")
        # plt.legend(loc="upper right")
        plt.show()

    def tokenize_word(self, word, direction="forward"):
        # if there is a spike in entropy, then we should segment
        # Here the spike means that there is a sudden increase in entropy followed by a decrease.
        # We can use the difference in entropy to detect the spike.

        # if word.startswith(self.whitespace_token):
        #     word = word[len(self.whitespace_token) :]
        # get the local entropy and the difference in entropy
        results = self.find_local_entropy(word, direction=direction)
        _, _, diffs = zip(*results)

        # get the spikes
        spikes = []
        for i in range(1, len(diffs) - 1):
            if diffs[i] > self.branching_threshold and diffs[i + 1] < 0:
                spikes.append(i)

        # segment the word
        segments = []
        start = 0
        for spike in spikes:
            segments.append(word[start : spike + 1])
            start = spike + 1
        if start < len(word):
            segments.append(word[start:])
        # if self.whitespace_token_as_prefix and len(segments) > 0:
        #     segments[0] = self.whitespace_token + segments[0]
        return tuple(segments)

    def tokenize(
        self, sequence, direction="forward", flatten=True, branching_threshold=None
    ):
        if branching_threshold is not None:
            self.branching_threshold = branching_threshold
        segments = []
        words = self.pre_tokenize(sequence)
        for word in words:
            segments.append(self.tokenize_word(word, direction=direction))
        if flatten:
            segments = [seg for word in segments for seg in word]
        return segments

    def tokenize_texts(self, texts, direction="forward"):
        return [self.tokenize(text, direction=direction) for text in texts]

    def naive_segment(self, text, direction="forward"):
        words = []

        _start, _pos = 0, 0
        # iterate over the text until we reach the end
        while _pos < len(text):
            _sentencepiece = text[_pos : _pos + self.max_piece_length]
            # print(_start, _pos, _sentencepiece)
            if len(_sentencepiece) < 1:
                break
            results = self.find_local_entropy(_sentencepiece, direction=direction)
            _, entropies, _ = zip(*results)

            if entropies[0] == 0:
                if _pos == len(text) - 1:
                    words.append(text[_start : _pos + 1])
                    _start = _pos + 1
                    break
                _pos += 1
            else:
                if _pos > _start:
                    words.append(text[_start:_pos])
                    _start = _pos
                    _pos += 1
                if len(entropies) > 1:
                    _pos += 1
                    for i in range(1, len(entropies)):
                        if entropies[i] == 0:
                            words.append(text[_start : _start + i])
                            # print(_start, words)
                            _start += i
                            _pos = _start
                            break
                        elif _pos == len(text) - 1:
                            words.append(text[_start : _pos + 1])
                        _pos += 1
                else:
                    if _pos == len(text) - 1:
                        words.append(text[_start : _pos + 1])
                    _pos += 1

        return words

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        trainer: BranchingEntropyTrainer = None,
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
        Instantiate a BranchingEntropy model from the given files.

        This method is roughly equivalent to doing::

           vocab = BranchingEntropy.read_file(vocab_filename)
           be = BranchingEntropy(vocab)

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            :class:`~tokenizers.models.BranchingEntropy`: An instance of BranchingEntropy loaded from these files
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
        vocab = json.load(vocab)
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
        vocab_filename = os.path.join(folder, "vocab.json.zst")
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(self.vocab, vocab_filename)
        return [vocab_filename]
