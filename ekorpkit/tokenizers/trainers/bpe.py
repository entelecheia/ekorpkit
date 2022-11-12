import collections
import logging
from typing import Iterator, List, Optional, Union
from tokenizers import AddedToken
from tqdm.auto import tqdm
from .base import Trainer


log = logging.getLogger(__name__)


class BpeTrainer(Trainer):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        end_of_word_suffix="▁",
        show_progress: bool = True,
        max_merges: int = None,
        verbose=False,
    ):
        self.vocab = {}
        self.merges = {}

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = initial_alphabet
        self.end_of_word_suffix = end_of_word_suffix
        self.show_progress = show_progress
        self.max_merges = max_merges
        self.verbose = verbose

        super().__init__()

    def get_word_freqs(self, texts):
        word_freqs = collections.defaultdict(int)
        for text_batch in texts:
            if isinstance(text_batch, str):
                text_batch = [text_batch]
            for text in text_batch:
                words = self.pre_tokenize(text)
                for word in words:
                    word_freqs[word] += 1
        # Remove words that are too rare
        word_freqs = {
            word: freq
            for word, freq in word_freqs.items()
            if freq >= self.min_frequency
        }
        return word_freqs

    def initialize_vocab(self, texts):
        word_freqs = self.get_word_freqs(texts)

        alphabet = self.initial_alphabet
        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        vocab = self.special_tokens + alphabet.copy() + [self.end_of_word_suffix]
        splits = {
            word: [c for c in word] + [self.end_of_word_suffix]
            for word in word_freqs.keys()
        }

        return vocab, splits, word_freqs

    def compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def fit(self, texts):
        vocab, splits, word_freqs = self.initialize_vocab(texts)
        vocab_size = self.vocab_size

        merges = {}
        num_merges = 0
        if self.show_progress:
            iterator = tqdm(range(vocab_size - len(vocab)))
        while len(vocab) < vocab_size:
            pair_freqs = self.compute_pair_freqs(splits, word_freqs)
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self.merge_pair(*best_pair, splits, word_freqs)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
            num_merges += 1
            if self.show_progress:
                iterator.update(1)
            if self.verbose:
                if (
                    isinstance(self.verbose, int) and num_merges % self.verbose == 0
                ) or (isinstance(self.verbose, list) and num_merges in self.verbose):
                    print(f"--- Round {num_merges}. Vocab size: {len(vocab)} ---")
                    print(f"Merge {num_merges}: {best_pair}")
                    print(f"Number of tokens: {len(vocab)}")
            if self.max_merges is not None and num_merges >= self.max_merges:
                break
        iterator.close()
        if self.verbose:
            print(f"--- Round {num_merges}. Vocab size: {len(vocab)} ---")
            print(f"Merge {num_merges}: {best_pair}")
            print(f"Number of tokens: {len(vocab)}")
            print(f"List of vocab: {vocab}")
        return vocab, merges

    def train(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        return self.fit(iterator)
