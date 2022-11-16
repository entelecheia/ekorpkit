import logging
import collections
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm
from .base import Trainer
from ..utils.trie import Trie


log = logging.getLogger(__name__)


class BranchingEntropyTrainer(Trainer):
    def __init__(
        self,
        whitespace_token="▁",
        max_sentencepiece_length=20,
        min_frequency: int = 2,
        show_progress=True,
        verbose=False,
    ):
        self.whitespace_token = whitespace_token
        self.max_sentencepiece_length = max_sentencepiece_length
        self.min_frequency = min_frequency
        self.show_progress = show_progress
        self.verbose = verbose

        self.fwd_trie = None
        self.bwd_trie = None

        super().__init__()

    def get_word_freqs(self, words):
        word_freqs = collections.Counter(words)
        if self.verbose:
            print("Total words:", len(word_freqs))
            print("Top 10 words: {}".format(word_freqs.most_common(10)))
        # word_freqs = {
        #     word
        #     if word[0] == self.whitespace_token
        #     else self.whitespace_token + word: freq
        #     for word, freq in word_freqs.items()
        #     if word != self.whitespace_token
        # }
        if self.verbose:
            print("Total words after filtering:", len(word_freqs))
        # return as dict
        return dict(word_freqs)

    def initialize_subwords(self, word_freqs):
        subwords_freqs = collections.defaultdict(int)

        if self.show_progress:
            iterator = tqdm(word_freqs.items())
        else:
            iterator = word_freqs.items()
        for word, freq in iterator:
            if word == self.whitespace_token:
                continue
            if word[0] != self.whitespace_token:
                word = self.whitespace_token + word
            word += self.whitespace_token
            for i in range(len(word)):
                for j in range(
                    i + 1, min(i + self.max_sentencepiece_length, len(word)) + 1
                ):
                    subwords_freqs[word[i:j]] += freq

        # Remove words that are too rare
        subwords_freqs = {
            subword: freq
            for subword, freq in subwords_freqs.items()
            if freq >= self.min_frequency
        }
        # Sort subwords by frequency
        sorted_subwords = sorted(
            subwords_freqs.items(), key=lambda x: x[1], reverse=True
        )
        if self.verbose:
            print("Total subwords:", len(sorted_subwords))
            print("Top 10 subwords: {}".format(sorted_subwords[:10]))

        subwords = collections.Counter(subwords_freqs)
        return subwords

    def initialize_trie(self, tokens, direction="forward"):
        trie = Trie(direction=direction)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, val)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def fit(self, texts, length=None):
        words = []
        iterator = tqdm(texts, total=length) if self.show_progress else texts
        for text_batch in iterator:
            if isinstance(text_batch, str):
                text_batch = [text_batch]
            batch_iterator = tqdm(text_batch) if self.show_progress else text_batch
            for text in batch_iterator:
                words += self.pre_tokenize(text)
        word_freqs = self.get_word_freqs(words)
        vocab = self.initialize_subwords(word_freqs)

        # self.fwd_trie, _ = self.initialize_trie(vocab, direction="forward")
        # self.bwd_trie, _ = self.initialize_trie(vocab, direction="backward")
        return vocab

    def train(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        return self.fit(iterator, length)

    # def normalize_word(self, word):
    #     # replace all non-alphanumeric characters at the end of the word with a space
    #     word = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9]+$", " ", word)
    #     # replace all non-alphanumeric characters at the beginning of the word with a space
    #     word = re.sub(r"^[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9]+", " ", word)
    #     return word.strip()
