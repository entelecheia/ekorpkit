import re
import math
import logging
import matplotlib.pyplot as plt
import collections
from scipy.special import digamma
from .base import Trie


log = logging.getLogger(__name__)


def entropy(trie, word):
    leafs = trie.get_leafs(word)
    val = trie.get_value(word)
    logsum = digamma(sum(leafs) + val)
    entropy = 0
    for freq in leafs:
        logprob = digamma(freq) - logsum
        entropy += math.exp(logprob) * logprob
    return -1 * entropy


class WordSegmenter:
    def __init__(
        self, lowercase=True, whitespace_token="▁", max_sentencepiece_length=20
    ):
        self.lowercase = lowercase
        self.whitespace_token = whitespace_token
        self.max_sentencepiece_length = max_sentencepiece_length

        self.word_freqs = None
        self.subwords = None
        self.fwd_trie = None
        self.bwd_trie = None
        self.max_subword_len = None

    def normalize_word(self, word):
        # replace all non-alphanumeric characters at the end of the word with a space
        word = re.sub(r"[^a-zA-Z0-9]+$", " ", word)
        # replace all non-alphanumeric characters at the beginning of the word with a space
        word = re.sub(r"^[^a-zA-Z0-9]+", " ", word)
        return word.strip()

    def pre_tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        # remove urls
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return [
            self.normalize_word(word)
            for word in text.split()
            if len(self.normalize_word(word)) > 0
        ]

    def initialize_subwords(self, texts):
        subwords_freqs = collections.defaultdict(int)
        all_words = []

        for text in texts:
            words = self.pre_tokenize(text)
            all_words.extend(words)
            for word in words:
                word = self.whitespace_token + word + self.whitespace_token
                for i in range(len(word)):
                    for j in range(
                        i + 1, min(i + self.max_sentencepiece_length + 1, len(word) + 1)
                    ):
                        subwords_freqs[word[i:j]] += 1

        # Sort subwords by frequency
        sorted_subwords = sorted(
            subwords_freqs.items(), key=lambda x: x[1], reverse=True
        )
        print("Top 10 subwords: {}".format(sorted_subwords[:10]))

        word_freqs = collections.Counter(all_words)
        subwords = collections.Counter(subwords_freqs)
        return word_freqs, subwords

    def initialize_trie(self, tokens, direction="forward"):
        trie = Trie(direction=direction)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, val)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def fit(self, texts):
        word_freqs, subwords = self.initialize_subwords(texts)
        self.word_freqs = word_freqs
        self.subwords = subwords

        self.fwd_trie, self.max_subword_len = self.initialize_trie(
            subwords, direction="forward"
        )
        self.bwd_trie, _ = self.initialize_trie(subwords, direction="backward")

    def find_local_entropy(self, word, direction="forward"):
        entropies = []
        if direction == "forward":
            _word = self.whitespace_token + word
            _trie = self.fwd_trie
        else:
            _word = word + self.whitespace_token
            _trie = self.bwd_trie
        for i in range(2, len(word) + 2):
            if direction == "forward":
                subword = _word[:i]
            else:
                subword = _word[-i:]
            entropies.append(entropy(_trie, subword))
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
        results = self.find_local_entropy(word, direction=direction)
        chars, entropies, diffs = zip(*results)
        plt.figure(figsize=figsize)
        plt.plot(entropies, label="entropy", marker="o")
        plt.legend(loc="upper left")

        # plot diffs on the right y-axis
        plt.twinx()
        plt.plot(diffs, label="diffs", color="red", linestyle="--", marker="o")
        plt.xticks(range(len(chars)), chars)
        plt.legend(loc="upper right")
        plt.show()

    def segment_word(self, word, direction="forward"):
        # if there is a spike in entropy, then we should segment
        # Here the spike means that there is a sudden increase in entropy followed by a decrease.
        # We can use the difference in entropy to detect the spike.

        # get the local entropy and the difference in entropy
        results = self.find_local_entropy(word)
        _, _, diffs = zip(*results)

        # get the spikes
        spikes = []
        for i in range(1, len(diffs) - 1):
            if diffs[i] > 0 and diffs[i + 1] < 0:
                spikes.append(i)

        # segment the word
        segments = []
        start = 0
        for spike in spikes:
            segments.append(word[start : spike + 1])
            start = spike + 1
        if start < len(word):
            segments.append(word[start:])

        return tuple(segments)

    def segment_text(self, text, direction="forward", flatten=True):
        segments = []
        words = self.pre_tokenize(text)
        for word in words:
            segments.append(self.segment_word(word, direction=direction))
        if flatten:
            segments = [seg for word in segments for seg in word]
        return segments

    def segment_texts(self, texts, direction="forward"):
        return [self.segment_text(text, direction=direction) for text in texts]
