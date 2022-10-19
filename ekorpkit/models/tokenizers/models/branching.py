import re
import math
import logging
import matplotlib.pyplot as plt
import collections
from scipy.special import digamma
from ekorpkit.visualize.base import _configure_font
from ..utils.trie import Trie


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


class BranchingEntropy:
    def __init__(
        self,
        lowercase=True,
        whitespace_token="▁",
        max_sentencepiece_length=20,
    ):
        self.lowercase = lowercase
        self.whitespace_token = whitespace_token
        self.max_sentencepiece_length = max_sentencepiece_length

        self.fwd_trie = None
        self.bwd_trie = None

    def normalize_word(self, word):
        # replace all non-alphanumeric characters at the end of the word with a space
        word = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9]+$", " ", word)
        # replace all non-alphanumeric characters at the beginning of the word with a space
        word = re.sub(r"^[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9]+", " ", word)
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
        character_freqs = collections.defaultdict(int)
        subwords_freqs = collections.defaultdict(int)
        all_words = []

        for text in texts:
            words = self.pre_tokenize(text)
            all_words.extend(words)
            for word in words:
                # word = self.whitespace_token + word
                word = self.whitespace_token + word + self.whitespace_token
                # word = word + self.whitespace_token
                for i in range(len(word)):
                    if word[i] != self.whitespace_token:
                        character_freqs[word[i]] += 1
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
        return word_freqs, subwords, character_freqs

    def initialize_trie(self, tokens, direction="forward"):
        trie = Trie(direction=direction)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, val)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def fit(self, texts):
        word_freqs, subwords, characters = self.initialize_subwords(texts)

        self.fwd_trie, self.max_subword_len = self.initialize_trie(
            subwords, direction="forward"
        )
        self.bwd_trie, _ = self.initialize_trie(subwords, direction="backward")
        return word_freqs, subwords, characters

    def get_entropy(self, word, direction="forward"):
        if direction == "forward":
            _trie = self.fwd_trie
        else:
            _trie = self.bwd_trie
        return entropy(_trie, word)

    def score(self, word, direction="forward"):
        return self.get_entropy(word, direction=direction)

    def find_local_entropy(self, word, direction="forward", verbose=False):
        entropies = []
        _word = word
        # if direction == "forward":
        #     _word = self.whitespace_token + word
        # else:
        #     _word = word + self.whitespace_token
        for i in range(1, len(_word) + 1):
            if direction == "forward":
                subword = _word[:i]
            else:
                subword = _word[-i:]
            _score = self.get_entropy(subword, direction=direction)
            entropies.append(_score)
            if verbose:
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
        _configure_font()

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

    def branching_word(self, word, direction="forward"):
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

    def branching_words(self, text, direction="forward", flatten=True):
        segments = []
        words = self.pre_tokenize(text)
        for word in words:
            segments.append(self.branching_word(word, direction=direction))
        if flatten:
            segments = [seg for word in segments for seg in word]
        return segments

    def branching(self, texts, direction="forward"):
        return [self.branching_words(text, direction=direction) for text in texts]

    def naive_segment(self, text, direction="forward"):
        words = []

        _start, _pos = 0, 0
        # iterate over the text until we reach the end
        while _pos < len(text):
            _sentencepiece = text[_pos : _pos + self.max_sentencepiece_length]
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
