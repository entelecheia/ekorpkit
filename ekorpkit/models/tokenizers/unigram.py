import re
import collections
import copy
from math import log


class UnigramTokenizer:
    def __init__(
        self,
        initial_vocab_size=2000,
        vocab_size=1000,
        whitespace_token="▁",
        unknown_token="[unk]",
        lowercase=True,
        percent_to_prune=0.1,
    ):
        self.vocab_tokenized = {}

        self.word_freqs = {}
        self.subword_freqs = {}
        self.token_freqs = {}
        self.inital_vocab = {}
        self.vocab = {}

        self.initial_vocab_size = initial_vocab_size
        self.vocab_size = vocab_size
        self.suffix_indicator = whitespace_token
        self.unknown_token = unknown_token
        self.lowercase = lowercase
        self.percent_to_prune = percent_to_prune

    def initialize_vocab(self, texts, initial_vocab_size=None):
        if initial_vocab_size is None:
            initial_vocab_size = self.initial_vocab_size
        self.word_freqs = self.get_word_frequency(texts)
        sorted_subwords, character_freqs = self.initialize_subwords(self.word_freqs)
        token_freqs = (
            list(character_freqs.items())
            + sorted_subwords[: self.initial_vocab_size - len(character_freqs)]
        )
        self.token_freqs = {token: freq for token, freq in token_freqs}

        total_sum = sum([freq for _, freq in self.token_freqs.items()])
        self.inital_vocab = {
            token: -log(freq / total_sum) for token, freq in self.token_freqs.items()
        }

    def get_word_frequency(self, texts):
        word_freqs = {}
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        return word_freqs

    def initialize_subwords(self, word_freqs, verbose=True):
        character_freqs = collections.defaultdict(int)
        subwords_freqs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word)):
                character_freqs[word[i]] += freq
                # Loop through the subwords of length at least 2
                for j in range(i + 2, len(word) + 1):
                    subwords_freqs[word[i:j]] += freq

        # Sort subwords by frequency
        sorted_subwords = sorted(
            subwords_freqs.items(), key=lambda x: x[1], reverse=True
        )
        if verbose:
            print("Top 10 subwords: {}".format(sorted_subwords[:10]))
        return sorted_subwords, character_freqs

    def encode_word(self, word, vocab):
        best_segmentations = [{"start": 0, "score": 1}] + [
            {"start": None, "score": None} for _ in range(len(word))
        ]
        for start_idx in range(len(word)):
            # This should be properly filled by the previous steps of the loop
            best_score_at_start = best_segmentations[start_idx]["score"]
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in vocab and best_score_at_start is not None:
                    score = vocab[token] + best_score_at_start
                    # If we have found a better segmentation ending at end_idx, we update
                    if (
                        best_segmentations[end_idx]["score"] is None
                        or best_segmentations[end_idx]["score"] > score
                    ):
                        best_segmentations[end_idx] = {
                            "start": start_idx,
                            "score": score,
                        }

        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # We did not find a tokenization of the word -> unknown
            return [self.unknown_token], None

        score = segmentation["score"]
        start = segmentation["start"]
        end = len(word)
        tokens = []
        while start != 0:
            tokens.insert(0, word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        return tokens, score

    def compute_loss(self, vocab):
        loss = 0
        for word, freq in self.word_freqs.items():
            _, word_loss = self.encode_word(word, vocab)
            loss += freq * word_loss
        return loss

    def compute_scores(self, vocab):
        scores = {}
        vocab_loss = self.compute_loss(vocab)
        for token, _ in vocab.items():
            # We always keep tokens of length 1
            if len(token) == 1:
                continue
            vocab_without_token = copy.deepcopy(vocab)
            _ = vocab_without_token.pop(token)
            scores[token] = self.compute_loss(vocab_without_token) - vocab_loss
        return scores

    def fit(
        self,
        texts,
        vocab_size=None,
        percent_to_prune=None,
        verbose=True,
        print_every=2,
    ):
        if vocab_size is None:
            vocab_size = self.vocab_size
        if percent_to_prune is None:
            percent_to_prune = self.percent_to_prune

        self.initialize_vocab(texts)
        vocab = copy.deepcopy(self.inital_vocab)
        iter = 0

        while len(vocab) > vocab_size:
            scores = self.compute_scores(vocab)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            # Remove percent_to_remove tokens with the lowest scores.
            for i in range(int(len(vocab) * percent_to_prune)):
                if len(vocab) <= vocab_size:
                    break
                _ = self.token_freqs.pop(sorted_scores[i][0])

            total_sum = sum([freq for _, freq in self.token_freqs.items()])
            vocab = {
                token: -log(freq / total_sum)
                for token, freq in self.token_freqs.items()
            }

            if verbose and print_every > 0 and i % print_every == 0:
                print("Iteration: {}, vocab size: {}".format(iter, len(vocab)))
            iter += 1

        self.vocab = vocab
        if verbose:
            print("Final vocab size: {}".format(len(self.vocab)))

    def pre_tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.split()

    def tokenize(self, text):
        words = self.pre_tokenize(text)
        encoded_words = [self.encode_word(word, self.vocab)[0] for word in words]
        return sum(encoded_words, [])
