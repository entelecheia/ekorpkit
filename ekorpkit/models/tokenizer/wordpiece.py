import re
import collections


class WordPieceTokenizer:
    def __init__(
        self,
        suffix_indicator="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        unknown_token="[UNK]",
        lowercase=True,
    ):
        self.vocab_tokenized = {}

        self.word_freqs = {}
        self.characters = []
        self.vocab = []
        self.initial_splits = {}

        self.suffix_indicator = suffix_indicator
        self.special_tokens = special_tokens
        self.unknown_token = unknown_token
        self.lowercase = lowercase

    def format_word(self, word):
        return " ".join(list(word))

    def initialize_vocab(self, texts):
        self.word_freqs = self.get_word_frequency(texts)
        self.characters = self.initialize_characters(self.word_freqs)
        self.initial_splits = self.split_words(self.word_freqs)
        self.vocab = self.special_tokens + self.characters.copy()

    def get_word_frequency(self, texts):
        word_freqs = {}
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] = word_freqs.get(word, 0) + 1
        return word_freqs

    def initialize_characters(self, word_freqs):
        characters = []
        for word in word_freqs.keys():
            if word[0] not in characters:
                characters.append(word[0])
            for letter in word[1:]:
                if f"{self.suffix_indicator}{letter}" not in characters:
                    characters.append(f"{self.suffix_indicator}{letter}")
        return characters

    def split_words(self, word_freqs):
        splits = {
            word: [
                c if i == 0 else f"{self.suffix_indicator}{c}"
                for i, c in enumerate(word)
            ]
            for word in word_freqs.keys()
        }
        return splits

    def compute_pair_scores(self, splits):
        letter_freqs = collections.defaultdict(int)
        pair_freqs = collections.defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith(self.suffix_indicator) else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def fit(self, texts, vocab_size, verbose=True, print_every=100):
        self.initialize_vocab(texts)
        i = 0
        splits = self.initial_splits.copy()
        while len(self.vocab) < vocab_size:
            scores = self.compute_pair_scores(splits)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            splits = self.merge_pair(*best_pair, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith(self.suffix_indicator)
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)
            if verbose and print_every > 0 and i % print_every == 0:
                print("Iteration: {}, vocab size: {}".format(i, len(self.vocab)))
                print("Best pair: {}".format(best_pair))
                print("New token: {}".format(new_token))
            i += 1
        if verbose:
            print("Final vocab size: {}".format(len(self.vocab)))

    def encode_word(self, word):
        if word in self.vocab_tokenized:
            return self.vocab_tokenized[word]
        else:
            tokens = []
            while len(word) > 0:
                i = len(word)
                while i > 0 and word[:i] not in self.vocab:
                    i -= 1
                if i == 0:
                    return [self.unknown_token]
                tokens.append(word[:i])
                word = word[i:]
                if len(word) > 0:
                    word = f"{self.suffix_indicator}{word}"
            self.vocab_tokenized[word] = tokens
            return tokens

    def print_enocoded_word(self, word):
        print("Encoding word: {}...".format(word))
        if word in self.vocab_tokenized:
            print("Encoding of the known word:")
            print(self.vocab_tokenized[word])
            print("Encoding treating the known word as unknown:")
            print(self.encode_word(word))
        else:
            print("Encoding of the unknown word:")
            print(self.encode_word(word))

    def pre_tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.split()

    def tokenize(self, text):
        words = self.pre_tokenize(text)
        encoded_words = [self.encode_word(word) for word in words]
        return sum(encoded_words, [])
