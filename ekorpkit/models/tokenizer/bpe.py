import re
import collections


class BytePairEncoder:
    def __init__(self, whitespace_token="▁", unknown_token="</u>", lowercase=True):
        self.merges = {}
        self.tokens = {}
        self.sorted_tokens = []
        self.vocab = {}
        self.vocab_tokenized = {}
        self.whitespace_token = whitespace_token
        self.lowercase = lowercase
        self.unknown_token = unknown_token

    def format_word(self, word):
        return " ".join(list(word))

    def initialize_vocab(self, texts):
        vocab = {}
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                word = self.format_word(word)
                vocab[word] = vocab.get(word, 0) + 1
        return vocab

    def get_tokens_from_vocab(self, vocab):
        tokens = collections.defaultdict(int)
        vocab_tokenized = {}
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens[token] += freq
            vocab_tokenized["".join(word_tokens)] = word_tokens
        return tokens, vocab_tokenized

    def get_bigram_counts(self, vocab):
        pairs = {}
        for word, count in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    def merge_vocab(self, pair, vocab_in):
        vocab_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        bytepair = "".join(pair)
        for word in vocab_in:
            w_out = p.sub(bytepair, word)
            vocab_out[w_out] = vocab_in[word]
        return vocab_out, (bigram, bytepair)

    def find_merges(self, vocab, tokens, num_merges, indices_to_print=[]):
        merges = []
        for i in range(num_merges):
            pairs = self.get_bigram_counts(vocab)
            best_pair = max(pairs, key=pairs.get)
            best_count = pairs[best_pair]
            vocab, (bigram, bytepair) = self.merge_vocab(best_pair, vocab)
            merges.append((r"(?<!\S)" + bigram + r"(?!\S)", bytepair))
            tokens, vocab_tokenized = self.get_tokens_from_vocab(vocab)
            if indices_to_print and i in indices_to_print:
                print(f"Merge {i}: {best_pair} with count {best_count}")
                print("All tokens: {}".format(tokens.keys()))
                print("Number of tokens: {}".format(len(tokens.keys())))
        return vocab, tokens, merges, vocab_tokenized

    def fit(self, texts, num_merges, indices_to_print=[]):
        vocab = self.initialize_vocab(texts)

        self.vocab, self.tokens, self.merges, self.vocab_tokenized = self.find_merges(
            vocab, self.tokens, num_merges, indices_to_print
        )
        self.sorted_tokens = sorted(
            self.tokens.keys(),
            key=self.measure_token_length,
            reverse=True,
        )

    def measure_token_length(self, token):
        whitespace_symbol_len = len(self.whitespace_token)
        if token[-whitespace_symbol_len:] == self.whitespace_token:
            return len(token) - whitespace_symbol_len + 1
        else:
            return len(token)

    def encode_word(self, string, sorted_tokens):

        if string == "":
            return []
        sorted_tokens = sorted_tokens.copy()
        if sorted_tokens == []:
            return [self.unknown_token]

        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace(".", "[.]"))

            matched_positions = [
                (m.start(0), m.end(0)) for m in re.finditer(token_reg, string)
            ]
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [
                matched_position[0] for matched_position in matched_positions
            ]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.encode_word(
                    substring, sorted_tokens=sorted_tokens[i + 1 :]
                )
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.encode_word(
                remaining_substring, sorted_tokens=sorted_tokens[i + 1 :]
            )
            break
        return string_tokens

    def print_enocoded_word(self, word):
        print("Encoding word: {}...".format(word))
        if word in self.vocab_tokenized:
            print("Encoding of the known word:")
            print(self.vocab_tokenized[word])
            print("Encoding treating the known word as unknown:")
            print(self.encode_word(word, self.sorted_tokens))
        else:
            print("Encoding of the unknown word:")
            print(self.encode_word(word, self.sorted_tokens))

    def pre_tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return [word + self.whitespace_token for word in text.split()]

    def tokenize(self, text):
        words = self.pre_tokenize(text)
        encoded_words = [self.encode_word(word, self.sorted_tokens) for word in words]
        return sum(encoded_words, [])
