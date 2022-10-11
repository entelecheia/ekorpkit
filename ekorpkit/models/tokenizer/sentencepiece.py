# original source:
# https://gist.githubusercontent.com/Jmkernes/01da3b560eb12218119f00a0969787e8/raw/2cc222706bdd8b15f2807b6fbd2c0612d7ce17ea/sentence_piece_trainer.py
import re
import collections
import numpy as np
from scipy.special import digamma


# To efficiently determine the next possible words
# We need a Trie data structure
class Trie:
    def __init__(self, end_symbol="<END>"):
        self.root = {}
        self.end_symbol = end_symbol

    def add(self, word, value):
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node[self.end_symbol] = value

    def get_value(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                return 0
            node = node[ch]
        if self.end_symbol not in node:
            return 0
        return node[self.end_symbol]

    def set_value(self, word, value):
        node = self.root
        for ch in word:
            if ch not in node:
                raise ValueError("word not in trie")
            node = node[ch]
        if self.end_symbol not in node:
            raise ValueError("word not in trie")
        node[self.end_symbol] = value


class SentencePieceTokenizer:
    def __init__(
        self,
        initial_vocab_size=2000,
        vocab_size=1000,
        percent_to_prune=0.2,
        whitespace_token="▁",
        lowercase=True,
    ):
        self.trie = None
        self.maxlen = None

        self.initial_vocab_size = initial_vocab_size
        self.vocab_size = vocab_size
        self.percent_to_prune = percent_to_prune
        self.whitespace_token = whitespace_token
        self.lowercase = lowercase

    def _initialize_trie(self, tokens):
        trie = Trie()
        norm = sum(list(tokens.values()))
        logsum = digamma(norm)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, digamma(val) - logsum)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def pre_tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"\s+", self.whitespace_token, text)
        return text

    def initialize_vocab(self, texts, initial_vocab_size=None):
        if initial_vocab_size is None:
            initial_vocab_size = self.initial_vocab_size
        text = self.pre_tokenize(" ".join(texts))
        word_freqs = collections.Counter(text.split(self.whitespace_token))
        sorted_subwords, characters = self.initialize_subwords(word_freqs)
        tokens = (
            list(characters.items())
            + sorted_subwords[: self.initial_vocab_size - len(characters)]
        )
        tokens = {token: freq for token, freq in tokens}
        tokens = collections.Counter(tokens)
        return text, tokens, characters

    def initialize_subwords(self, word_freqs, verbose=True):
        character_freqs = collections.defaultdict(int)
        subwords_freqs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            word = self.whitespace_token + word
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

    def forward_step(self, text, trie):
        N = len(text)

        # d[i] contains the maximum log_prob of any tokenization
        # of text[:i], initialized to 0 (i.e. log(0)=-infty)
        d = [-np.inf] * (N + 1)

        # p[i] (stands for parent) contains the number of characters of
        # the final token in the most likely sequence that ends at index i
        p = [None] * (N + 1)
        d[0] = 0

        for i in range(1, N + 1):

            # find all possible final words. Have to look back
            # a distance set by the length of the longest token
            for j in range(max(i - self.maxlen, 0), i):

                final_token = text[j:i]
                final_value = trie.get_value(final_token)

                # if the current ending word has a higher log-probability,
                # save that value and store the word (i.e. # chars to backtrack)
                if final_value and d[j] + final_value > d[i]:
                    d[i] = d[j] + final_value
                    p[i] = len(final_token)
            if p[i] is None:
                raise ValueError(f"Encountered unknown token '{text[i-1]}'.")

        loss = d[-1]
        return loss, p

    def backward_step(self, text, p):
        idx = len(p)
        tokenization = []
        while idx > 1:
            # move back the number of steps p tells you to
            next_idx = idx - p[idx - 1]

            # extract the final token
            tok = text[next_idx - 1 : idx - 1]
            tokenization.append(tok)

            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def E_step(self, tokenization, trie):
        # get the new token counts based on updated tokenization
        counts = collections.Counter(tokenization)
        norm = sum(list(counts.values()))

        # Bayesianify them: https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        # https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc
        # we are returning the log probabilties here (alpha=0 prior)
        logsum = digamma(norm)
        for k, v in counts.items():
            counts[k] = digamma(v) - logsum

        for k, v in counts.items():
            trie.set_value(k, v)
        return trie

    def M_step(self, text, trie):
        loss, p = self.forward_step(text, trie)
        tokenization = self.backward_step(text, p)
        return tokenization, loss

    def EM_step(self, text, tokenization, trie):
        trie = self.E_step(tokenization, trie)
        tokenization, loss = self.M_step(text, trie)
        return loss, tokenization, trie

    def EM_round(self, text, tokens, delta=0.01, max_iter=10):
        tokenization, old_loss = self.M_step(text, self.trie)
        for step in range(max_iter):
            print(f"EM iter {step}: ", end="")
            loss, tokenization, trie = self.EM_step(text, tokenization, self.trie)
            print(f"Loss={loss:.2f}")
            if abs(old_loss - loss) < delta:
                break
            old_loss = loss

    def prune_tokens(self, tokens, characters, vocab_size, percent_to_prune=None):
        """Tokens are passed by reference and modified in place.
        Returns:
            True: to indicate to caller that more rounds are needed
            False: to indicate we successfully hit the target vocab size
            ValueError: if the vocab size cannot be reached."""
        sorted_tokens = tokens.most_common()
        N = len(sorted_tokens)
        if percent_to_prune is None:
            percent_to_prune = self.percent_to_prune
        n_trim = int(percent_to_prune * N)
        for i in reversed(range(N)):
            if N <= vocab_size:
                return False
            if n_trim <= 0:
                return True
            tok = sorted_tokens[i][0]
            if tok not in characters:
                self.trie.set_value(
                    tok, 0
                )  # we need to delete it from the trie (that sticks around)
                tokens.pop(
                    tok
                )  # also need to delete from tokens, so the next round doesn't see it
                n_trim -= 1
                N -= 1
        if n_trim > 0:
            raise ValueError(
                "Could not reduce tokens further. Please increase vocab size"
            )
        return False

    def fit(self, texts, vocab_size=None, delta=0.01, max_iter=5, max_rounds=5):
        """To turn off pruning, just set max_rounds=1"""
        # text = re.sub(" ", "_", text)
        text, tokens, characters = self.initialize_vocab(texts)
        if vocab_size is None:
            vocab_size = self.vocab_size

        if vocab_size > len(tokens):
            raise ValueError(
                f"Vocab size is larger than the availble number of tokens {len(tokens)}."
            )
        self.trie, self.maxlen = self._initialize_trie(tokens)
        for i in range(1, max_rounds + 1):
            print(f"--- Round {i}. Vocab size: {len(tokens)} ---")
            self.EM_round(text, tokens, delta, max_iter)
            if not self.prune_tokens(tokens, characters, vocab_size):
                break
        self.vocab_size = len(tokens)

    def generalized_forward_step(self, text, trie, nbest_size=1):
        N = len(text)
        d = [-np.inf] * (N + 1)
        p = [None] * (N + 1)
        d[0] = 0
        for i in range(1, N + 1):
            d_queue = []
            p_queue = []
            for j in range(max(i - self.maxlen, 0), i):
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

    def tokenize(self, text, nbest_size=1):
        text = self.pre_tokenize(text)
        if self.trie is None:
            raise ValueError("Trainer has not yet been fit. Cannot tokenize.")
        p = self.generalized_forward_step(text, self.trie, nbest_size)
        tokenization = self.generalized_backward_step(text, p)
        return tokenization
