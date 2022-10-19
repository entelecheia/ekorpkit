import collections
import numpy as np
from .base import Trie
from .words import WordSegmenter
from .sentencepiece import SentencePieceTokenizer


class BranchingEntropyTokenizer(SentencePieceTokenizer):
    def __init__(
        self,
        initial_vocab_size=2000,
        vocab_size=1000,
        percent_to_prune=0.2,
        whitespace_token="▁",
        lowercase=True,
        max_sentencepiece_length=20,
        **kwargs,
    ):
        self.ws = WordSegmenter(
            lowercase=lowercase,
            whitespace_token=whitespace_token,
            max_sentencepiece_length=max_sentencepiece_length,
        )
        super().__init__(
            initial_vocab_size=initial_vocab_size,
            vocab_size=vocab_size,
            percent_to_prune=percent_to_prune,
            whitespace_token=whitespace_token,
            lowercase=lowercase,
            max_sentencepiece_length=max_sentencepiece_length,
            **kwargs,
        )

    def initialize_trie(self, tokens):
        trie = Trie()

        maxlen = 0
        for tok in tokens:
            trie.add(tok, self.ws.get_entropy(tok))
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

    def initialize_vocab(self, texts, initial_vocab_size=None):
        if initial_vocab_size is None:
            initial_vocab_size = self.initial_vocab_size
        self.ws.fit(texts)
        text = self.ws.text
        sorted_subwords = self.ws.subwords.most_common()
        characters = self.ws.characters
        tokens = (
            list(characters.items())
            + sorted_subwords[: self.initial_vocab_size - len(characters)]
        )
        tokens = {token: freq for token, freq in tokens}
        tokens = collections.Counter(tokens)
        return text, tokens, characters

    def E_step(self, tokenization, trie):
        for tok in tokenization:
            trie.set_value(tok, self.ws.get_entropy(tok))
        return trie

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
        self.trie, self.maxlen = self.initialize_trie(tokens)
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
