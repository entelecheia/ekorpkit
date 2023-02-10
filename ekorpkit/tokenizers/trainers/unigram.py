import collections
import logging
import numpy as np
from typing import Iterator, Optional, Union
from tqdm.auto import tqdm
from scipy.special import digamma
from .base import Trainer
from ..utils.trie import Trie


log = logging.getLogger(__name__)


class UnigramTrainer(Trainer):
    """
    Trainer capable of training a Unigram model
    Args:
        vocab_size (:obj:`int`):
            The size of the final vocabulary, including all tokens and alphabet.
        show_progress (:obj:`bool`):
            Whether to show progress bars while training.
        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.
        initial_alphabet (:obj:`List[str]`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.
        shrinking_factor (:obj:`float`):
            The shrinking factor used at each step of the training to prune the
            vocabulary.
        unk_token (:obj:`str`):
            The token used for out-of-vocabulary tokens.
        max_piece_length (:obj:`int`):
            The maximum length of a given token.
        n_sub_iterations (:obj:`int`):
            The number of iterations of the EM algorithm to perform before
            pruning the vocabulary.
    """

    def __init__(
        self,
        vocab_size=8000,
        show_progress=True,
        special_tokens=[],
        initial_alphabet=[],
        shrinking_factor=0.75,
        unk_token=None,
        max_piece_length=16,
        n_sub_iterations=5,
        max_rounds=5,
        delta=0.01,
        min_frequency: int = 2,
        verbose=False,
    ):
        self.vocab_size = vocab_size
        self.show_progress = show_progress
        self.special_tokens = special_tokens
        self.initial_alphabet = initial_alphabet
        self.shrinking_factor = shrinking_factor
        self.unk_token = unk_token
        self.max_piece_length = max_piece_length
        self.n_sub_iterations = n_sub_iterations
        self.max_rounds = max_rounds
        self.delta = delta
        self.min_frequency = min_frequency
        self.verbose = verbose

        super().__init__()

    def get_word_freqs(self, words):
        word_freqs = collections.Counter(words)
        # Remove words that are too rare
        word_freqs = {
            word: freq
            for word, freq in word_freqs.items()
            if freq >= self.min_frequency
        }
        return word_freqs

    def initialize_subwords(self, word_freqs):
        character_freqs = collections.defaultdict(int)
        subwords_freqs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            # word = self.whitespace_token + word
            for i in range(len(word)):
                character_freqs[word[i]] += freq
                # Loop through the subwords of length at least 2
                for j in range(i + 2, len(word) + 1):
                    subwords_freqs[word[i:j]] += freq
        # Sort subwords by frequency
        sorted_subwords = sorted(
            subwords_freqs.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_subwords, character_freqs

    def initialize_vocab(self, words):
        word_freqs = self.get_word_freqs(words)
        sorted_subwords, characters = self.initialize_subwords(word_freqs)

        alphabet = {
            char: self.min_frequency
            for char in self.initial_alphabet
            if char not in characters
        }
        characters.update(alphabet)
        tokens = list(characters.items()) + sorted_subwords
        tokens = {token: freq for token, freq in tokens}
        tokens = collections.Counter(tokens)
        return tokens, characters

    def initialize_trie(self, tokens):
        trie = Trie()
        norm = sum(list(tokens.values()))
        logsum = digamma(norm)

        maxlen = 0
        for tok, val in tokens.items():
            trie.add(tok, digamma(val) - logsum)
            maxlen = max(maxlen, len(tok))

        return trie, maxlen

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

    def EM_round(self, text, tokens, delta=0.01, n_iterations=10):
        tokenization, old_loss = self.M_step(text, self.trie)
        for step in range(n_iterations):
            print(f"EM iter {step}: ", end="")
            loss, tokenization, trie = self.EM_step(text, tokenization, self.trie)
            print(f"Loss={loss:.2f}")
            if abs(old_loss - loss) < delta:
                break
            old_loss = loss

    def prune_tokens(self, tokens, characters, vocab_size, shrinking_factor=None):
        """Tokens are passed by reference and modified in place.
        Returns:
            True: to indicate to caller that more rounds are needed
            False: to indicate we successfully hit the target vocab size
            ValueError: if the vocab size cannot be reached."""
        sorted_tokens = tokens.most_common()
        N = len(sorted_tokens)
        if shrinking_factor is None:
            shrinking_factor = self.shrinking_factor
        n_trim = int((1 - shrinking_factor) * N)
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

    def fit(self, texts):
        """To turn off pruning, just set max_rounds=1"""
        words = []
        iterator = tqdm(texts) if self.show_progress else texts
        for text_batch in iterator:
            if isinstance(text_batch, str):
                text_batch = [text_batch]
            for text in text_batch:
                words += self.pre_tokenize(text)
        text = "".join(words)
        tokens, characters = self.initialize_vocab(words)
        vocab_size = self.vocab_size

        if vocab_size > len(tokens):
            raise ValueError(
                f"Vocab size is larger than the availble number of tokens {len(tokens)}."
            )
        self.trie, self.maxlen = self.initialize_trie(tokens)
        for i in range(1, self.max_rounds + 1):
            print(f"--- Round {i}. Vocab size: {len(tokens)} ---")
            self.EM_round(text, tokens, self.delta, self.n_sub_iterations)
            if not self.prune_tokens(tokens, characters, vocab_size):
                break
        self.vocab_size = len(tokens)
        return tokens

    def train(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        return self.fit(iterator)
