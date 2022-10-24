from typing import Iterator, Optional, Union
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import Normalizer


class Trainer:
    def __init__(
        self,
    ):
        self._normalizer = None
        self._pre_tokenizer = None

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    @normalizer.setter
    def normalizer(self, normalizer: Normalizer):
        self._normalizer = normalizer

    @property
    def pre_tokenizer(self) -> PreTokenizer:
        return self._pre_tokenizer

    @pre_tokenizer.setter
    def pre_tokenizer(self, pre_tokenizer: PreTokenizer):
        self._pre_tokenizer = pre_tokenizer

    def pre_tokenize(self, text):
        text = self.normalizer.normalize_str(text)
        words_with_offsets = self.pre_tokenizer.pre_tokenize_str(text)
        return [word for word, _ in words_with_offsets]

    def initialize_vocab(self, texts):
        pass

    def fit(self, texts, vocab_size=None):
        pass

    def train(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""
        pass
