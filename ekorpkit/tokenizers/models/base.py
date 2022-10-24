import json
import logging
from typing import Iterator, List, Optional, Union, Dict
from ..trainers.base import Trainer


log = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        vocab=None,
        **kwargs,
    ):
        self.initialize_vocab(vocab, **kwargs)
        self.normalizer = None
        self.pre_tokenizer = None
        self.decoder = None

    def initialize_vocab(self, vocab, **kwargs):
        self.vocab = vocab
        self.token2id = {}
        self.id2token = {}
        if vocab:
            self.token2id = {token: id for id, token in enumerate(self.vocab)}
            self.id2token = {v: k for k, v in self.token2id.items()}

    def id_to_token(self, id):
        """
        Get the token associated to an ID

        Args:
            id (:obj:`int`):
                An ID to convert to a token

        Returns:
            :obj:`str`: The token associated to the ID
        """
        return self.id2token[id]

    def save(self, folder, prefix=None, pretty: bool = False):
        """
        Save the current model

        Save the current model in the given folder, using the given prefix for the various
        files that will get created.
        Any file with the same name that already exists in this folder will be overwritten.

        Args:
            folder (:obj:`str`):
                The path to the target folder in which to save the various files

            prefix (:obj:`str`, `optional`):
                An optional prefix, used to prefix each file name

        Returns:
            :obj:`List[str]`: The list of saved files
        """
        pass

    def to_str(self, pretty: bool = False):
        """Get a serialized JSON version of the Tokenizer as a str

        Args:
            pretty: bool:
                Whether the JSON string should be prettified

        Returns:
            str
        """
        return json.dumps(self.vocab, indent=2 if pretty else None)

    def token_to_id(self, token):
        """
        Get the ID associated to a token

        Args:
            token (:obj:`str`):
                A token to convert to an ID

        Returns:
            :obj:`int`: The ID associated to the token
        """
        return self.token2id[token]

    def encode(
        self,
        sequence: str,
        pair: Optional[str] = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ):
        """Encode the given sequence and pair. This method can process raw text sequences as well
        as already pre-tokenized sequences.

        Args:
            sequence: InputSequence:
                The sequence we want to encode. This sequence can be either raw text or
                pre-tokenized, according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            is_pretokenized: bool:
                Whether the input is already pre-tokenized.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            An Encoding
        """
        sequence_ids = [self.token_to_id(token) for token in self.tokenize(sequence)]
        # if add_special_tokens:
        #     sequence_ids = (
        #         [self.token_to_id("[CLS]")] + sequence_ids + [self.token_to_id("[SEP]")]
        #     )

        pair_ids = []
        if pair is not None:
            pair_ids = [self.token_to_id(token) for token in self.tokenize(pair)]
            # if add_special_tokens:
            #     pair_ids += [self.token_to_id("[SEP]")]

        return sequence_ids + pair_ids

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        pass

    def tokenize(self, sequence):
        """
        Tokenize a sequence

        Args:
            sequence (:obj:`str`):
                A sequence to tokenize

        Returns:
            A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
        """
        pass

    def train(self, files: Union[str, List[str]], trainer: Trainer = None):
        """Train the model using the given files"""

        if isinstance(files, str):
            files = [files]
        texts = []
        length = 0
        for file in files:
            with open(file, "r") as f:
                texts.append(f.read())
            length += len(texts[-1])
        self.train_from_iterator(texts, trainer, length)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        trainer: Trainer = None,
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""
        pass

    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        """Returns the vocabulary

        Args:
            with_added_tokens: boolean:
                Whether to include the added tokens in the vocabulary

        Returns:
            The vocabulary
        """
        return self.vocab

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """Return the size of vocabulary, with or without added tokens.

        Args:
            with_added_tokens: (`optional`) bool:
                Whether to count in added special tokens or not

        Returns:
            Size of vocabulary
        """
        return len(self.vocab)

    def normalize(self, sequence: str) -> str:
        """Normalize the given sequence

        Args:
            sequence: str:
                The sequence to normalize

        Returns:
            The normalized string
        """
        return self.normalizer.normalize_str(sequence)

    def pre_tokenize(self, sequence: str):
        """Pre-tokenize the given sequence

        Args:
            sequence: str:
                The sequence to pre-tokenize

        Returns:
            A list of tuple (str, offsets)
        """
        sequence = self.normalize(sequence)
        words_with_offsets = self.pre_tokenizer.pre_tokenize_str(sequence)
        return [word for word, _ in words_with_offsets]
