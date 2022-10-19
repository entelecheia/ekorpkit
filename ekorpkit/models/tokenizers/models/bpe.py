import json
import os
import collections
import logging
from typing import Iterator, List, Optional, Union, Dict
from tokenizers import AddedToken
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import Normalizer
from tqdm.auto import tqdm


log = logging.getLogger(__name__)


class BpeTrainer:
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        end_of_word_suffix="▁",
        show_progress: bool = True,
        max_merges: int = None,
        verbose=False,
    ):
        self.vocab = {}
        self.merges = {}

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = initial_alphabet
        self.end_of_word_suffix = end_of_word_suffix
        self.show_progress = show_progress
        self.max_merges = max_merges
        self.verbose = verbose

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

    def get_word_freqs(self, texts):
        word_freqs = collections.defaultdict(int)
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                word_freqs[word] += 1
        # Remove words that are too rare
        word_freqs = {
            word: freq
            for word, freq in word_freqs.items()
            if freq >= self.min_frequency
        }
        return word_freqs

    def initialize_vocab(self, texts):
        word_freqs = self.get_word_freqs(texts)

        alphabet = self.initial_alphabet
        for word in word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        vocab = self.special_tokens + alphabet.copy() + [self.end_of_word_suffix]
        splits = {
            word: [c for c in word] + [self.end_of_word_suffix]
            for word in word_freqs.keys()
        }

        return vocab, splits, word_freqs

    def compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def fit(self, texts, vocab_size=None):
        vocab, splits, word_freqs = self.initialize_vocab(texts)

        if vocab_size is None:
            vocab_size = self.vocab_size
        else:
            self.vocab_size = vocab_size

        merges = {}
        num_merges = 0
        if self.show_progress:
            iterator = tqdm(range(vocab_size - len(vocab)))
        while len(vocab) < vocab_size:
            pair_freqs = self.compute_pair_freqs(splits, word_freqs)
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self.merge_pair(*best_pair, splits, word_freqs)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
            num_merges += 1
            if self.show_progress:
                iterator.update(1)
            if self.verbose:
                if (
                    isinstance(self.verbose, int) and num_merges % self.verbose == 0
                ) or (isinstance(self.verbose, list) and num_merges in self.verbose):
                    print(f"--- Round {num_merges}. Vocab size: {len(vocab)} ---")
                    print(f"Merge {num_merges}: {best_pair}")
                    print(f"Number of tokens: {len(vocab)}")
            if self.max_merges is not None and num_merges >= self.max_merges:
                break
        iterator.close()
        if self.verbose:
            print(f"--- Round {num_merges}. Vocab size: {len(vocab)} ---")
            print(f"Merge {num_merges}: {best_pair}")
            print(f"Number of tokens: {len(vocab)}")
            print(f"List of vocab: {vocab}")
        # convert vocab to dict with values being the index of the token
        vocab = {token: i for i, token in enumerate(vocab)}
        return vocab, merges

    def train(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        vocab, merges = self.fit(iterator, vocab_size=self.vocab_size)
        self.vocab = vocab
        self.merges = merges
        return vocab, merges


class BPE:
    """
    An implementation of the BPE (Byte-Pair Encoding) algorithm

    Args:
        vocab (:obj:`Dict[str, int]`, `optional`):
            A dictionnary of string keys and their ids :obj:`{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`, `optional`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`) :obj:`[("a", "b"),...]`

        cache_capacity (:obj:`int`, `optional`):
            The number of words that the BPE cache can contain. The cache allows
            to speed-up the process by keeping the result of the merge operations
            for a number of words.

        dropout (:obj:`float`, `optional`):
            A float between 0 and 1 that represents the BPE dropout to use.

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

        continuing_subword_prefix (:obj:`str`, `optional`):
            The prefix to attach to subword units that don't represent a beginning of word.

        end_of_word_suffix (:obj:`str`, `optional`):
            The suffix to attach to subword units that represent an end of word.

        fuse_unk (:obj:`bool`, `optional`):
            Whether to fuse any subsequent unknown tokens into a single one
    """

    def __init__(
        self,
        vocab=None,
        merges=None,
        cache_capacity=None,
        dropout=None,
        unk_token="</u>",
        continuing_subword_prefix=None,
        end_of_word_suffix="</w>",
        fuse_unk=None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.cache_capacity = cache_capacity
        self.dropout = dropout
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.fuse_unk = fuse_unk

        self.normalizer = None
        self.pre_tokenizer = None
        self.decoder = None

        self.id2token = {}
        if self.vocab is not None:
            self.id2token = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_file(cls, vocab, merges, **kwargs):
        """
        Instantiate a BPE model from the given files.

        This method is roughly equivalent to doing::

           vocab, merges = BPE.read_file(vocab_filename, merges_filename)
           bpe = BPE(vocab, merges)

        If you don't need to keep the :obj:`vocab, merges` values lying around,
        this method is more optimized than manually calling
        :meth:`~tokenizers.models.BPE.read_file` to initialize a :class:`~tokenizers.models.BPE`

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

            merges (:obj:`str`):
                The path to a :obj:`merges.txt` file

        Returns:
            :class:`~tokenizers.models.BPE`: An instance of BPE loaded from these files
        """
        vocab, merges = cls.read_file(vocab, merges)
        return cls(vocab, merges, **kwargs)

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

    @staticmethod
    def read_file(vocab, merges):
        """
        Read a :obj:`vocab.json` and a :obj:`merges.txt` files

        This method provides a way to read and parse the content of these files,
        returning the relevant data structures. If you want to instantiate some BPE models
        from memory, this method gives you the expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

            merges (:obj:`str`):
                The path to a :obj:`merges.txt` file

        Returns:
            A :obj:`Tuple` with the vocab and the merges:
                The vocabulary and merges loaded into memory
        """
        with open(vocab, "r") as f:
            vocab = json.load(f)
        with open(merges, "r") as f:
            merges = [tuple(line.rstrip().split()) for line in f]
            merges = {merge: merge[0] + merge[1] for merge in merges}
        return vocab, merges

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
        if prefix is not None:
            folder = os.path.join(folder, prefix)
        vocab_filename = os.path.join(folder, "vocab.json")
        merges_filename = os.path.join(folder, "merges.txt")
        if not os.path.exists(folder):
            os.makedirs(folder)
        indent = 2 if pretty else None
        json.dump(self.vocab, open(vocab_filename, "w"), indent=indent)
        with open(merges_filename, "w") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}" + os.linesep)
        print(f"Model files saved in {folder}")
        return [vocab_filename, merges_filename]

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
        return self.vocab[token]

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
        if ids is None:
            raise ValueError("None input is not valid. Should be a list of integers.")

        tokens = [self.id_to_token(id) for id in ids]
        # if skip_special_tokens:
        #     tokens = [token for token in tokens if token not in self.special_tokens]

        return "".join(tokens).replace(self.end_of_word_suffix, " ").strip()

    def tokenize(self, sequence):
        """
        Tokenize a sequence

        Args:
            sequence (:obj:`str`):
                A sequence to tokenize

        Returns:
            A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
        """
        sequence = self.normalize(sequence)
        pre_tokenize_result = self.pre_tokenize(sequence)
        pre_tokenized_text = [word for word, _ in pre_tokenize_result]
        splits = [
            [char for char in word] + [self.end_of_word_suffix]
            for word in pre_tokenized_text
        ]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

    def train(self, files: Union[str, List[str]], trainer: BpeTrainer = None):
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
        trainer: BpeTrainer = None,
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer.normalizer = self.normalizer
        trainer.pre_tokenizer = self.pre_tokenizer
        vocab, merges = trainer.train(iterator, length=length)

        self.vocab = vocab
        self.merges = merges
        self.id2token = {v: k for k, v in self.vocab.items()}

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
        return self.pre_tokenizer.pre_tokenize_str(sequence)
