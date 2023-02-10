import os
import ekorpkit.io.zjson as json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Iterator, Optional, Union
from enum import Enum
from ekorpkit.visualize.base import get_plot_font
from .base import Model
from ..trainers.branching import BranchingEntropyTrainer
from ..utils.trie import Trie
from ..utils.score import (
    Scores,
    ScoreResult,
    scores,
)


log = logging.getLogger(__name__)


class BranchingDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


class BranchingEntropy(Model):
    def __init__(
        self,
        vocab=None,
        branching_threshold=0.0,
        cohesion_threshold=0.0,
        whitespace_token="▁",
        whitespace_token_as_prefix=True,
        verbose=False,
        **kwargs,
    ):
        self.branching_threshold = branching_threshold
        self.cohesion_threshold = cohesion_threshold
        self.whitespace_token = whitespace_token
        self.whitespace_token_as_prefix = whitespace_token_as_prefix
        self.verbose = verbose
        super().__init__(vocab)
        # self._fwd_scores = {}
        # self._bwd_scores = {}

    def initialize_vocab(self, vocab, **kwargs):
        self.vocab = {}
        self.token2id = {}
        self.id2token = {}
        self.fwd_trie = None
        self.bwd_trie = None
        self.max_piece_length = None
        if vocab:
            self.vocab = vocab
            # if (
            #     self.direction == BranchingDirection.FORWARD
            #     or self.direction == BranchingDirection.BOTH
            # ):
            #     self.fwd_trie, self.max_piece_length = self.initialize_trie(
            #         vocab, BranchingDirection.FORWARD
            #     )
            # if (
            #     self.direction == BranchingDirection.BACKWARD
            #     or self.direction == BranchingDirection.BOTH
            # ):
            #     self.bwd_trie, self.max_piece_length = self.initialize_trie(
            #         vocab, BranchingDirection.BACKWARD
            #     )
            self.fwd_trie, self.bwd_trie, self.max_piece_length = self.initialize_trie(
                vocab
            )

    def initialize_trie(self, tokens):
        fwd_trie = Trie(direction=BranchingDirection.FORWARD)
        bwd_trie = Trie(direction=BranchingDirection.BACKWARD)

        maxlen = 0
        # for tok, val in tqdm(tokens.items(), desc=f"Building {direction} trie"):
        for tok, val in tqdm(tokens.items(), desc=f"Building tries"):
            fwd_trie.add(tok, val)
            bwd_trie.add(tok, val)
            maxlen = max(maxlen, len(tok))

        return fwd_trie, bwd_trie, maxlen

    def get_scores(
        self, word, direction: BranchingDirection = BranchingDirection.FORWARD
    ) -> Scores:
        if direction == BranchingDirection.FORWARD:
            _trie = self.fwd_trie
        else:
            _trie = self.bwd_trie
        return scores(_trie, word, self.whitespace_token)

    def show_local_entropy(self, word):
        results = self.find_local_entropy(word)
        results = [r.dict() for r in results]

        df = pd.DataFrame(results)
        df = pd.concat(
            [
                df.char,
                pd.json_normalize(df["L_scores"]).add_prefix("L_"),
                # df.avg_coh,
                pd.json_normalize(df["diffs"]).add_prefix("D_"),
                pd.json_normalize(df["R_scores"]).add_prefix("R_"),
            ],
            axis=1,
        )
        return df

    def find_local_entropy(self, word) -> Iterator[ScoreResult]:
        # get the local entropy and the difference in entropy

        results: Iterator[ScoreResult] = []
        for i, char in enumerate(word):
            L_subword = word[: i + 1]
            R_subword = word[i + 1 :]
            if not L_subword.startswith(self.whitespace_token):
                L_subword = self.whitespace_token + L_subword
            if not R_subword.endswith(self.whitespace_token):
                R_subword += self.whitespace_token
            l_scores = self.get_scores(L_subword, direction="forward")
            r_scores = self.get_scores(R_subword, direction="backward")
            result = ScoreResult(char=char)
            result.L_scores = l_scores
            result.R_scores = r_scores
            # result.avg_coh = (l_scores.cohesion + r_scores.cohesion) / 2

            results.append(result)
            if self.verbose:
                print(L_subword, result)

        # calculate the difference in entropy and cohesion
        if len(results) > 1:
            for i, result in enumerate(results):
                if i == 0:
                    result_prev: ScoreResult = None
                    result_next: ScoreResult = results[i + 1]
                    Lscores_prev: Scores = None
                    Rscores_next: Scores = result_next.R_scores
                elif i == len(results) - 1:
                    result_prev: ScoreResult = results[i - 1]
                    result_next: ScoreResult = None
                    Lscores_prev: Scores = result_prev.L_scores
                    Rscores_next: Scores = None
                else:
                    result_prev: ScoreResult = results[i - 1]
                    result_next: ScoreResult = results[i + 1]
                    Lscores_prev: Scores = result_prev.L_scores
                    Rscores_next: Scores = result_next.R_scores
                result.diffs.f_ent = (
                    result.L_scores.entropy - Lscores_prev.entropy
                    if Lscores_prev and result.L_scores.entropy and Lscores_prev.entropy
                    else 0
                )
                result.diffs.b_ent = (
                    result.R_scores.entropy - Rscores_next.entropy
                    if Rscores_next and result.R_scores.entropy and Rscores_next.entropy
                    else 0
                )
                result.diffs.coh = (
                    result_next.L_scores.cohesion - result.L_scores.cohesion
                    if result_next
                    and result_next.L_scores.cohesion
                    and result.L_scores.cohesion
                    else 0
                )
                # result.diffs.avg_coh = (
                #     result_next.avg_coh - result.avg_coh if result_next else 0
                # )
        return results

    # plot entropies
    def plot_local_entropy(self, word, figsize=(12, 5)):
        get_plot_font()

        results = self.find_local_entropy(word)
        chars = [result.char for result in results]
        L_entropies = [result.L_scores.entropy for result in results]
        R_entropies = [result.R_scores.entropy for result in results]
        L_cohesions = [result.L_scores.cohesion for result in results]
        # R_cohesions = [result.R_scores.cohesion for result in results]
        # avg_cohesions = [result.avg_coh for result in results]
        if chars[0] == self.whitespace_token:
            chars = chars[1:]
            L_entropies = L_entropies[1:]
            R_entropies = R_entropies[1:]
            L_cohesions = L_cohesions[1:]

        plt.figure(figsize=figsize)
        plt.plot(L_entropies, label="fwd. entropy", color="blue", marker="o")
        plt.plot(R_entropies, label="bwd. entropy", color="green", marker="o")
        plt.xticks(range(len(chars)), chars)
        plt.legend(loc="upper left")

        # plot cohesions on the right y-axis
        plt.twinx()
        plt.plot(
            L_cohesions,
            label="cohesions",
            color="red",
            linestyle="--",
            marker="o",
        )
        # plt.plot(
        #     R_cohesions, label="R cohesions", color="orange", linestyle="--", marker="o"
        # )
        plt.legend(loc="upper right")
        plt.show()

    def tokenize_word(self, word):
        # if there is a spike in entropy, then we should segment
        # Here the spike means that there is a sudden increase in entropy followed by a decrease.
        # We can use the difference in entropy to detect the spike.

        # get the local entropy and the difference in entropy
        results = self.find_local_entropy(word)
        f_diffs = [result.diffs.f_ent for result in results]
        b_diffs = [result.diffs.b_ent for result in results]
        coh_diffs = [result.diffs.coh for result in results]

        def check_entropy_threshold(f_diffs, b_diffs, pos, threshold, start_idx):
            if pos < len(f_diffs) - 2 and pos > start_idx:
                return b_diffs[pos] > threshold and b_diffs[pos - 1] < 0
            elif pos < len(f_diffs) - 2 and pos > start_idx:
                return (f_diffs[pos] > threshold and f_diffs[pos + 1] < 0) or (
                    b_diffs[pos] > threshold and b_diffs[pos - 1] < 0
                )
            elif pos == len(f_diffs) - 2 and pos > start_idx:
                return f_diffs[pos] > threshold and f_diffs[pos + 1] < 0
            elif pos == len(f_diffs) - 1 and pos > start_idx:
                return f_diffs[pos] > threshold
            else:
                return False

        def check_cohesion_threshold(coh_diffs, pos, threshold, start_idx):
            if threshold is None:
                return False
            if pos < len(coh_diffs) - 1 and pos > start_idx:
                return coh_diffs[pos] < threshold
            else:
                return False

        # get the spikes
        spikes = []
        start_idx = 1 if word[0] == self.whitespace_token else 0
        if len(word) > 1:
            for i in range(start_idx, len(f_diffs)):
                if check_entropy_threshold(
                    f_diffs, b_diffs, i, self.branching_threshold, start_idx
                ) or check_cohesion_threshold(
                    coh_diffs, i, self.cohesion_threshold, start_idx
                ):
                    spikes.append(i)

        # segment the word
        segments = []
        start = 0
        for spike in spikes:
            segments.append(word[start : spike + 1])
            start = spike + 1
        if start < len(word):
            segments.append(word[start:])
        # if self.whitespace_token_as_prefix and len(segments) > 0:
        #     segments[0] = self.whitespace_token + segments[0]
        return tuple(segments)

    def tokenize(
        self,
        sequence,
        flatten=True,
        branching_threshold=None,
        cohesion_threshold=None,
        **kwargs,
    ):
        if branching_threshold is not None:
            self.branching_threshold = branching_threshold
        if cohesion_threshold is not None:
            self.cohesion_threshold = cohesion_threshold
        segments = []
        words = self.pre_tokenize(sequence)
        for word in words:
            segments.append(self.tokenize_word(word))
        if flatten:
            segments = [seg for word in segments for seg in word]
        return segments

    def tokenize_texts(self, texts):
        return [self.tokenize(text) for text in texts]

    def naive_segment(self, text):
        words = []
        # TODO: need to implement with a change in the tokenizer
        # _start, _pos = 0, 0
        # # iterate over the text until we reach the end
        # while _pos < len(text):
        #     _sentencepiece = text[_pos : _pos + self.max_piece_length]
        #     # print(_start, _pos, _sentencepiece)
        #     if len(_sentencepiece) < 1:
        #         break
        #     results = self.find_local_entropy(_sentencepiece)
        #     _, entropies, _ = zip(*results)

        #     if entropies[0] == 0:
        #         if _pos == len(text) - 1:
        #             words.append(text[_start : _pos + 1])
        #             _start = _pos + 1
        #             break
        #         _pos += 1
        #     else:
        #         if _pos > _start:
        #             words.append(text[_start:_pos])
        #             _start = _pos
        #             _pos += 1
        #         if len(entropies) > 1:
        #             _pos += 1
        #             for i in range(1, len(entropies)):
        #                 if entropies[i] == 0:
        #                     words.append(text[_start : _start + i])
        #                     # print(_start, words)
        #                     _start += i
        #                     _pos = _start
        #                     break
        #                 elif _pos == len(text) - 1:
        #                     words.append(text[_start : _pos + 1])
        #                 _pos += 1
        #         else:
        #             if _pos == len(text) - 1:
        #                 words.append(text[_start : _pos + 1])
        #             _pos += 1

        return words

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        trainer: BranchingEntropyTrainer = None,
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer.normalizer = self.normalizer
        trainer.pre_tokenizer = self.pre_tokenizer
        vocab = trainer.train(iterator, length=length)
        self.initialize_vocab(vocab)

    @classmethod
    def from_file(cls, vocab, **kwargs):
        """
        Instantiate a BranchingEntropy model from the given files.

        This method is roughly equivalent to doing::

           vocab = BranchingEntropy.read_file(vocab_filename)
           be = BranchingEntropy(vocab)

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            :class:`~tokenizers.models.BranchingEntropy`: An instance of BranchingEntropy loaded from these files
        """
        vocab = cls.read_file(vocab)
        return cls(vocab, **kwargs)

    @staticmethod
    def read_file(vocab):
        """
        Read a :obj:`vocab.json` file

        This method provides a way to read and parse the content of these files,
        returning the relevant data structures. If you want to instantiate some BPE models
        from memory, this method gives you the expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            A :obj:`Tuple` with the vocab and the merges:
                The vocabulary and merges loaded into memory
        """
        vocab = json.load(vocab)
        return vocab

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
        vocab_filename = os.path.join(folder, "vocab.json.zst")
        if not os.path.exists(folder):
            os.makedirs(folder)
        json.dump(self.vocab, vocab_filename)
        return [vocab_filename]

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
        tokens_ = [token for (token, _) in words_with_offsets]
        # if token value is replacement, concatenate with the next token
        tokens = []
        for i, token in enumerate(tokens_):
            if token == self.whitespace_token:
                continue
            if i > 0 and tokens_[i - 1] == self.whitespace_token:
                token = tokens_[i - 1] + token
            tokens.append(token)
        return tokens
