import os
import logging
import pandas as pd
from collections import namedtuple
from tqdm import tqdm
from .score import get_process_memory, prune_vocab, NEG_INF
from ekorpkit.io.file import save_dataframe, load_dataframe
from ekorpkit import eKonf

# from inspect import getfullargspec as getargspec


log = logging.getLogger(__name__)


def _exist_ordered_overlap(list_s, list_l):
    if len(list_s) > len(list_l):
        return _exist_ordered_overlap(list_l, list_s)
    matched_first_item = False
    for s_i, s in enumerate(list_s):
        if not list_l:
            break
        matched = -1
        for l_i, l in enumerate(list_l):
            if s == l:
                matched = l_i
                if l_i == 0 or s_i == 0:
                    matched_first_item = True
                break
        list_l = list_l[matched + 1 :]
    if matched == -1:
        return False
    if not matched_first_item and len(list_l) > 0:
        return False
    return True


def _remove_overlapping_ngrams_by_score(ngram_pos_scores):
    """Remove overlapping ngrams by score"""
    result = []
    unigram_pos_scores = []
    for nram_pos_score in ngram_pos_scores:
        ngram, pos, score = nram_pos_score
        if len(ngram) == 1:
            unigram_pos_scores.append(nram_pos_score)
            continue
        exist_overlap = False
        for _ngram_pos_score in ngram_pos_scores:
            _ngram, _pos, _score = _ngram_pos_score
            if _ngram == ngram or len(_ngram) == 1:
                continue
            if min(_pos) > max(pos) or max(_pos) < min(pos):
                continue
            if _exist_ordered_overlap(pos, _pos):
                if score < _score:
                    exist_overlap = True
                    break
        if not exist_overlap:
            result.append(nram_pos_score)

    for uniram_pos_score in unigram_pos_scores:
        unigram, pos, score = uniram_pos_score
        exist_overlap = False
        for _ngram_pos_score in result:
            _ngram, _pos, _score = _ngram_pos_score
            if _ngram == unigram or len(_ngram) == 1:
                continue
            if min(_pos) > max(pos) or max(_pos) < min(pos):
                continue
            if pos[0] in _pos:
                exist_overlap = True
                break
        if not exist_overlap:
            result.append(uniram_pos_score)

    result = sorted(result, key=lambda x: x[1][0], reverse=False)
    return result


def _prepare_ngram_tuples(
    words, max_n=5, max_window=None, max_skip=None, include_positions=False
):
    num_words = len(words)
    # ngrams = [(word,) for word in words]
    ngrams = []
    for ngram_index_set in _get_ngram_indices(num_words, max_n, max_window, max_skip):
        if include_positions:
            ngrams.append(
                (
                    tuple(words[i] for i in ngram_index_set),
                    tuple(i for i in ngram_index_set),
                )
            )
        else:
            ngrams.append(tuple([words[i] for i in ngram_index_set]))

    return ngrams


def _get_ngram_indices(num_words, max_n, max_window=None, max_skip=None):
    from itertools import combinations

    if max_window is None:
        max_window = max_n
    if max_skip is None:
        max_skip = 0
    if max_skip > max_n:
        max_skip = max_n

    word_positions = list(range(num_words))
    indices = set()
    for window in range(1, min(max_window, num_words) + 1):
        for starting in range(num_words - window + 1):
            position_set = word_positions[starting : starting + window]
            for n in range(max(1, window - max_skip), min(window, max_n) + 1):
                indices.update(set(combinations(position_set, n)))

    return sorted(indices)


def _get_surface(
    ngram_tuple,
    surface_delim="",
    postag_delim="/",
    strip_pos=True,
    postag_length=None,
):
    surfaces = [
        _get_word_surface(token, postag_delim, strip_pos, postag_length)
        for token in ngram_tuple
    ]
    return surface_delim.join(surfaces)


def _get_word_surface(token, postag_delim="/", strip_pos=True, postag_length=None):
    if not isinstance(postag_delim, str):
        return token
    token_pos = token.split(postag_delim)
    if strip_pos:
        return token_pos[0]
    return (
        token
        if len(token_pos) == 1
        else token_pos[0]
        + postag_delim
        + (token_pos[1][:postag_length] if postag_length else token_pos[1])
    )


class Ngrams:
    def __init__(
        self,
        **args,
    ):
        args = eKonf.to_config(args)

        self.name = args.name
        self._data = args.data

        self._ngram = args.ngram
        self._candidates = args.candidates
        self._ngramize = args.ngramize
        self._postag = args.postag
        self._analyze = args.analyze
        self.score_function = eKonf.partial(args.score_function)
        self.force_train = args.force_train

        assert type(self._ngram.max_n) == int

        if self._ngram.max_n <= 0:
            self._ngram.max_n = 4
        if self._ngram.max_window < self._ngram.max_n:
            self._ngram.max_window = self._ngram.max_n
        if not self._ngram.max_skip:
            self._ngram.max_skip = self._ngram.max_window - self._ngram.max_n

        if self._candidates.min_count <= 0:
            self._candidates.min_count = 10
        if self.tokenize is None:
            self.tokenize = lambda x: x.split()
        if self._candidates.max_candidates <= 0:
            self._candidates.max_candidates = 40000000
        self.progress_per = args.progress_per
        self.verbose = args.verbose

        self.output_dir = args.output_dir
        self.output_file = os.path.join(self.output_dir, args.output_file)
        os.makedirs(self.output_dir, exist_ok=True)

        self._tokenizer = args.preprocessor.tokenizer
        if eKonf.is_instantiatable(self._tokenizer):
            log.info(f"instantiating {self._tokenizer['_target_']}...")
            self._tokenizer = eKonf.instantiate(self._tokenizer)
        self._postag.stop_tags = eKonf.ensure_list(self._postag.stop_tags)

        self.sentences = []
        self.ngrams = {}
        self.candidates = {}
        self.total_words = 0

        self.initialize()

    def initialize(self):
        self.load_candidates()
        if not self.candidates or self.force_train:
            self.train()
            self.save_candidates()

    def tokenize(self, text):
        tokens = self._tokenizer.tokenize(text)
        return self._tokenizer.extract(
            tokens,
            strip_pos=self._postag.strip_pos,
            stop_postags=self._postag.stop_tags,
        )

    def load_candidates(self):
        """Load a previously saved model"""
        if os.path.exists(self.output_file):
            df = load_dataframe(self.output_file, verbose=self.verbose)
            Ngram = namedtuple("ngram", df.columns)
            _cands = df.to_dict(orient="records")
            self.candidates = {
                tuple(cand["words"].split(self._ngram.delimiter)): Ngram(**cand)
                for cand in _cands
            }
            log.info(f"loaded {len(self.candidates)} candidates")
        else:
            log.info("no candidates to load")

    def save_candidates(self):
        """Save the candidates to a file"""
        if len(self.candidates) > 0:
            df = pd.DataFrame(self.candidates.values())
            save_dataframe(df, self.output_file, verbose=self.verbose)
        else:
            log.info("no candidates to save")

    def __str__(self):
        return (
            "%s<%i candidates, min_ngram_count=%s, threshold=%s, max_candidates=%s>"
            % (
                self.__class__.__name__,
                len(self.candidates),
                self._candidates.min_count,
                self._candidates.threshold,
                self._candidates.max_candidates,
            )
        )

    def _load_data(self):
        """Load data"""
        if self._data is None:
            log.warning("No data config found")
            return
        data = eKonf.instantiate(self._data)
        docs = data.data[data.COLUMN.TEXT]
        self.sentences = []
        for doc in docs:
            self.sentences.extend(doc.split("\n"))

    def train(self):
        """Train the model"""
        if not self.sentences:
            self._load_data()
        self._learn_ngrams()
        if self.score_function is not None and callable(self.score_function):
            self.candidates = self.score_function(
                self.ngrams, total_words=self.total_words
            )

    def _learn_ngrams(self):

        self.ngrams = {}
        sentence_no, total_words = -1, 0
        for sentence_no, sentence in tqdm(enumerate(self.sentences)):
            words = self.tokenize(sentence)
            total_words += len(words)

            ngrams = self.prepare_ngram_tuples(
                words, max_n=self._ngram.max_n, max_window=self._ngram.max_window
            )
            for ngram in ngrams:
                self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1

            if (
                (sentence_no > 0)
                and (self._candidates.num_sents_per_pruning > 0)
                and (sentence_no % self._candidates.num_sents_per_pruning == 0)
            ):
                prune_vocab(self.ngrams, self._candidates.prune_min_ngram_count)
            if len(self.ngrams) > self._candidates.max_candidates:
                prune_vocab(self.ngrams, self._candidates.prune_min_ngram_count)
                self._candidates.prune_min_ngram_count += 1

        if self.verbose:
            print("learning ngrams was done. memory= %.3f Gb" % get_process_memory())

        self.ngrams = {
            ngram: freq
            for ngram, freq in self.ngrams.items()
            if freq >= self._candidates.min_count
        }
        self.total_words = total_words

    def find_ngrams(
        self,
        sentences,
        exclude_overlaps=True,
        threshold=None,
        surface_delim=";",
        postag_delim="/",
        strip_pos=True,
        postag_length=None,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        result = {}
        for sentence in tqdm(sentences):
            for ngram, _, score in self.analyze_sentence(
                sentence, exclude_overlaps=exclude_overlaps, threshold=threshold
            ):
                ngram = self.get_surface(
                    ngram,
                    surface_delim=surface_delim,
                    strip_pos=strip_pos,
                    postag_delim=postag_delim,
                    postag_length=postag_length,
                )
                if score > NEG_INF:
                    result[ngram] = score
        return result

    def __getitem__(self, sentence):
        return self.ngramize_sentence(sentence)

    def ngramize_sentence(self, sentence):
        """
        Return a list of ngrams of the sentence
        """
        exclude_overlaps = self._analyze.exclude_overlaps
        threshold = self._analyze.threshold
        strip_pos = self._ngramize.strip_pos
        suface_delim = self._ngramize.delimiter
        postag_delim = self._postag.delimiter
        postag_length = self._postag.max_len
        return [
            self.get_surface(
                ngram,
                surface_delim=suface_delim,
                postag_delim=postag_delim,
                strip_pos=strip_pos,
                postag_length=postag_length,
            )
            for ngram, _, _ in self.analyze_sentence(
                sentence, exclude_overlaps, threshold
            )
        ]

    def analyze_sentence(
        self,
        sentence,
        exclude_overlaps=True,
        threshold=None,
    ):
        """Analyze a sentence, concatenating any detected ngrams into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.
        """

        words = self.tokenize(sentence)
        ngram_with_positions = self.prepare_ngram_tuples(
            words,
            max_n=self._ngram.max_n,
            max_window=self._ngram.max_window,
            include_positions=True,
        )

        tokens = []
        for ngram_pos in ngram_with_positions:
            ngram, pos = ngram_pos
            score = self.ngram_score(ngram, threshold=threshold)
            if score is not None:
                tokens.append((ngram, pos, score))
        if exclude_overlaps:
            tokens = self.remove_overlapping_ngrams_by_score(tokens)
        return tokens

    def ngram_score(self, ngram, threshold=None, unigram_score=NEG_INF):
        """Score a ngram"""

        if len(ngram) == 1:
            return unigram_score

        if ngram in self.candidates:
            score = self.candidates[ngram].score
            if threshold is None or score >= threshold:
                return score
        return None

    def export_ngrams(self, threshold=None):
        """Extract all found ngrams.
        Returns
        ------
        dict(str, float)
            Mapping between phrases and their scores.
        """
        result = {}
        for ngram, score in self.candidates.items():
            if len(ngram) < 2:
                continue  # no phrases here
            if threshold is None or score.score > threshold:
                result[ngram] = score
        # sort by score
        result = sorted(result.items(), key=lambda x: x[1].score, reverse=True)
        return result

    @staticmethod
    def prepare_ngram_tuples(
        words, max_n=5, max_window=None, max_skip=None, include_positions=False
    ):
        """Prepare ngram tuples from a list of words."""

        return _prepare_ngram_tuples(
            words,
            max_n=max_n,
            max_window=max_window,
            max_skip=max_skip,
            include_positions=include_positions,
        )

    @staticmethod
    def remove_overlapping_ngrams_by_score(ngram_pos_scores):
        """Remove overlapping ngrams by score.

        Parameters
        ----------
        ngram_pos_scores : list of (str, int, float)
            List of ngrams and their positions in the sentence.

        Returns
        -------
        list of (str, int, float)
            List of ngrams and their positions in the sentence.
        """
        return _remove_overlapping_ngrams_by_score(ngram_pos_scores)

    @staticmethod
    def get_surface(
        ngram_tuple,
        surface_delim="",
        postag_delim="/",
        strip_pos=True,
        postag_length=None,
    ):
        """Get the surface form of a ngram tuple."""

        return _get_surface(
            ngram_tuple,
            surface_delim=surface_delim,
            postag_delim=postag_delim,
            strip_pos=strip_pos,
            postag_length=postag_length,
        )
