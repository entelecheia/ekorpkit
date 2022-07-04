import os
import logging
from collections import namedtuple
from .score import NEG_INF
from .base import (
    _match_any_rules,
    _prepare_ngram_tuples,
    _remove_overlaps,
    _get_ngram_tuple,
    _get_ngram_str,
    _KEEP,
)
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class Ngrams:
    def __init__(
        self,
        **args,
    ):
        args = eKonf.to_config(args)

        self.name = args.name
        self._data = args.data

        self._ngram = args.ngram
        self._ngramize = args.ngramize
        self._postag = args.postag
        self._scores = args.scores
        self._info = args.info
        self.auto = args.auto
        self.force = args.force

        assert type(self._ngram.max_n) == int

        if self._ngram.max_n <= 0:
            self._ngram.max_n = 4
        if self._ngram.max_window < self._ngram.max_n:
            self._ngram.max_window = self._ngram.max_n
        if not self._ngram.max_skip:
            self._ngram.max_skip = self._ngram.max_window - self._ngram.max_n
        self.postag_rules = eKonf.ensure_list(self._ngram.postag_rules)

        if self.tokenize is None:
            self.tokenize = lambda x: x.split()
        self.progress_per = args.progress_per
        self.verbose = args.verbose

        self.score_path = args.scores.score_path

        self._tokenizer = args.preprocessor.tokenizer
        if eKonf.is_instantiatable(self._tokenizer):
            self._tokenizer = eKonf.instantiate(self._tokenizer)
        self._postag.stop_tags = eKonf.ensure_list(self._postag.stop_tags)

        self._sentences = []
        self.candidates = {}
        self._surface_to_tuples = {}

        if self.auto.load:
            eKonf.methods(args._method_, self)

    def __len__(self):
        return len(self.candidates)

    def __contains__(self, ngram):
        return ngram in self.candidates

    def __iter__(self):
        return iter(self.candidates)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname}\n----------\n"
        s += f"name: {self.name}\n"
        s += f"fullname: {self._info.fullname}\n"
        s += f"source: {self._info.source}\n"
        s += f"lang: {self._info.lang}\n"
        s += f"features: {self._score.features}\n"
        s += f"score path: {self.score_path}\n"
        s += f"no. of candidates: {len(self.candidates)}\n"
        return s

    def initialize(self):
        self.load_candidates()
        if not self.candidates or self.force.train:
            self.train()
            self.save_candidates()

    def tokenize(self, text):
        tokens = self._tokenizer.tokenize(text)
        return self._tokenizer.extract(
            tokens,
            postags=self._postag.use_tags,
            strip_pos=self._postag.strip_pos,
            stop_postags=self._postag.stop_tags,
            postag_length=self._postag.max_len,
        )

    def load_candidates(self):
        """Load a previously saved model"""
        if os.path.exists(self.score_path):
            _words = self._scores.columns.words
            _features = self._scores.features
            _lowercase = self._scores.lowercase

            df = eKonf.load_data(self.score_path, verbose=self.verbose)
            df = df.dropna(subset=[_words])
            columns = [_words] + _features
            Score = namedtuple("score", columns)
            _cands = df[columns].to_dict(orient="records")
            self.candidates = {}
            self._surface_to_tuples = {}
            for cand in _cands:
                ngram_tuple = self.to_ngram_tuple(
                    cand[_words],
                    ngram_delim=self._scores.ngram_delim,
                )
                ngram_str = self.to_ngram_str(
                    ngram_tuple,
                    ngram_delim=self._ngram.delimiter,
                    postag_delim=self._postag.delimiter,
                    strip_pos=self._postag.strip_pos,
                    postag_length=self._postag.max_len,
                    lowercase=_lowercase,
                )
                surface_str = self.to_ngram_str(
                    ngram_tuple,
                    ngram_delim="",
                    postag_delim=self._postag.delimiter,
                    strip_pos=True,
                    lowercase=_lowercase,
                )
                cand[_words] = ngram_str
                ngram_tuple = self.to_ngram_tuple(
                    ngram_str, ngram_delim=self._ngram.delimiter
                )
                self.candidates[ngram_tuple] = Score(**cand)
                self._surface_to_tuples[surface_str] = ngram_tuple

            log.info(f"loaded {len(self.candidates)} candidates")
        else:
            log.info("no candidates to load")

    def save_candidates(self):
        raise NotImplementedError

    @property
    def sentences(self):
        if len(self._sentences) == 0:
            self.load_data()
        return self._sentences

    def load_data(self):
        """Load data"""
        if self._data is None:
            log.warning("No data config found")
            return
        data = eKonf.instantiate(self._data)
        docs = data.data[data.COLUMN.TEXT]
        self._sentences = []
        for doc in docs:
            self._sentences.extend(doc.split("\n"))

    def train(self):
        """Train the model"""
        raise NotImplementedError

    def score_ngrams(self):
        raise NotImplementedError

    def learn_ngrams(self):
        raise NotImplementedError

    def find_features(self, words, features=None, **kwargs):
        """Find features for a given ngram tuple"""
        _features = features or self._scores.features
        return self.find_ngrams(
            words,
            features=_features,
            exclude_overlaps=self._ngramize.exclude_overlaps,
            overlaps_to_keep=self._ngramize.overlaps_to_keep,
            threshold=self._ngramize.threshold,
            ignore_scores=self._ngramize.ignore_scores,
            apply_postag_rules=self._ngramize.apply_postag_rules,
            use_surfaces_to_score=self._ngramize.use_surfaces_to_score,
            strip_pos=self._ngramize.strip_pos,
            surface_delim=self._ngramize.delimiter,
            postag_delim=self._postag.delimiter,
            postag_length=self._postag.max_len,
            **kwargs,
        )

    def find_ngrams(
        self,
        words,
        exclude_overlaps=None,
        overlaps_to_keep: _KEEP = None,
        threshold=None,
        ignore_scores=None,
        apply_postag_rules=None,
        use_surfaces_to_score=None,
        strip_pos=None,
        surface_delim=None,
        postag_delim=None,
        postag_length=None,
        features=None,
    ):
        exclude_overlaps = (
            exclude_overlaps
            if exclude_overlaps is not None
            else self._ngramize.exclude_overlaps
        )
        overlaps_to_keep = overlaps_to_keep or _KEEP(self._ngramize.overlaps_to_keep)
        threshold = threshold or self._ngramize.threshold
        ignore_scores = (
            ignore_scores if ignore_scores is not None else self._ngramize.ignore_scores
        )
        apply_postag_rules = (
            apply_postag_rules
            if apply_postag_rules is not None
            else self._ngramize.apply_postag_rules
        )
        use_surfaces_to_score = (
            use_surfaces_to_score
            if use_surfaces_to_score is not None
            else self._ngramize.use_surfaces_to_score
        )
        surface_delim = surface_delim or self._ngramize.delimiter
        postag_delim = postag_delim or self._postag.delimiter
        strip_pos = strip_pos if strip_pos is not None else self._ngramize.strip_pos
        postag_length = postag_length or self._postag.max_len

        postag_rules = self.postag_rules if apply_postag_rules else []

        _count = self._scores.columns.count
        _score = self._scores.columns.score
        features = features or [_score]

        results = {}
        for ngram, _, score in self.analyze_sentence(
            words,
            exclude_overlaps=exclude_overlaps,
            overlaps_to_keep=overlaps_to_keep,
            threshold=threshold,
            ignore_scores=ignore_scores,
            use_surfaces_to_score=use_surfaces_to_score,
            postag_rules=postag_rules,
        ):
            ngram_str = self.to_ngram_str(
                ngram,
                ngram_delim=surface_delim,
                strip_pos=strip_pos,
                postag_delim=postag_delim,
                postag_length=postag_length,
            )
            if ignore_scores or score > NEG_INF:
                if ngram_str in results:
                    results[ngram_str][_count] += 1
                else:
                    feats = self.get_features(ngram, features)
                    if feats:
                        feats[_count] = 1
                        results[ngram_str] = feats
        if self.verbose:
            log.info(f"found {len(results)} ngrams")
        return results

    def get_features(self, ngram, features):
        """Get features"""
        _features = {}
        score = self.candidates.get(ngram)
        if score is None or features is None:
            return _features
        for _f in features:
            if _f in score._fields:
                _features[_f] = getattr(score, _f)
        return _features

    def __getitem__(self, words):
        return self.ngramize_sentence(words)

    def ngramize_sentence(
        self,
        words,
        exclude_overlaps=None,
        overlaps_to_keep: _KEEP = None,
        threshold=None,
        ignore_scores=None,
        apply_postag_rules=None,
        use_surfaces_to_score=None,
        strip_pos=None,
        surface_delim=None,
        postag_delim=None,
        postag_length=None,
        **kwargs,
    ):
        """
        Return a list of ngrams of the sentence
        """
        exclude_overlaps = (
            exclude_overlaps
            if exclude_overlaps is not None
            else self._ngramize.exclude_overlaps
        )
        overlaps_to_keep = overlaps_to_keep or _KEEP(self._ngramize.overlaps_to_keep)
        threshold = threshold or self._ngramize.threshold
        ignore_scores = (
            ignore_scores if ignore_scores is not None else self._ngramize.ignore_scores
        )
        apply_postag_rules = (
            apply_postag_rules
            if apply_postag_rules is not None
            else self._ngramize.apply_postag_rules
        )
        use_surfaces_to_score = (
            use_surfaces_to_score
            if use_surfaces_to_score is not None
            else self._ngramize.use_surfaces_to_score
        )
        surface_delim = surface_delim or self._ngramize.delimiter
        postag_delim = postag_delim or self._postag.delimiter
        strip_pos = strip_pos if strip_pos is not None else self._ngramize.strip_pos
        postag_length = postag_length or self._postag.max_len

        postag_rules = self.postag_rules if apply_postag_rules else []

        return [
            self.to_ngram_str(
                ngram,
                ngram_delim=surface_delim,
                postag_delim=postag_delim,
                strip_pos=strip_pos,
                postag_length=postag_length,
            )
            for ngram, _, _ in self.analyze_sentence(
                words,
                exclude_overlaps=exclude_overlaps,
                overlaps_to_keep=overlaps_to_keep,
                threshold=threshold,
                ignore_scores=ignore_scores,
                use_surfaces_to_score=use_surfaces_to_score,
                postag_rules=postag_rules,
            )
        ]

    def analyze_sentence(
        self,
        words,
        exclude_overlaps=True,
        overlaps_to_keep: _KEEP = _KEEP.HIGHEST_SCORE,
        threshold=None,
        ignore_scores=False,
        use_surfaces_to_score=False,
        postag_rules=[],
        **kwargs,
    ):
        """Analyze a sentence, concatenating any detected ngrams into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.
        """

        words = self.tokenize(words) if isinstance(words, str) else words
        ngram_with_positions = self.prepare_ngram_tuples(
            words,
            max_n=self._ngram.max_n,
            max_window=self._ngram.max_window,
            max_skip=self._ngram.max_skip,
            postag_rules=postag_rules,
            postag_delim=self._postag.delimiter,
            include_positions=True,
        )

        ngrams = []
        for ngram_pos in ngram_with_positions:
            ngram, pos = ngram_pos
            score = self.ngram_score(
                ngram, threshold=threshold, use_surfaces_to_score=use_surfaces_to_score
            )
            if score is not None or ignore_scores:
                ngrams.append((ngram, pos, score))
        if exclude_overlaps:
            ngrams = self.remove_overlaps(ngrams, keep=overlaps_to_keep)
        return ngrams

    def ngram_score(
        self,
        ngram,
        threshold=None,
        unigram_score=NEG_INF,
        use_surfaces_to_score=False,
    ):
        """Score a ngram"""

        _score = self._scores.columns.score

        if use_surfaces_to_score:
            surface_str = self.to_ngram_str(
                ngram,
                ngram_delim="",
                postag_delim=self._postag.delimiter,
                strip_pos=True,
            )
            if surface_str in self._surface_to_tuples:
                ngram = self._surface_to_tuples[surface_str]
                score = getattr(self.candidates[ngram], _score)
                if threshold is None or score >= threshold:
                    return score

        if ngram in self.candidates:
            score = getattr(self.candidates[ngram], _score)
            if threshold is None or score >= threshold:
                return score
        if len(ngram) == 1:
            return unigram_score
        return None

    def export_ngrams(
        self, threshold=None, postag_rules=None
    ):
        """Extract all found ngrams.
        Returns
        ------
        dict(str, float)
            Mapping between phrases and their scores.
        """
        postag_rules = postag_rules or self.postag_rules

        result = {}
        _score = self._scores.columns.score
        for ngram, score in self.candidates.items():
            if len(ngram) < 2:
                continue  # no phrases here
            if postag_rules and not self.match_any_rules(
                ngram, postag_rules, self._postag.delimiter
            ):
                continue
            if threshold is None or getattr(score, _score) > threshold:
                result[ngram] = score
        # sort by score
        result = sorted(
            result.items(), key=lambda x: getattr(x[1], _score), reverse=True
        )
        return result

    @staticmethod
    def match_any_rules(ngram, postag_rules, postag_delimiter):
        return _match_any_rules(ngram, postag_rules, postag_delimiter)

    @staticmethod
    def prepare_ngram_tuples(
        words,
        max_n=5,
        max_window=None,
        max_skip=None,
        postag_rules=[],
        postag_delim=";",
        include_positions=False,
    ):
        """Prepare ngram tuples from a list of words."""

        return _prepare_ngram_tuples(
            words,
            max_n=max_n,
            max_window=max_window,
            max_skip=max_skip,
            postag_rules=postag_rules,
            postag_delim=postag_delim,
            include_positions=include_positions,
        )

    @staticmethod
    def remove_overlaps(ngram_pos_scores, keep="highest_score"):
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
        return _remove_overlaps(ngram_pos_scores, keep=keep)

    @staticmethod
    def to_ngram_tuple(ngram_str, ngram_delim=";"):
        """Get a ngram tuple from a string."""
        return _get_ngram_tuple(ngram_str, ngram_delim)

    @staticmethod
    def to_ngram_str(
        ngram,
        ngram_delim="",
        postag_delim="/",
        strip_pos=True,
        postag_length=None,
        lowercase=True,
    ):
        """Get the surface form of a ngram tuple."""

        return _get_ngram_str(
            ngram,
            ngram_delim=ngram_delim,
            postag_delim=postag_delim,
            strip_pos=strip_pos,
            postag_length=postag_length,
            lowercase=lowercase,
        )
