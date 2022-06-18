import logging
import pandas as pd
import numpy as np
from enum import Enum
from ekorpkit import eKonf

log = logging.getLogger(__name__)


class _Keys(str, Enum):
    DEFAULT = eKonf.Keys.DEFAULT.value
    FEATURE = "feature"
    FEATURES = "features"
    LEXICON_FEATURES = "lexicon_features"
    POLARITY = "polarity"
    NUM_TOKENS = "num_tokens"
    COUNT = "count"
    EVAL = "eval"
    AGG = "agg"
    SCORE = "score"
    LABELS = "labels"


class SentimentAnalyser:
    """
    A base class for sentiment analysis.
    """

    EPSILON = 1e-6

    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self._predict_ = args[eKonf.Keys.PREDICT]

        self._predict_.features = (
            self._predict_.get(_Keys.FEATURES) or _Keys.POLARITY.value
        )
        self._predict_.features = eKonf.ensure_list(self._predict_.features)
        self._eval_ = args.get(eKonf.Keys.EVAL)
        self._features = args.get(_Keys.FEATURES)
        self.verbose = args.get("verbose", False)

        self._ngram = eKonf.instantiate(args.model.ngram)

        self._sentence_separator = eKonf.Defaults.SENT_SEP.value

    def analyze(self, text, **kwargs):
        """
        :type text: str
        :returns: list
        """
        return self._ngram.find_features(text, **kwargs)

    def tokenize(self, text):
        """Tokenize text.

        :returns: list
        """
        return self._ngram.tokenize(text)

    def predict(self, text, features=None):
        """Get score for a list of terms.

        :returns: dict
        """
        features = features or self._predict_.features
        if isinstance(features, str):
            features = [features]
        tokens = self._ngram.tokenize(text)
        _lex_feats = self._predict_.get(_Keys.LEXICON_FEATURES)
        lexicon_features = self._ngram.find_features(tokens, features=_lex_feats)
        scores = {}
        for feather in features:
            score = self._get_score(tokens, lexicon_features, feature=feather)
            score = self._assign_class(score, feature=feather)
            scores.update(score)

        return scores

    def predict_article(self, article, features=["polarity"]):
        if article is None:
            return None

        for sent in article.split(self._sentence_separator):
            sent = sent.strip()
            if sent:
                yield self.predict(sent, features=features)

    def _get_score(self, tokens, lexicon_features, feature="polarity"):
        """Get score for features.

        :returns: int
        """
        _feature = self._features[feature]
        _default_feature = self._features[_Keys.DEFAULT]
        _lex_feats = _feature.get(_Keys.LEXICON_FEATURES)
        lxfeat = pd.DataFrame.from_dict(lexicon_features, orient="index")

        score = {}
        if lxfeat.empty:
            return score
        num_tokens = len(tokens)
        score[_Keys.NUM_TOKENS.value] = num_tokens

        eps = self.EPSILON
        if _Keys.EVAL in _feature:
            _evals = _feature.get(_Keys.EVAL)
            _count = _feature.get(_Keys.COUNT)
            _agg = eKonf.to_dict(_feature.get(_Keys.AGG))
            _score = _feature.get(_Keys.SCORE)
        else:
            _evals = _default_feature.get(_Keys.EVAL)
            _count = _default_feature.get(_Keys.COUNT)
            _agg = eKonf.to_dict(_default_feature.get(_Keys.AGG))
            _score = _default_feature.get(_Keys.SCORE)
            lxfeat.rename(columns={_lex_feats: _Keys.FEATURE.value}, inplace=True)
        if self.verbose:
            log.info("Evaluating %s", feature)
            print(lxfeat)
        for _name, _expr in _evals.items():
            lxfeat[_name] = np.where(lxfeat.eval(_expr), lxfeat[_count], 0)
        lxfeat_agg = lxfeat.agg(_agg)
        lxfeat_agg = pd.DataFrame(lxfeat_agg).T
        _feat_score = lxfeat_agg.eval(_score)
        if not _feat_score.empty:
            score[feature] = _feat_score[0]

        return score

    def _assign_class(self, score, feature="polarity"):
        """Assign class to a score.

        :returns: str
        """
        _label_key = feature + "_label"
        _labels = self._features[feature].get(_Keys.LABELS)
        if _labels:
            score[_label_key] = ""
            for _label, _thresh in _labels.items():
                if isinstance(_thresh, str):
                    _thresh = eval(_thresh)
                if score[feature] >= _thresh[0] and score[feature] <= _thresh[1]:
                    score[_label_key] = _label
        return score
