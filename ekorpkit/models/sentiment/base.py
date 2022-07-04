from itertools import groupby
import logging
import pandas as pd
import numpy as np
from enum import Enum
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer


log = logging.getLogger(__name__)


class _Keys(str, Enum):
    DEFAULT = eKonf.Keys.DEFAULT.value
    FEATURE = "feature"
    FEATURES = "features"
    ARTICLE_FEATURES = "article_features"
    AGGREGATE_SCORES = "aggregate_scores"
    LEXICON_FEATURES = "lexicon_features"
    POLARITY = "polarity"
    NUM_TOKENS = "num_tokens"
    NUM_EXAMPLES = "num_examples"
    COUNT = "count"
    CONDITIONS = "conditions"
    APPLY = "apply"
    AGGS = "aggs"
    EVALS = "evals"
    SCORE = "score"
    SCORES = "scores"
    LABELS = "labels"
    LABEL_BY = "label_by"
    AGG_METHOD = "agg_method"


class BaseSentimentAnalyser:
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
        self._eval_ = args.get(eKonf.Keys.EVAL)
        self._features = args.get(_Keys.FEATURES)
        self._article_features = args.get(_Keys.ARTICLE_FEATURES)
        self._aggregate_scores = args.get(_Keys.AGGREGATE_SCORES)
        self.num_workers = args.num_workers
        self.use_batcher = args.use_batcher
        self.verbose = args.get("verbose", False)

        self._ngram = None
        self._sentence_separator = eKonf.Defaults.SENT_SEP.value

    def analyze(self, text, **kwargs):
        """
        :type text: str
        :returns: list
        """
        raise NotImplementedError

    def tokenize(self, text):
        """Tokenize text.

        :returns: list
        """
        raise NotImplementedError

    def predict(self, data, _predict_={}):
        if _predict_:
            self._predict_ = _predict_

        input_col = self._predict_[eKonf.Keys.INPUT]
        log.info(
            f"Predicting sentiments of the column [{input_col}] using {self._predict_}"
        )
        with elapsed_timer(format_time=True) as elapsed:
            predictions = eKonf.apply(
                self.predict_sentence,
                data[input_col],
                description=f"Predicting [{input_col}]",
                verbose=self.verbose,
                use_batcher=self.use_batcher,
                num_workers=self.num_workers,
            )
            pred_df = pd.DataFrame(predictions.tolist(), index=predictions.index)
            data = data.join(pred_df)
            log.info(" >> elapsed time to predict: {}".format(elapsed()))

        return data

    def predict_sentence(self, text, features="polarity", min_examples=5):
        """Get score for a list of terms.

        :returns: dict
        """
        raise NotImplementedError

    def predict_article(
        self,
        article,
        features=["polarity"],
        min_examples=2,
        **kwargs,
    ):
        """Predict sentiment for an article.

        :returns: dict
        """
        raise NotImplementedError

    def aggregate_scores(
        self,
        data,
        groupby,
        feature="polarity",
        min_examples=2,
        _method_=None,
    ):
        """Get aggreagate score for features.

        :returns: dataframe
        """
        _aggregate_scores = self._aggregate_scores
        _method_ = _method_ or feature
        _agg_method = (
            _aggregate_scores.get(_method_) or _aggregate_scores[_Keys.DEFAULT]
        )
        _num_examples = _Keys.NUM_EXAMPLES.value

        groupby = eKonf.ensure_list(groupby)
        if feature in data.columns:
            data = data.copy().dropna(subset=[feature])
        else:
            data = data.copy()

        eps = self.EPSILON
        _conditions = _agg_method.get(_Keys.CONDITIONS)
        _apply = _agg_method.get(_Keys.APPLY)
        _aggs = eKonf.to_dict(_agg_method.get(_Keys.AGGS))
        _evals = _agg_method.get(_Keys.EVALS)
        _scores = _agg_method.get(_Keys.SCORES)
        # _label_by = _agg_method.get(_Keys.LABEL_BY)
        _scores = {
            _name: _name.replace(_Keys.FEATURE.value, feature) for _name in _scores
        }
        _labels = self._article_features.get(feature, {}).get(_Keys.LABELS)

        data.rename(columns={feature: _Keys.FEATURE.value}, inplace=True)
        if self.verbose > 5:
            log.info("Evaluating %s", feature)
            print(data.head())
        if _apply is not None:
            for _name, _expr in _apply.items():
                data[_name] = data.apply(lambda x: eval(_expr), axis=1)
        if _conditions is not None:
            for _name, _expr in _conditions.items():
                data[_name] = np.where(data.eval(_expr), 1, 0)

        _aggs = {k: v for k, v in _aggs.items() if k in data.columns}
        agg_scores = data.groupby(groupby).agg(_aggs)
        agg_scores.columns = agg_scores.columns.to_flat_index().str.join("_")
        if _evals is not None:
            for _name, _expr in _evals.items():
                agg_scores[_name] = agg_scores.eval(_expr)
        agg_scores.rename(columns=_scores, inplace=True)
        agg_scores = agg_scores[_scores.values()]
        agg_scores = agg_scores.reset_index().copy()

        def _label_score(score):
            for _label, _thresh in _labels.items():
                if isinstance(_thresh, str):
                    _thresh = eval(_thresh)
                if score >= _thresh[0] and score <= _thresh[1]:
                    return _label
            return np.nan

        for f in _scores.values():
            if f.startswith(feature):
                agg_scores[f] = np.where(
                    agg_scores[_num_examples] > min_examples, agg_scores[f], np.nan
                )
                _label_key = f + "_label"
                if _labels:
                    agg_scores[_label_key] = agg_scores[f].apply(_label_score)

        return agg_scores
