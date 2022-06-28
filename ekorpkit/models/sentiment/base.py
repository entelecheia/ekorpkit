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
    LEXICON_FEATURES = "lexicon_features"
    POLARITY = "polarity"
    NUM_TOKENS = "num_tokens"
    NUM_EXAMPLES = "num_examples"
    COUNT = "count"
    EVAL = "eval"
    AGG = "agg"
    SCORE = "score"
    LABELS = "labels"
    AGG_METHOD = "agg_method"


class _AggMethods(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    DIFFUSION = "diffusion"
    RATIO = "ratio"
    STD = "std"


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
        self._eval_ = args.get(eKonf.Keys.EVAL)
        self._features = args.get(_Keys.FEATURES)
        self._article_features = args.get(_Keys.ARTICLE_FEATURES)
        self.num_workers = args.num_workers
        self.use_batcher = args.use_batcher
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

    def predict(self, data, _predict_={}):
        if _predict_:
            self._predict_ = _predict_

        input_col = self._predict_[eKonf.Keys.INPUT]
        _method_ = self._predict_[eKonf.Keys.METHOD]
        _meth_name_ = _method_.get(eKonf.Keys.METHOD_NAME)
        log.info(f"Predicting sentiments of the column [{input_col}] using {_meth_name_}")
        _fn = lambda doc: getattr(self, _meth_name_)(doc)
        with elapsed_timer(format_time=True) as elapsed:
            predictions = eKonf.apply(
                _fn,
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
        scores = {}

        features = self._predict_.features or features
        features = eKonf.ensure_list(features)
        _lex_feats = self._predict_.get(_Keys.LEXICON_FEATURES)
        min_examples = self._predict_.min_tokens or min_examples

        tokens = self._ngram.tokenize(text)
        num_examples = len(tokens)
        lexicon_features = self._ngram.find_features(tokens, features=_lex_feats)
        for feather in features:
            if num_examples < min_examples:
                score = {feather: np.nan}
            else:
                score = self._get_score(
                    lexicon_features, feature=feather, num_examples=num_examples
                )
            score = self._assign_class(score, feature=feather)
            scores.update(score)

        return scores

    def _get_score(
        self, lexicon_features, feature="polarity", num_examples=None, _features=None
    ):
        """Get score for features.

        :returns: dict
        """
        _features = _features or self._features
        _feature = _features[feature]
        _default_feature = self._features[_Keys.DEFAULT]
        _lex_feats = _feature.get(_Keys.LEXICON_FEATURES)
        _num_examples = _feature.get(_Keys.NUM_EXAMPLES) or _Keys.NUM_TOKENS.value

        lxfeat = pd.DataFrame.from_dict(lexicon_features, orient="index")

        score = {_num_examples: num_examples, feature: np.nan}
        if lxfeat.empty:
            return score
        if num_examples is None:
            num_examples = lxfeat.shape[0]
        score[_num_examples] = num_examples

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
        if self.verbose > 5:
            log.info("Evaluating %s", feature)
            print(lxfeat)
        for _name, _expr in _evals.items():
            lxfeat[_name] = np.where(
                lxfeat.eval(_expr), lxfeat[_count] if _count else 1, 0
            )
        lxfeat_agg = lxfeat.agg(_agg)
        lxfeat_agg = pd.DataFrame(lxfeat_agg).T
        _feat_score = lxfeat_agg.eval(_score)
        if not _feat_score.empty:
            score[feature] = _feat_score[0]
        else:
            score[feature] = None

        return score

    def _assign_class(self, score, feature="polarity"):
        """Assign class to a score.

        :returns: dict
        """
        _label_key = feature + "_label"
        _labels = self._features[feature].get(_Keys.LABELS)
        if _labels:
            score[_label_key] = None
            if feature not in score:
                if self.verbose > 5:
                    log.info(f"No score for {feature}")
                return score
            for _label, _thresh in _labels.items():
                if isinstance(_thresh, str):
                    _thresh = eval(_thresh)
                if score[feature] >= _thresh[0] and score[feature] <= _thresh[1]:
                    score[_label_key] = _label
        return score

    def predict_article(
        self,
        article,
        features=["polarity"],
        agg_method="mean",
        min_examples=2,
        **kwargs,
    ):
        scores = {}

        agg_method = self._predict_.agg_method or agg_method
        features = self._predict_.features or features
        features = eKonf.ensure_list(features)
        min_examples = self._predict_.min_sentences or min_examples

        article_scores = {}
        if isinstance(article, str):
            sents = article.split(self._sentence_separator)
            nun_examples = len(sents)
            for sent_no, sent in enumerate(sents):
                sent = sent.strip()
                if sent:
                    _scores = self.predict_sentence(sent, features=features)
                    if _scores:
                        article_scores[sent_no] = _scores
        for feather in features:
            if len(article_scores) < min_examples:
                score = {feather: np.nan}
            else:
                score = self._get_aggregate_scores(
                    article_scores,
                    feature=feather,
                    agg_method=agg_method,
                    article_features=self._article_features,
                )
            score = self._assign_class(score, feature=feather)
            scores.update(score)

        return scores

    def _get_aggregate_scores(
        self,
        scores,
        feature="polarity",
        agg_method="mean",
        article_features=None,
        num_examples=None,
    ):
        """Get aggreagate score for features.

        :returns: dict
        """
        article_features = article_features or self._article_features
        _agg_method = article_features.agg_method[agg_method]
        _num_examples = _agg_method.get(_Keys.NUM_EXAMPLES) or _Keys.NUM_TOKENS.value

        scores_df = pd.DataFrame.from_dict(scores, orient="index")
        scores_df.dropna(subset=[feature], inplace=True)

        score = {_num_examples: np.nan, feature: np.nan}
        if scores_df.empty:
            return score
        num_examples = scores_df.shape[0]
        score[_num_examples] = num_examples

        eps = self.EPSILON
        _evals = _agg_method.get(_Keys.EVAL)
        _count = _agg_method.get(_Keys.COUNT)
        _agg = eKonf.to_dict(_agg_method.get(_Keys.AGG))
        _score = _agg_method.get(_Keys.SCORE) or _Keys.FEATURE.value
        scores_df.rename(columns={feature: _Keys.FEATURE.value}, inplace=True)
        if self.verbose > 5:
            log.info("Evaluating %s", feature)
            print(scores_df)
        if _evals is not None:
            for _name, _expr in _evals.items():
                scores_df[_name] = np.where(
                    scores_df.eval(_expr), scores_df[_count] if _count else 1, 0
                )
        lxfeat_agg = scores_df.agg(_agg)
        lxfeat_agg = pd.DataFrame(lxfeat_agg).T
        _feat_score = lxfeat_agg.eval(_score)
        if not _feat_score.empty:
            score[feature] = _feat_score[0]
        else:
            score[feature] = None

        return score
