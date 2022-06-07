import logging
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf

log = logging.getLogger(__name__)


class BaseSentimentAnalyser:
    __metaclass__ = ABCMeta
    """
    A base class for sentiment analysis.
    """

    EPSILON = 1e-6

    def __init__(self, model={}, **kwargs):
        self._predict = kwargs.get("predict") or {}
        self._predict_features = self._predict.get("features")
        self._predict_features = self._predict_features or "polarity"
        self._eval = kwargs.get("eval") or {}
        self._features = kwargs.get("features") or {}
        self.verbose = kwargs.get("verbose", False)

        self._ngram = model.get("ngram")
        self._ngram = eKonf.instantiate(self._ngram)

    def analyze(self, text, **kwargs):
        """
        :type text: str
        :returns: list
        """
        return self._ngram.find_features(text, **kwargs)

    @abstractmethod
    def _get_score(self, tokens, lexicon_features, feature="polarity"):
        """Get score for features.

        :returns: int
        """
        raise NotImplementedError("Must override segment")

    @abstractmethod
    def _assign_class(self, score, feature="polarity"):
        """Assign class to a score.

        :returns: str
        """
        raise NotImplementedError("Must override segment")

    def tokenize(self, text):
        """Tokenize text.

        :returns: list
        """
        return self._ngram.tokenize(text)

    def predict(self, text, features=None):
        """Get score for a list of terms.

        :returns: dict
        """
        features = features or self._predict_features
        if isinstance(features, str):
            features = [features]
        tokens = self._ngram.tokenize(text)
        lex_feats = self._predict.get("lexicon_features")
        lexicon_features = self._ngram.find_features(tokens, features=lex_feats)
        scores = {}
        for feather in features:
            score = self._get_score(tokens, lexicon_features, feature=feather)
            score = self._assign_class(score, feature=feather)
            scores.update(score)

        return scores
