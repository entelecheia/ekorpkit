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

    def __init__(self, preprocessor=None, lexicon=None, **kwargs):
        self._predict = kwargs.get("predict") or {}
        self._predict_features = self._predict.get("features")
        self._predict_features = self._predict_features or "polarity"
        self._eval = kwargs.get("eval") or {}
        self._features = kwargs.get("features") or {}
        self.verbose = kwargs.get("verbose", False)

        self._tokenizer = preprocessor["tokenizer"]
        if eKonf.is_instantiatable(self._tokenizer):
            if self.verbose:
                log.info(f"instantiating {self._tokenizer['_target_']}...")
            self._tokenizer = eKonf.instantiate(self._tokenizer)

        self._lexicon = lexicon
        if eKonf.is_instantiatable(self._lexicon):
            if self.verbose:
                log.info(f"instantiating {self._lexicon['_target_']}...")
            self._lexicon = eKonf.instantiate(self._lexicon)

    def tokenize(self, text):
        """
        :type text: str
        :returns: list
        """
        return self._tokenizer.tokenize(text)

    def analyze(self, text, **kwargs):
        """
        :type text: str
        :returns: list
        """
        return self._lexicon.analyze(self.tokenize(text), **kwargs)

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

    def predict(self, text, features=None):
        """Get score for a list of terms.

        :returns: dict
        """
        features = features or self._predict_features
        if isinstance(features, str):
            features = [features]
        tokens = self.tokenize(text)
        lex_feats = self._predict.get("lexicon_features")
        lexicon_features = self._lexicon.analyze(tokens, features=lex_feats)
        scores = {}
        for feather in features:
            score = self._get_score(tokens, lexicon_features, feature=feather)
            score = self._assign_class(score, feature=feather)
            scores.update(score)

        return scores
