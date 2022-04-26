from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf


class BaseSentimentAnalyser:
    __metaclass__ = ABCMeta
    """
    A base class for sentiment analysis.
    """

    EPSILON = 1e-6

    def __init__(self, preprocessor=None, lexicon=None, **kwargs):
        self._predict = kwargs.get("predict") or {}
        self._predict_feature = self._predict.get("feature")
        self._predict_feature = self._predict_feature or "polarity"
        self._eval = kwargs.get("eval") or {}
        self._features = kwargs.get("features") or {}
        self.verbose = kwargs.get("verbose", False)

        self._tokenizer = preprocessor["tokenizer"]
        if eKonf.is_instantiatable(self._tokenizer):
            if self.verbose:
                print(f"[ekorpkit]: instantiating {self._tokenizer['_target_']}...")
            self._tokenizer = eKonf.instantiate(self._tokenizer)

        self._lexicon = lexicon
        if eKonf.is_instantiatable(self._lexicon):
            if self.verbose:
                print(f"[ekorpkit]: instantiating {self._lexicon['_target_']}...")
            self._lexicon = eKonf.instantiate(self._lexicon)

    def tokenize(self, text):
        """
        :type text: str
        :returns: list
        """
        return self._tokenizer.tokenize(text)

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

    def predict(self, text, feature="polarity"):
        """Get score for a list of terms.

        :returns: dict
        """
        tokens = self.tokenize(text)
        lexicon_features = self._lexicon.analyze(tokens)

        score = self._get_score(tokens, lexicon_features, feature=feature)

        return self._assign_class(score, feature=feature)
