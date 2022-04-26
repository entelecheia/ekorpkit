import pandas as pd
from .base import BaseSentimentAnalyser


class HIV4SA(BaseSentimentAnalyser):
    """
    A class for sentiment analysis using the HIV4 lexicon.
    """

    def __init__(self, preprocessor=None, lexicon=None, **kwargs):
        super().__init__(preprocessor=preprocessor, lexicon=lexicon, **kwargs)

    def _get_score(self, tokens, lexicon_features, feature="polarity"):
        """Get score for features.

        :returns: int
        """
        lxfeat = pd.DataFrame.from_dict(lexicon_features, orient="index")
        score = {}
        if feature == "polarity":
            lxfeat["pos"] = lxfeat.apply(
                lambda x: 1 * x["count"] if x["Positiv"] else 0, axis=1
            )
            lxfeat["neg"] = lxfeat.apply(
                lambda x: 1 * x["count"] if x["Negativ"] else 0, axis=1
            )
            lxfeat_agg = lxfeat.agg({"pos": "sum", "neg": "sum"})
            polarity = (lxfeat_agg["pos"] - lxfeat_agg["neg"]) / (
                lxfeat_agg["pos"] + lxfeat_agg["neg"] + self.EPSILON
            )
            subjectivity = (lxfeat_agg["pos"] + lxfeat_agg["neg"]) / (
                len(tokens) + self.EPSILON
            )
            score["polarity"] = polarity
            score["subjectivity"] = subjectivity

        return score

    def _assign_class(self, score, feature="polarity"):
        """Assign class to a score.

        :returns: str
        """
        labels = self._features.get(feature).get("labels")
        if labels:
            score["label"] = ""
            for label, thresh in labels.items():
                if isinstance(thresh, str):
                    thresh = eval(thresh)
                if score[feature] >= thresh[0] and score[feature] <= thresh[1]:
                    score["label"] = label
        return score


class LMSA(BaseSentimentAnalyser):
    """
    A class for sentiment analysis using the LM lexicon.
    """

    def __init__(self, preprocessor=None, lexicon=None, **kwargs):
        super().__init__(preprocessor=preprocessor, lexicon=lexicon, **kwargs)

    def _get_score(self, tokens, lexicon_features, feature="polarity"):
        """Get score for features.

        :returns: int
        """
        lxfeat = pd.DataFrame.from_dict(lexicon_features, orient="index")
        score = {}
        if feature == "polarity":
            lxfeat["pos"] = lxfeat.apply(
                lambda x: 1 * x["count"] if x["Positive"] > 0 else 0, axis=1
            )
            lxfeat["neg"] = lxfeat.apply(
                lambda x: 1 * x["count"] if x["Negative"] > 0 else 0, axis=1
            )
            lxfeat_agg = lxfeat.agg({"pos": "sum", "neg": "sum"})
            polarity = (lxfeat_agg["pos"] - lxfeat_agg["neg"]) / (
                lxfeat_agg["pos"] + lxfeat_agg["neg"] + self.EPSILON
            )
            subjectivity = (lxfeat_agg["pos"] + lxfeat_agg["neg"]) / (
                len(tokens) + self.EPSILON
            )
            score["polarity"] = polarity
            score["subjectivity"] = subjectivity

        return score

    def _assign_class(self, score, feature="polarity"):
        """Assign class to a score.

        :returns: str
        """
        labels = self._features.get(feature).get("labels")
        if labels:
            score["label"] = ""
            for label, thresh in labels.items():
                if isinstance(thresh, str):
                    thresh = eval(thresh)
                if score[feature] >= thresh[0] and score[feature] <= thresh[1]:
                    score["label"] = label
        return score
