import math
from scipy.special import digamma
from pydantic import BaseModel
from .trie import Trie


class Scores(BaseModel):
    entropy: float = 0
    cohesion: float = 0
    freq: int = 0
    word: str = None


class ScoreDiffs(BaseModel):
    f_ent: float = 0.0
    b_ent: float = 0.0
    coh: float = 0.0
    # avg_coh: float = 0.0


class ScoreResult(BaseModel):
    char: str = None
    L_scores: Scores = Scores()
    # avg_coh: float = 0.0
    diffs: ScoreDiffs = ScoreDiffs()
    R_scores: Scores = Scores()


def entropy(trie: Trie, word, whitespace_token="▁"):
    if word == whitespace_token:
        return None, None
    leafs = trie.get_leafs(word)
    val = trie.get_value(word)
    logsum = digamma(sum(leafs) + val)
    entropy = 0
    for freq in leafs:
        logprob = digamma(freq) - logsum
        entropy += math.exp(logprob) * logprob
    return -1 * entropy, val


def cohesion(trie: Trie, word, whitespace_token="▁"):
    if word[0] == whitespace_token and len(word) > 1:
        word = word[1:]
    word_len = len(word)
    if (not word) or (word_len <= 1):
        return None
    if word[-1] == whitespace_token and len(word) > 1:
        word = word[:-1]
    word_len = len(word)
    if (not word) or (word_len <= 1):
        return None
    val = trie.get_value(word)
    val0 = trie.get_value_pos(word, 0)
    # cohesion = np.power((val / val0), 1 / (word_len - 1)) if val0 and val else 0
    # using digamma
    cohesion = (
        math.exp((digamma(val) - digamma(val0)) / (word_len - 1))
        if val0 and val
        else None
    )
    return cohesion


def scores(trie: Trie, word, whitespace_token="▁") -> Scores:
    scores_ = Scores()
    if not word:
        return scores_

    scores_.word = word
    scores_.entropy, scores_.freq = entropy(trie, word, whitespace_token)
    scores_.cohesion = cohesion(trie, word, whitespace_token)

    return scores_
