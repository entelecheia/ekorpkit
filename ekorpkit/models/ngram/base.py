import logging
from enum import Enum

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


class _KEEP(str, Enum):
    """Split keys in configs used by Dataset."""

    HIGHEST_SCORE = "highest_score"
    HIGHEST_ABS_SCORE = "highest_abs_score"
    SHORTEST_WITH_SCORE = "shortest_with_score"
    LONGEST_WITH_SCORE = "longest_with_score"


def _remove_overlaps(ngram_pos_scores, keep: _KEEP = _KEEP.HIGHEST_SCORE):
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
                if keep == _KEEP.HIGHEST_SCORE and score and _score:
                    if score < _score:
                        exist_overlap = True
                        break
                    elif score == _score:
                        if len(ngram) > len(_ngram):
                            exist_overlap = True
                            break
                        elif len(ngram) == len(_ngram):
                            if ngram < _ngram:
                                exist_overlap = True
                                break
                elif keep == _KEEP.HIGHEST_ABS_SCORE and score and _score:
                    if abs(score) < abs(_score):
                        exist_overlap = True
                        break
                    elif abs(score) == abs(_score):
                        if len(ngram) > len(_ngram):
                            exist_overlap = True
                            break
                        elif len(ngram) == len(_ngram):
                            if ngram < _ngram:
                                exist_overlap = True
                                break
                elif keep == _KEEP.SHORTEST_WITH_SCORE:
                    if len(ngram) > len(_ngram):
                        exist_overlap = True
                        break
                    elif len(ngram) == len(_ngram):
                        if score and _score:
                            if abs(score) < abs(_score):
                                exist_overlap = True
                                break
                        elif score:
                            exist_overlap = True
                            break
                        elif _score:
                            exist_overlap = True
                            break
                else:
                    if len(ngram) < len(_ngram):
                        exist_overlap = True
                        break
                    elif len(ngram) == len(_ngram):
                        if ngram < _ngram:
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
    words,
    max_n=5,
    max_window=None,
    max_skip=None,
    postag_rules=[],
    postag_delim=";",
    include_positions=False,
):
    num_words = len(words)
    ngrams = []
    for ngram_index_set in _get_ngram_indices(num_words, max_n, max_window, max_skip):
        ngram = tuple(words[i] for i in ngram_index_set)
        position = tuple(i for i in ngram_index_set)
        if _match_any_rules(ngram, postag_rules, postag_delim) or len(ngram) == 1:
            if include_positions:
                ngrams.append((ngram, position))
            else:
                ngrams.append(ngram)

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


def _get_ngram_tuple(ngram_str, ngram_delim=";"):
    ngram = ngram_str.split(ngram_delim)
    return tuple(ngram)


def _get_ngram_str(
    ngram_tuple,
    ngram_delim=";",
    postag_delim="/",
    strip_pos=True,
    postag_length=None,
    lowercase=True,
):
    surfaces = [
        _get_word(token, postag_delim, strip_pos, postag_length, lowercase)
        for token in ngram_tuple
    ]
    return ngram_delim.join(surfaces)


def _get_word(
    token, postag_delim="/", strip_pos=True, postag_length=None, lowercase=True
):
    if not isinstance(postag_delim, str):
        return token.lower() if lowercase else token
    token_pos = token.split(postag_delim)
    if strip_pos:
        return token_pos[0].lower() if lowercase else token_pos[0]
    return (
        (token.lower() if lowercase else token)
        if len(token_pos) == 1
        else (token_pos[0].lower() if lowercase else token_pos[0])
        + postag_delim
        + (token_pos[1][:postag_length] if postag_length else token_pos[1]).upper()
    )


def _match(ngram, postag_rule, postag_delim="/"):
    if isinstance(postag_rule, str):
        if not postag_rule.startswith(postag_delim):
            postag_rule = postag_delim + postag_rule
        for token in ngram:
            if postag_rule in token:
                return True
        return False
    if isinstance(postag_rule, list):
        postag_rule = tuple(postag_rule)
        if len(ngram) != len(postag_rule):
            return False
        for token, tag in zip(ngram, postag_rule):
            if not tag.startswith(postag_delim):
                tag = postag_delim + tag
            if tag not in token:
                return False
        return True


def _match_any_rules(
    ngram,
    postag_rules,
    postag_delim="/",
):
    if not postag_rules:
        return True
    for rule in postag_rules:
        if _match(ngram, rule, postag_delim):
            return True
    return False
