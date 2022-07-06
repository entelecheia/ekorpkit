import os
import math
import logging
from collections import defaultdict, namedtuple
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

NEG_INF = float("-inf")


MutualInformation = namedtuple("score", "words length frequency score")
CohesionScore = namedtuple(
    "score", "words length frequency cohesion_forward cohesion_backward score"
)
BranchingEntropy = namedtuple(
    "score", "words length frequency leftside_entropy rightside_entropy score"
)


def cohesion_score(
    ngrams,
    total_words=None,
    min_count=30,
    expansion_method="max",
    delimiter="_",
    **kwargs,
):
    def _max(cf, cb):
        return max(cf, cb)

    def _average(cf, cb):
        return (cf + cb) / 2

    candidates = {}
    for ngram, freq in tqdm(ngrams.items()):
        n = len(ngram)
        if n <= 1 or freq < min_count:
            continue

        cohesion_forward = pow(freq / ngrams.get(tuple(ngram[:1]), 0), 1 / (n - 1))
        cohesion_backward = pow(freq / ngrams.get(tuple(ngram[-1:]), 0), 1 / (n - 1))

        if expansion_method == "max":
            score = _max(cohesion_forward, cohesion_backward)
        elif expansion_method == "average":
            score = _average(cohesion_forward, cohesion_backward)
        elif expansion_method == "backward":
            score = cohesion_backward
        else:
            score = cohesion_forward

        candidates[ngram] = CohesionScore(
            delimiter.join(ngram),
            n,
            freq,
            cohesion_forward,
            cohesion_backward,
            score,
        )

    return candidates


def branching_entropy(
    ngrams,
    total_words=None,
    min_count=10,
    delimiter="_",
    **kwargs,
):
    def entropy(dic):
        if not dic:
            return 0.0
        sum_ = sum(dic.values())
        entropy = 0
        for freq in dic.values():
            prob = freq / sum_
            entropy += prob * math.log(prob)
        return -1 * entropy

    def parse_left(extension):
        return extension[:-1]

    def parse_right(extension):
        return extension[1:]

    def sort_by_length(counter, min_count):
        sorted_by_length = defaultdict(lambda: [])
        for ngram, freq in counter.items():
            if freq < min_count or len(ngram) <= 2:
                continue
            sorted_by_length[len(ngram)].append(ngram)
        return sorted_by_length

    def get_entropy_table(parse, sorted_by_length):
        be = {}
        for n, ngram_list in tqdm(sorted_by_length.items()):
            extensions = defaultdict(lambda: [])
            for ngram in ngram_list:
                extensions[parse(ngram)].append(ngram)
            for ngram, extension_ngrams in extensions.items():
                extension_frequency = {
                    ext: ngrams.get(ext, 0) for ext in extension_ngrams if ext in ngrams
                }
                be[ngram] = entropy(extension_frequency)
        return be

    sorted_by_length = sort_by_length(ngrams, min_count)
    be_l = get_entropy_table(parse_right, sorted_by_length)
    be_r = get_entropy_table(parse_left, sorted_by_length)
    bes = {ngram: (bel, be_r.get(ngram, 0)) for ngram, bel in be_l.items()}
    for ngram, ber in be_r.items():
        if not (ngram in be_l):
            bes[ngram] = (0, ber)

    candidates = {}
    for ngram, (bel, ber) in bes.items():
        score = bel * ber
        candidates[ngram] = BranchingEntropy(
            delimiter.join(ngram),
            len(ngram),
            ngrams.get(ngram, 0),
            bel,
            ber,
            score,
        )

    return candidates


def mutual_information(
    ngrams,
    total_words=None,
    delta=0.0,
    expansion_method="average",
    normalize=False,
    delimiter="_",
    **kwargs,
):
    def _average(scores):
        return 0 if not scores else sum(scores) / len(scores)

    def _max(scores):
        return 0 if not scores else max(scores)

    def _top3_average(scores):
        return (
            0
            if not scores
            else sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))
        )

    # max_n = max([len(ngram) for ngram in ngrams.keys()])
    candidates = {}
    total_words = float(total_words)
    num_ngrams = len(ngrams)
    for ngram, ab in tqdm(ngrams.items()):
        if (len(ngram) == 1) or (ab <= delta):
            continue
        score_candidates = {}
        for i in range(1, len(ngram)):
            a = ngrams.get(tuple(ngram[:i]), 0)
            b = ngrams.get(tuple(ngram[i:]), 0)
            if (a == 0) or (b == 0):
                continue
            if normalize:
                pa = a / total_words
                pb = b / total_words
                pab = ab / total_words
                score = math.log(pab / (pa * pb)) / -math.log(pab)
            else:
                score = (ab - delta) / float(a * b) * num_ngrams
            score_candidates[i] = score

        if not score_candidates:
            continue

        if expansion_method == "max":
            score = _max(score_candidates.values())
        elif expansion_method == "top3_average":
            score = _top3_average(score_candidates.values())
        else:
            score = sum(score_candidates.values()) / len(score_candidates)

        candidates[ngram] = MutualInformation(
            delimiter.join(ngram), num_ngrams, ab, score
        )
    return candidates


def get_available_memory():
    """It returns remained memory as percentage"""
    import psutil

    mem = psutil.virtual_memory()
    return 100 * mem.available / (mem.total)


def get_process_memory():
    """It returns the memory usage of current process"""
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def prune_vocab(vocab, prune_min_count):
    """Remove all entries from the `vocab` dictionary with count smaller than `prune_min_count`.
    Modifies `vocab` in place, returns the sum of all counts that were pruned.
    Parameters
    ----------
    vocab : dict
        Input dictionary.
    prune_min_count : int
        Frequency threshold for tokens in `vocab`.
    Returns
    -------
    result : int
        Sum of all counts that were pruned.
    """
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if vocab[w] <= prune_min_count:
            result += vocab[w]
            del vocab[w]
    log.info(
        "pruned out %i tokens with count <=%i (before %i, after %i)",
        old_len - len(vocab),
        prune_min_count,
        old_len,
        len(vocab),
    )
    return result
