import os
import logging
import pandas as pd
from collections import namedtuple
from tqdm import tqdm
from .score import get_process_memory, prune_vocab, NEG_INF
from ekorpkit.io.file import save_dataframe, load_dataframe
from ekorpkit import eKonf

# from inspect import getfullargspec as getargspec


log = logging.getLogger(__name__)


class Ngrams:
    def __init__(
        self,
        **args,
    ):
        args = eKonf.to_config(args)

        self.name = args.name
        self._data = args.data
        self.tokenize = args.tokenize

        self._ngram = args.ngram
        self._candidates = args.candidates
        self._surface = args.surface
        self._postag = args.postag
        self.score_function = eKonf.partial(args.score_function)

        assert type(self._ngram.max_n) == int

        if self._ngram.max_n <= 0:
            self._ngram.max_n = 4
        if self._ngram.max_window < self._ngram.max_n:
            self._ngram.max_window = self._ngram.max_n
        if not self._ngram.max_skip:
            self._ngram.max_skip = self._ngram.max_window - self._ngram.max_n

        if self._candidates.min_count <= 0:
            self._candidates.min_count = 10
        if self.tokenize is None:
            self.tokenize = lambda x: x.split()
        if self._candidates.max_candidates <= 0:
            self._candidates.max_candidates = 40000000
        self.progress_per = args.progress_per
        self.verbose = args.verbose

        self.output_dir = args.output_dir
        self.output_file = os.path.join(self.output_dir, args.output_file)
        os.makedirs(self.output_dir, exist_ok=True)

        self.sentences = []
        self.ngrams = {}
        self.candidates = {}
        self.total_words = 0

    def load_candidates(self):
        """Load a previously saved model"""
        if os.path.exists(self.output_file):
            df = load_dataframe(self.output_file, verbose=self.verbose)
            Ngram = namedtuple("ngram", df.columns)
            _cands = df.to_dict(orient="records")
            self.candidates = {
                tuple(cand["words"].split(self._ngram.delimiter)): Ngram(*cand)
                for cand in _cands
            }
            log.info(f"loaded {len(self.candidates)} candidates")
        else:
            log.info("no candidates to load")

    def save_candidates(self):
        """Save the candidates to a file"""
        if len(self.candidates) > 0:
            df = pd.DataFrame(self.candidates.values())
            save_dataframe(df, self.output_file, verbose=self.verbose)
        else:
            log.info("no candidates to save")

    def __str__(self):
        return (
            "%s<%i candidates, min_ngram_count=%s, threshold=%s, max_candidates=%s>"
            % (
                self.__class__.__name__,
                len(self.candidates),
                self._candidates.min_count,
                self._candidates.threshold,
                self._candidates.max_candidates,
            )
        )

    def _load_data(self):
        """Load data"""
        if self._data is None:
            log.warning("No data config found")
            return
        data = eKonf.instantiate(self._data)
        docs = data.data[data.COLUMN.TEXT]
        self.sentences = []
        for doc in docs:
            self.sentences.extend(doc.split("\n"))

    def train(self):
        """Train the model"""
        if not self.sentences:
            self._load_data()
        self._learn_ngrams()
        if self.score_function is not None and callable(self.score_function):
            self.candidates = self.score_function(self.ngrams)

    def _learn_ngrams(self):

        self.ngrams = {}
        sentence_no, total_words = -1, 0
        for sentence_no, sentence in tqdm(enumerate(self.sentences)):
            words = self.tokenize(sentence)
            total_words += len(words)

            ngrams = [
                tuple(words[pos : pos + n])
                for n in range(1, self._ngram.max_n + 1)
                for pos in range(0, len(words) - n + 1)
            ]
            ngrams = _prepare_ngram_tuples(
                words, max_n=self._ngram.max_n, max_window=self._ngram.max_window
            )
            for ngram in ngrams:
                self.ngrams[ngram] = self.ngrams.get(ngram, 0) + 1

            if (
                (sentence_no > 0)
                and (self._candidates.num_sents_per_pruning > 0)
                and (sentence_no % self._candidates.num_sents_per_pruning == 0)
            ):
                prune_vocab(self.ngrams, self._candidates.prune_min_ngram_count)
            if len(self.ngrams) > self._candidates.max_candidates:
                prune_vocab(self.ngrams, self._candidates.prune_min_ngram_count)
                self._candidates.prune_min_ngram_count += 1

            # if self.verbose and sentence_no % self.progress_per == 0:
            #     args = (
            #         len(self.ngrams),
            #         sentence_no + 1,
            #         len(self.sentences),
            #         get_process_memory(),
            #     )
            #     log.info(
            #         "learning ngrams ... # candidates= %d, (%d in %d) memory= %.3f Gb"
            #         % args
            #     )
        if self.verbose:
            print("learning ngrams was done. memory= %.3f Gb" % get_process_memory())

        self.ngrams = {
            ngram: freq
            for ngram, freq in self.ngrams.items()
            if freq >= self._candidates.min_count
        }
        self.total_words = total_words

    def find_ngrams(self, sentences):
        result = {}
        for sentence in sentences:
            for phrase, score in self.analyze_sentence(sentence):
                if score is not None:
                    result[phrase] = score
        return result

    def __getitem__(self, sentence):
        return [token for token, _ in self.analyze_sentence(sentence)]

    def concat_ngrams(self, sentence):
        return " ".join([token for token, _ in self.analyze_sentence(sentence)])

    def analyze_sentence(self, sentence, ngram_delim=None, strip_pos=None):
        """Analyze a sentence, concatenating any detected ngrams into a single token.

        Parameters
        ----------
        sentence : iterable of str
            Token sequence representing the sentence to be analyzed.

        Yields
        ------
            Iterate through the input sentence tokens and yield 2-tuples of:
            - ``(concatenated_ngram_tokens, score)`` for token sequences that form a phrase.
            - ``(word, None)`` if the token is not a part of a phrase.

        """

        start_token, in_between = None, []
        for word in sentence:
            if word not in self.connector_words:
                # The current word is a normal token, not a connector word, which means it's a potential
                # beginning (or end) of a phrase.
                if start_token:
                    # We're inside a potential phrase, of which this word is the end.
                    phrase, score = self.score_candidate(start_token, word, in_between)
                    if score is not None:
                        # Phrase detected!
                        yield phrase, score
                        start_token, in_between = None, []
                    else:
                        # Not a phrase after all. Dissolve the candidate's constituent tokens as individual words.
                        yield start_token, None
                        for w in in_between:
                            yield w, None
                        start_token, in_between = (
                            word,
                            [],
                        )  # new potential phrase starts here
                else:
                    # Not inside a phrase yet; start a new phrase candidate here.
                    start_token, in_between = word, []
            else:  # We're a connector word.
                if start_token:
                    # We're inside a potential phrase: add the connector word and keep growing the phrase.
                    in_between.append(word)
                else:
                    # Not inside a phrase: emit the connector word and move on.
                    yield word, None
        # Emit any non-phrase tokens at the end.
        if start_token:
            yield start_token, None
            for w in in_between:
                yield w, None

    def score_candidate(self, word_a, word_b, in_between):
        # Micro optimization: check for quick early-out conditions, before the actual scoring.
        word_a_cnt = self.candidates.get(word_a, 0)
        if word_a_cnt <= 0:
            return None, None

        word_b_cnt = self.candidates.get(word_b, 0)
        if word_b_cnt <= 0:
            return None, None

        phrase = self._ngram.delimiter.join([word_a] + in_between + [word_b])
        # XXX: Why do we care about *all* phrase tokens? Why not just score the start+end bigram?
        phrase_cnt = self.candidates.get(phrase, 0)
        if phrase_cnt <= 0:
            return None, None

        score = self.scoring(
            worda_count=word_a_cnt,
            wordb_count=word_b_cnt,
            bigram_count=phrase_cnt,
            len_vocab=len(self.candidates),
            min_count=self._candidates.min_count,
            corpus_word_count=self.corpus_word_count,
        )
        if score <= self._candidates.threshold:
            return None, None

        return phrase, score

    def export_ngrams(self, threshold=None):
        """Extract all found ngrams.
        Returns
        ------
        dict(str, float)
            Mapping between phrases and their scores.
        """
        result, source_vocab = {}, self.candidates
        for ngram in source_vocab:
            if len(ngram) < 2:
                continue  # no phrases here
            ngram, score = self.score_candidate(
                ngram[0], ngram[-1], ngram[1:-1]
            )
            if score is not None:
                result[ngram] = score
        return result


def _is_ordered_subset(subset, superset, distance_tolerance=None):
    if len(subset) > len(superset):
        return False
    if distance_tolerance and len(superset) - len(subset) > distance_tolerance:
        return False
    if len(subset) == len(superset):
        return subset == superset
    for sub_item in subset:
        if not superset:
            return False
        matched = -1
        for i, super_item in enumerate(superset):
            if sub_item == super_item:
                matched = i
                break
        if matched == -1:
            return False
        superset = superset[matched + 1 :]
    return True


def _prepare_ngram_tuples(words, max_n=5, max_window=None, max_skip=None):
    num_words = len(words)
    ngrams = [(word,) for word in words]
    ngrams += [
        tuple([words[i] for i in ngram_index_set])
        for ngram_index_set in _get_ngram_indices(
            num_words, max_n, max_window, max_skip
        )
    ]

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
    for window in range(2, min(max_window, num_words) + 1):
        for starting in range(num_words - window + 1):
            position_set = word_positions[starting : starting + window]
            for n in range(max(2, window - max_skip), min(window, max_n) + 1):
                indices.update(set(combinations(position_set, n)))

    return sorted(indices)


def _get_surface(
    ngram,
    ngram_delim=";",
    surface_delim="",
    postag_delim="/",
    strip_pos=True,
    postag_length=None,
):
    tokens = ngram.split(ngram_delim)
    surfaces = [
        _get_word_surface(token, postag_delim, strip_pos, postag_length)
        for token in tokens
    ]
    return surface_delim.join(surfaces)


def _get_word_surface(token, postag_delim="/", strip_pos=True, postag_length=None):
    if not isinstance(postag_delim, str):
        return token
    token_pos = token.split(postag_delim)
    if strip_pos:
        return token_pos[0]
    return (
        token
        if len(token_pos) == 1
        else token_pos[0]
        + postag_delim
        + (token_pos[1][:postag_length] if postag_length else token_pos[1])
    )
