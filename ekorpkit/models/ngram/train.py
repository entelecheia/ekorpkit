import logging
import pandas as pd
from tqdm.auto import tqdm
from ekorpkit import eKonf
from .ngram import Ngrams
from .score import get_process_memory, prune_vocab


log = logging.getLogger(__name__)


class NgramTrainer(Ngrams):
    def __init__(
        self,
        **args,
    ):
        args = eKonf.to_config(args)
        self._candidates = args.candidates
        if self._candidates.min_count <= 0:
            self._candidates.min_count = 10
        if self._candidates.max_candidates <= 0:
            self._candidates.max_candidates = 40000000

        self.score_function = eKonf.partial(args.score_function)
        self._ngrams = {}
        self._total_words = 0

        super().__init__(**args)

    def save_candidates(self):
        """Save the candidates to a file"""
        if len(self.candidates) > 0:
            df = pd.DataFrame(self.candidates.values())
            eKonf.save_data(df, self.score_path, verbose=self.verbose)
        else:
            log.info("no candidates to save")

    def train(self):
        """Train the model"""
        if not self._sentences:
            self.load_data()
        self.learn_ngrams()
        self.score_ngrams()

    def score_ngrams(self):
        if self.score_function is not None and callable(self.score_function):
            self.candidates = self.score_function(
                self._ngrams, total_words=self._total_words
            )

    def learn_ngrams(self):

        self._ngrams = {}
        sentence_no, total_words = -1, 0
        for sentence_no, sentence in tqdm(enumerate(self._sentences)):
            words = self.tokenize(sentence)
            total_words += len(words)

            ngrams = self.prepare_ngram_tuples(
                words,
                max_n=self._ngram.max_n,
                max_window=self._ngram.max_window,
                max_skip=self._ngram.max_skip,
            )
            for ngram in ngrams:
                self._ngrams[ngram] = self._ngrams.get(ngram, 0) + 1

            if (
                (sentence_no > 0)
                and (self._candidates.num_sents_per_pruning > 0)
                and (sentence_no % self._candidates.num_sents_per_pruning == 0)
            ):
                prune_vocab(self._ngrams, self._candidates.prune_min_ngram_count)
            if len(self._ngrams) > self._candidates.max_candidates:
                prune_vocab(self._ngrams, self._candidates.prune_min_ngram_count)
                self._candidates.prune_min_ngram_count += 1

        if self.verbose:
            print("learning ngrams was done. memory= %.3f Gb" % get_process_memory())

        self._ngrams = {
            ngram: freq
            for ngram, freq in self._ngrams.items()
            if freq >= self._candidates.min_count
        }
        self._total_words = total_words
