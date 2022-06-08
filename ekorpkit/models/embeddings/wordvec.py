import os
import logging
import pandas as pd
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from unicodedata import category
from collections import namedtuple
from tqdm import tqdm
from ekorpkit.io.file import save_dataframe, load_dataframe
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class WordVector:
    __gensim__version__ = gensim.__version__

    def __init__(
        self,
        **args,
    ):
        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.corpus = args.corpus
        self.benchmarks = args.benchmarks
        self.verbose = args.verbose
        self.wv = None

    def load(self):
        if self.args.model_type == "glove":
            self.load_glove(self.args.model_path)
        elif self.args.model_type == "word2vec":
            self.load_word2vec(self.args.model_path)
        elif self.args.model_type == "pretrained":
            self.load_pretrained(self.args.model_name)
        else:
            self.load_model(self.args.model_path)

    def load_model(self, model_file):
        self.wv = KeyedVectors.load(model_file)

    def load_word2vec(self, word2vec_file):
        self.wv = KeyedVectors.load_word2vec_format(word2vec_file)

    def load_glove(self, glove_file):
        self.wv = KeyedVectors.load_word2vec_format(
            glove_file, binary=False, no_header=True
        )

    def load_pretrained(self, model_name):
        self.wv = api.load(model_name)

    def evaluate_word_analogies(
        self,
        analogies,
        restrict_vocab=300000,
        case_insensitive=True,
        dummy4unknown=False,
        similarity_function="most_similar",
    ):
        _analogies = self.benchmarks.analogies.get(analogies)
        if _analogies is None:
            raise ValueError(f"Analogies {analogies} not found.")

        score, sections = self.wv.evaluate_word_analogies(
            _analogies,
            restrict_vocab=restrict_vocab,
            case_insensitive=case_insensitive,
            dummy4unknown=dummy4unknown,
            similarity_function=similarity_function,
        )
        log.info(f"Evaluation score: {score}")
        summary = (
            pd.DataFrame(
                [
                    [c["section"], len(c["correct"]), len(c["incorrect"])]
                    for c in sections
                ],
                columns=["category", "correct", "incorrect"],
            )
            .assign(samples=lambda x: x.correct.add(x.incorrect))
            .assign(average=lambda x: x.correct.div(x.samples))
        )
        category_names = self.benchmarks.categories.get(analogies)
        if category_names is not None:
            category_names = eKonf.to_dict(category_names)
            summary["category"] = summary.category.replace(category_names)
        summary["corpus"] = self.corpus
        summary = summary.rename(columns=str.capitalize)

        return {"score": score, "summary": summary, "sections": sections}
