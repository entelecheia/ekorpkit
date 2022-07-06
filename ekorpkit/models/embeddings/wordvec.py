import os
import logging
import pandas as pd
import numpy as np
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from unicodedata import category
from collections import namedtuple
from tqdm.auto import tqdm
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
        restrict_vocab=300_000,
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

        results = {"score": score, "summary": summary}

        sections = pd.DataFrame(sections)
        for _eval in ["correct", "incorrect"]:
            res = sections[["section", _eval]].explode(_eval).reset_index(drop=True)
            res.columns = ["category", "wordpairs"]
            res["category"] = res.category.replace(category_names)
            if case_insensitive:
                res.wordpairs = res.wordpairs.map(lambda x: tuple(map(str.lower, x)))
            results[_eval] = res
        return results

    def reduce_embeddings_2d(self, restrict_vocab=100_000):

        from numpy.linalg import norm
        from sklearn.decomposition import IncrementalPCA

        vectors = self.wv.vectors[:restrict_vocab]
        vectors /= norm(vectors, axis=1).reshape(-1, 1)
        log.info(f"dimensions: {vectors.shape}")

        words = self.wv.index_to_key[:100000]
        word2idx = {w: i for i, w in enumerate(words)}

        pca = IncrementalPCA(n_components=2)

        vectors2D = pca.fit_transform(vectors)
        log.info(f"explained variance: {pca.explained_variance_ratio_}")
        return {"vectors": vectors2D, "word2idx": word2idx}

    def find_most_similar_analogies(self, eval_results, word2idx, vectors, n=1):
        from scipy.spatial.distance import cosine

        def word_to_index(wordpairs):
            idx = []
            for word in wordpairs:
                if word not in word2idx:
                    return None
                idx.append(word2idx[word])
            return tuple(idx)

        def cosine_similarity(idx):
            if idx is None:
                return None
            v1 = vectors[idx[1]] - vectors[idx[0]]
            v2 = vectors[idx[3]] - vectors[idx[2]]
            return cosine(v1, v2)

        eval_results["word2idx"] = eval_results.wordpairs.map(word_to_index)
        eval_results["similarity"] = eval_results.word2idx.map(cosine_similarity)

        # find the most similar wordpairs by cosine similarity of each category group
        eval_results = eval_results.sort_values(by="similarity", ascending=False)
        # first row of each category is the most similar wordpairs
        best_analogies = eval_results.groupby("category").first()
        return best_analogies

    def plot_similar_analogies(
        self, best_analogies, vectors, ncols=3, figsize=(15, 15)
    ):
        ax_args = eKonf.compose(config_group="visualize/plot/ax")
        axes = []
        fc = ec = "darkgrey"
        for s, (category, result) in enumerate(best_analogies.iterrows()):
            best_analogy = result.wordpairs
            analogy_idx = result.word2idx
            best_analogy = [a.capitalize() for a in best_analogy]

            coords = vectors[list(analogy_idx)].tolist()
            xlim, ylim = _get_xylims_of_wordpairs(coords)

            _ax = ax_args.copy()
            _ax.axno = s
            _ax.xlim = str(xlim)
            _ax.ylim = str(ylim)

            annotations = []
            for i in [0, 2]:
                annotations.append(
                    dict(
                        text=best_analogy[i],
                        x=coords[i + 1][0],
                        y=coords[i + 1][1],
                        xtext=coords[i][0],
                        ytext=coords[i][1],
                        arrowprops=dict(
                            width=1, headwidth=5, headlength=5, fc=fc, ec=ec, shrink=0.1
                        ),
                        fontsize=12,
                    )
                )

                annotations.append(
                    dict(
                        text=best_analogy[i + 1],
                        x=coords[i + 1][0],
                        y=coords[i + 1][1],
                        xtext=coords[i + 1][0],
                        ytext=coords[i + 1][1],
                        va="center",
                        ha="center",
                        fontsize=12,
                        color="darkred" if i == 2 else "k",
                    )
                )
            _ax.annotations = annotations
            _ax.title = category
            _ax.grid = True
            axes.append(_ax)

        num_analogies = len(best_analogies)
        nrows = (num_analogies + ncols - 1) // ncols

        cfg = eKonf.compose(config_group="visualize/plot")
        cfg.figure.figsize = figsize
        cfg.figure.fontsize = 10
        cfg.subplots.ncols = ncols
        cfg.subplots.nrows = nrows
        cfg.axes = axes
        eKonf.instantiate(cfg, data=None)


def _get_xylims_of_wordpairs(coordinates):
    coordinates = pd.DataFrame(coordinates)
    xlim, ylim = coordinates.agg(["min", "max"]).T.values
    xrange, yrange = (xlim[1] - xlim[0]) * 0.1, (ylim[1] - ylim[0]) * 0.1
    xlim[0], xlim[1] = xlim[0] - xrange, xlim[1] + xrange
    ylim[0], ylim[1] = ylim[0] - yrange, ylim[1] + yrange
    return tuple(xlim), tuple(ylim)
