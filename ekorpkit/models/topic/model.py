import os
import sys
import random
import itertools
import pandas as pd
import tomotopy as tp
from pathlib import Path
from collections import namedtuple
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.io.load.list import load_wordlist, save_wordlist
from ekorpkit.io.file import save_dataframe, load_dataframe
from ekorpkit.visualize.wordcloud import generate_wordclouds, savefig
from ekorpkit.pipelines.pipe import apply


ModelSummary = namedtuple(
    "ModelSummary",
    [
        "train_dt",
        "filename",
        eKonf.Keys.CORPUS,
        "model_id",
        "model_type",
        "sample_ratio",
        "num_docs",
        "num_words",
        "total_vocabs",
        "used_vocabs",
        "iterations",
        "interval",
        "burn_in",
        "ll_per_word",
        "tw",
        "min_cf",
        "min_df",
        "rm_top",
        "k",
        "k1",
        "k2",
        "alpha",
        "eta",
        "seed",
        "perplexity",
        "u_mass",
        "c_uci",
        "c_npmi",
        "c_v",
    ],
    defaults=[None] * 29,
)

IDF = tp.TermWeight.IDF
ONE = tp.TermWeight.ONE
PMI = tp.TermWeight.PMI


class TopicModel:
    def __init__(
        self,
        model_name,
        model_dir,
        output_dir,
        num_workers=0,
        ngram=None,
        files=None,
        verbose=False,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_dir = Path(str(model_dir))
        self.output_dir = Path(str(output_dir))
        self.num_workers = num_workers
        self.ngram = ngram
        self.files = files
        self.verbose = verbose

        self._raw_corpus = tp.utils.Corpus()
        self._raw_corpus_keys = None
        self.ngrams = None
        self.ngram_model = None
        self._ngram_docs = None
        self.stopwords = []
        self.docs = None
        self.corpus = None
        self.corpora = None
        self.sample_ratio = 1.0
        self.active_model_id = None
        self.model = None
        self.models = {}
        self.labels = []

        self.summary_file = Path(self.files.summary)
        self.summaries = []
        if self.summary_file.is_file():
            df = eKonf.load_data(self.summary_file, index_col=0)
            for row in df.itertuples():
                self.summaries.append(ModelSummary(*row[1:]))

        self.corpus_key_path = Path(self.files.corpus_key)
        self._raw_corpus_key_path = Path(self.files.raw_corpus_key)
        self.ngram_candidates_path = Path(self.files.ngram_candidates)
        self.ngram_model_path = Path(self.files.ngram_model)
        self.ngram_docs_path = Path(self.files.ngram_docs)
        self.stoplist_paths = self.files.stoplist
        if self.stoplist_paths is None:
            self.stoplist_paths = []
        else:
            if isinstance(self.stoplist_paths, str):
                self.stoplist_paths = [self.stoplist_paths]
            else:
                self.stoplist_paths = list(self.stoplist_paths)
        self.stopwords_path = Path(self.files.stopwords)
        self.default_stopwords_path = Path(self.files.default_stopwords)
        self.default_word_prior_path = Path(self.files.default_word_prior)
        self.word_prior_path = Path(self.files.word_prior)
        (self.model_dir).mkdir(exist_ok=True, parents=True)
        (self.output_dir / "figures/wc").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "figures/train").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "output/train").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "figures/tune").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "output/tune").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "logs").mkdir(exist_ok=True, parents=True)

    def _load_raw_corpus(self, reload_corpus=False):
        def data_feeder(docs):
            for doc in docs:
                fd = doc.strip().split(maxsplit=1)
                timepoint = int(fd[0])
                yield fd[1], None, {"timepoint": timepoint}

        if not self._raw_corpus or reload_corpus:
            self._raw_corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
            if self.corpora is None:
                raise ValueError("corpora is not set")
            with elapsed_timer() as elapsed:
                self.corpora.load()
                self.corpora.concat_corpora()
                df = self.corpora._data

                self._raw_corpus_keys = df[self.corpora._id_keys].values.tolist()
                self._raw_corpus.process(df[self.corpora._text_key].to_list())

                eKonf.save_data(
                    df[self.corpora.IDs],
                    self._raw_corpus_key_path,
                    verbose=self.verbose,
                )
                print("Elapsed time is %.2f seconds" % elapsed())

    def extract_ngrams(self):
        if self.ngrams is None:
            self._load_raw_corpus()
            assert self._raw_corpus, "Load a corpus first"
            with elapsed_timer() as elapsed:
                print("Extracting ngram candidates")
                self.ngrams = self._raw_corpus.extract_ngrams(
                    min_cf=self.ngram.min_cf,
                    min_df=self.ngram.min_df,
                    max_len=self.ngram.max_len,
                    max_cand=self.ngram.max_cand,
                    min_score=self.ngram.min_score,
                    normalized=self.ngram.normalized,
                    workers=self.num_workers,
                )
                # print(self.ngrams)
                ngram_list = [
                    {"words": ",".join(cand.words), "score": cand.score}
                    for cand in self.ngrams
                ]
                df = pd.DataFrame(ngram_list)
                eKonf.save_data(df, self.ngram_candidates_path, verbose=self.verbose)
                print("Elapsed time is %.2f seconds" % elapsed())

    def _load_ngram_docs(self, rebuild=False):
        if self.ngram_docs_path.is_file() and not rebuild:
            with elapsed_timer() as elapsed:
                print(f"Starting to load ngram documents from {self.ngram_docs_path}")
                self._raw_corpus = tp.utils.Corpus().load(self.ngram_docs_path)
                df = eKonf.load_data(self._raw_corpus_key_path)
                self._raw_corpus_keys = df[self.corpora._id_keys].values.tolist()
                # self._raw_corpus.load(self.ngram_doc_path)
                print(f"{len(self._raw_corpus)} documents are loaded.")
                print("Elapsed time is %.2f seconds" % elapsed())
        else:
            self.extract_ngrams()
            assert self.ngrams, "Load a ngrams first"
            print("Building ngram docs by concatenaing words in ngram list")
            self._raw_corpus.concat_ngrams(self.ngrams, self.ngram.delimiter)
            self._raw_corpus.save(self.ngram_docs_path)

    def _load_stopwords(self):
        self.stopwords = []
        if self.stopwords_path.is_file():
            self.stopwords = load_wordlist(self.stopwords_path, lowercase=True)
        else:
            if self.default_stopwords_path.is_file():
                self.stopwords = load_wordlist(
                    self.default_stopwords_path, lowercase=True
                )
            else:
                self.stopwords = ["."]
        save_wordlist(self.stopwords, self.stopwords_path)
        if self.verbose:
            print(f"{len(self.stopwords)} stopwords are loaded.")

        for path in self.stoplist_paths:
            if os.path.exists(path):
                stopwords = load_wordlist(path, lowercase=True)
                self.stopwords += stopwords
                save_wordlist(stopwords, path)
                if self.verbose:
                    print(f"{len(stopwords)} stopwords are loaded from {path}")

    def _load_word_prior(self):
        if self.word_prior_path.is_file():
            self.word_prior = eKonf.load(self.word_prior_path)
            print(self.word_prior)
        else:
            if self.default_word_prior_path.is_file():
                self.word_prior = eKonf.load(self.default_word_prior_path)
                print(self.word_prior)
            else:
                self.word_prior = {}
        eKonf.save(self.word_prior, self.word_prior_path)

    def load_corpus(
        self,
        sample_ratio=1.0,
        reload_corpus=False,
        min_num_words=5,
        min_word_len=2,
        rebuild=False,
        **kwargs,
    ):
        sample_ratio = sample_ratio if sample_ratio else self.sample_ratio
        if self.corpus and self.sample_ratio == sample_ratio and not reload_corpus:
            print("Corpus is already loaded w/ sample_ratio: {}".format(sample_ratio))
            return True
        else:
            print("Start loading corpus w/ sample_ratio: {}".format(sample_ratio))
        if not self._raw_corpus:
            self._load_ngram_docs(rebuild=rebuild)
        self._load_stopwords()
        assert self._raw_corpus, "Load ngram documents first"
        assert self.stopwords, "Load stopwords first"
        if sample_ratio and sample_ratio < 1.0 and sample_ratio > 0.0:
            docs = random.sample(
                self._raw_corpus, int(len(self._raw_corpus) * sample_ratio)
            )
            self.sample_ratio = sample_ratio
        else:
            docs = self._raw_corpus
            self.sample_ratio = 1.0
        self.corpus = tp.utils.Corpus()
        self.corpus_keys = []

        n_skipped = 0
        for i_doc, doc in tqdm(enumerate(docs)):
            words = [
                w for w in doc if w not in self.stopwords and len(w) >= min_word_len
            ]
            if len(words) > min_num_words:
                self.corpus.add_doc(words=words)
                self.corpus_keys.append(self._raw_corpus_keys[i_doc])
            else:
                if self.verbose > 5:
                    print(
                        f"Skipped - index={i_doc}, key={self._raw_corpus_keys[i_doc]}, words={list(words)}"
                    )
                n_skipped += 1
        print(f"Total {i_doc-n_skipped+1} documents are loaded.")
        print(f"Total {n_skipped} documents are removed from the corpus.")
        df = pd.DataFrame(self.corpus_keys, columns=self.corpora._id_keys)
        eKonf.save_data(
            df[self.corpora._id_keys], self.corpus_key_path, verbose=self.verbose
        )

    def infer_topics(
        self,
        output_dir=None,
        output_file=None,
        iterations=100,
        min_num_words=5,
        min_word_len=2,
        num_workers=0,
        use_batcher=True,
        minibatch_size=None,
        **kwargs,
    ):

        self._load_stopwords()
        assert self.stopwords, "Load stopwords first"
        assert self.model, "Model not found"
        print("Infer document out of the model")

        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        num_workers = num_workers if num_workers else 1
        text_key = self.corpora._text_key
        id_keys = self.corpora._id_keys

        df_ngram = eKonf.load_data(self.ngram_candidates_path)
        ngrams = []
        for ngram in df_ngram['words'].to_list():
            ngrams.append(ngram.split(','))
        
        simtok = SimpleTokenizer(
            stopwords=self.stopwords,
            min_word_len=min_word_len,
            min_num_words=min_num_words,
            ngrams=ngrams,
            ngram_delimiter=self.ngram.delimiter,
            verbose=self.verbose,
        )

        if self.corpora is None:
            raise ValueError("corpora is not set")
        with elapsed_timer() as elapsed:
            self.corpora.load()
            self.corpora.concat_corpora()
            df = self.corpora._data
            df.dropna(subset=[text_key], inplace=True)
            df[text_key] = apply(
                simtok.tokenize,
                df[text_key],
                description=f"tokenize",
                verbose=self.verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            df = df.dropna(subset=[text_key]).reset_index(drop=True)
            if self.verbose:
                print(df.tail())

            docs = []
            indexes_to_drop = []
            for ix in df.index:
                doc = df.loc[ix, text_key]
                mdoc = self.model.make_doc(doc)
                if mdoc:
                    docs.append(mdoc)
                else:
                    print(f"Skipped - {doc}")
                    indexes_to_drop.append(ix)
            df = df.drop(df.index[indexes_to_drop]).reset_index(drop=True)
            if self.verbose:
                print(f"{len(docs)} documents are loaded from: {len(df.index)}.")

            topic_dists, ll = self.model.infer(
                docs, workers=num_workers, iter=iterations
            )
            if self.verbose:
                print(topic_dists[-1:], ll)
                print(f"Total inferred: {len(topic_dists)}, from: {len(df.index)}")

            if len(topic_dists) == len(df.index):
                idx = range(len(topic_dists[0]))
                df_infer = pd.DataFrame(topic_dists, columns=[f"topic{i}" for i in idx])
                df_infer = pd.concat([df[id_keys], df_infer], axis=1)
                output_path = f"{output_dir}/{output_file}"
                eKonf.save_data(df_infer, output_path, verbose=self.verbose)
                print(f"Corpus is saved as {output_path}")
            else:
                print("The number of inferred is not same as the number of input.")

            print("Elapsed time is %.2f seconds" % elapsed())

    def save_document_topic_dists(self):
        assert self.model, "Model not found"

        topic_dists = []
        for doc in self.model.docs:
            topic_dists.append(doc.get_topic_dist())
        # print(topic_dists[-1:])

        if self.corpus_keys:
            df = pd.DataFrame(self.corpus_keys, columns=self.corpora._id_keys)
        elif self.corpus_key_path.is_file():
            df = eKonf.load_data(self.corpus_key_path, verbose=self.verbose)
        else:
            print("Corpus keys do not exist")
            return

        print(f"Total inferred: {len(topic_dists)}, from: {len(df.index)}")
        if len(topic_dists) == len(df.index):
            idx = range(len(topic_dists[0]))
            df_infer = pd.DataFrame(topic_dists, columns=[f"topic{i}" for i in idx])
            # print(df_infer.tail())
            df_infer = pd.concat([df[self.corpora._id_keys], df_infer], axis=1)
            print(df_infer.tail())

            filename = "{}-{}-topic_dists.csv".format(
                self.model_name, self.active_model_id
            )
            output_path = f"{self.model_dir}/{filename}"
            eKonf.save_data(df_infer, output_path, verbose=self.verbose)
            print(f"Corpus is saved as {output_path}")
        else:
            print("The number of inferred is not same as the number of input.")

    def tune_params(
        self,
        model_type="LDA",
        topics=[20],
        alphas=[0.1],
        etas=[0.01],
        sample_ratios=[0.1],
        tws=[IDF],
        min_cf=5,
        rm_top=0,
        min_df=0,
        burn_in=0,
        interval=10,
        iterations=100,
        seed=None,
        eval_coherence=True,
        save=False,
        save_full=False,
    ):
        """
        # Topics range
        topics = range(min_topics, max_topics, step_size)
        # Alpha parameter
        alphas = np.arange(0.01, 1, 0.3)
        # Beta parameter
        etas = np.arange(0.01, 1, 0.3)
        # Validation sets
        sample_ratios = [0.1, 0.5]
        """
        total_iters = (
            len(etas) * len(alphas) * len(topics) * len(tws) * len(sample_ratios)
        )
        exec_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_lls = None
        ys = []
        for i, (sr, k, a, e, tw) in tqdm(
            enumerate(
                itertools.product(sample_ratios, topics, alphas, etas, tws),
            ),
            total=total_iters,
        ):
            # train a model
            print(
                "sample_ratio: {}, k:{}, alpha:{}, eta:{}, tw:{}".format(
                    sr, k, a, e, str(tw)
                )
            )
            # self.load_corpus(sample_ratio=sr)
            df_ll, m_sum = self.train_model(
                model_type=model_type,
                sample_ratio=sr,
                k=k,
                tw=tw,
                alpha=a,
                eta=e,
                min_cf=min_cf,
                rm_top=rm_top,
                min_df=min_df,
                burn_in=burn_in,
                interval=interval,
                iterations=iterations,
                seed=seed,
                eval_coherence=eval_coherence,
                save=save,
                save_full=save_full,
            )
            margs = []
            if len(topics) > 1:
                margs.append("k={}".format(k))
            if len(alphas) > 1:
                margs.append("a={}".format(a))
            if len(etas) > 1:
                margs.append("e={}".format(e))
            if len(tws) > 1:
                margs.append("tw={}".format(tw))
            if len(sample_ratios) > 1:
                margs.append("sr={}".format(sr))
            y = ",".join(margs) if len(margs) > 0 else "ll_{}".format(i)
            ys.append(y)

            df_ll.rename(columns={"ll_per_word": y}, inplace=True)
            if df_lls is not None:
                df_lls = df_lls.merge(df_ll, on="iter")
            else:
                df_lls = df_ll

        out_file = "{}-{}-ll_per_word-{}.csv".format(
            self.model_name, model_type, exec_dt
        )
        out_file = str(self.model_dir / "output/tune" / out_file)
        df_lls.to_csv(out_file)
        ax = df_lls.plot(x="iter", y=ys, kind="line")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        out_file = "{}-{}-ll_per_word-{}.png".format(
            self.model_name, model_type, exec_dt
        )
        out_file = str(self.model_dir / "figures/tune" / out_file)
        savefig(out_file, transparent=False, dpi=300)

    def train_model(
        self,
        model_id=None,
        model_type="LDA",
        model_path=None,
        sample_ratio=None,
        k=None,
        k1=None,
        k2=None,
        t=1,
        tw=IDF,
        gamma=2,
        alpha=0.1,
        eta=0.01,
        phi=0.1,
        min_cf=5,
        rm_top=0,
        min_df=0,
        burn_in=0,
        interval=10,
        iterations=100,
        seed=None,
        eval_coherence=False,
        set_word_prior=False,
        save=True,
        save_full=True,
        **kwargs,
    ):

        self.load_corpus(sample_ratio=sample_ratio)
        assert self.corpus, "Load corpus first"

        exec_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = model_type.upper()
        if model_id is None:
            margs = [model_type]
            if k:
                margs.append("k{}".format(k))
            if k1:
                margs.append("k{}".format(k1))
            if k2:
                margs.append("k{}".format(k2))
            if self.sample_ratio < 1.0:
                margs.append("sr{}".format(int(self.sample_ratio * 100)))
            model_id = ".".join(margs)
        self.active_model_id = model_id

        if not seed:
            random.seed()
            seed = random.randint(0, 32767)

        if model_type == "LDA":
            mdl = tp.LDAModel(
                tw=tw,
                k=k,
                min_cf=min_cf,
                rm_top=rm_top,
                alpha=alpha,
                eta=eta,
                corpus=self.corpus,
                seed=seed,
            )
        elif model_type == "HPA":
            mdl = tp.HPAModel(
                tw=tw,
                k1=k1,
                k2=k2,
                min_cf=min_cf,
                rm_top=rm_top,
                alpha=alpha,
                eta=eta,
                corpus=self.corpus,
                seed=seed,
            )
        elif model_type == "HDP":
            mdl = tp.HDPModel(
                tw=tw,
                min_cf=min_cf,
                rm_top=rm_top,
                gamma=gamma,
                alpha=alpha,
                initial_k=k,
                corpus=self.corpus,
                seed=seed,
            )
        elif model_type == "CTM":
            mdl = tp.CTModel(
                tw=tw,
                k=k,
                min_cf=min_cf,
                rm_top=rm_top,
                eta=eta,
                smoothing_alpha=alpha,
                corpus=self.corpus,
                seed=seed,
            )
        elif model_type == "DTM":
            mdl = tp.DTModel(
                tw=tw,
                k=k,
                t=t,
                min_cf=min_cf,
                rm_top=rm_top,
                alpha_var=alpha,
                eta_var=eta,
                phi_var=phi,
                corpus=self.corpus,
                seed=seed,
            )
        else:
            print("{} is not supported".format(model_type))
            return False

        if set_word_prior:
            self._load_word_prior()
            for tno, words in self.word_prior.items():
                print(f"Set words {words} to topic #{tno} as prior.")
                for word in words:
                    mdl.set_word_prior(
                        word, [1.0 if i == int(tno) else 0.1 for i in range(k)]
                    )

        mdl.burn_in = burn_in
        mdl.train(0)
        print(
            "Num docs:",
            len(mdl.docs),
            ", Vocab size:",
            mdl.num_vocabs,
            ", Num words:",
            mdl.num_words,
        )
        print("Removed top words:", mdl.removed_top_words)

        print(
            "Training model by iterating over the corpus {} times, {} iterations at a time".format(
                iterations, interval
            )
        )
        ll_per_words = []
        for i in range(0, iterations, interval):
            mdl.train(interval)
            if model_type == "HDP":
                print(
                    "Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}".format(
                        i, mdl.ll_per_word, mdl.live_k
                    )
                )
            else:
                print("Iteration: {}\tLog-likelihood: {}".format(i, mdl.ll_per_word))
            ll_per_words.append((i, mdl.ll_per_word))
        df_ll = pd.DataFrame(ll_per_words, columns=["iter", "ll_per_word"])
        out_file = "{}-{}-ll_per_word-{}.csv".format(
            self.model_name, self.active_model_id, exec_dt
        )
        out_file = str(self.output_dir / "output/train" / out_file)
        df_ll.to_csv(out_file)
        ax = df_ll.plot(x="iter", y="ll_per_word", kind="line")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        out_file = "{}-{}-ll_per_word-{}.png".format(
            self.model_name, self.active_model_id, exec_dt
        )
        out_file = str(self.output_dir / "figures/train" / out_file)
        savefig(out_file, transparent=False, dpi=300)

        mdl.summary()
        self.model = mdl
        self.models[model_id] = mdl

        coh_values = self.eval_coherence_value() if eval_coherence else {}
        # calculate coherence using preset

        original_stdout = sys.stdout
        out_file = "{}-{}-summary-{}.txt".format(
            self.model_name, self.active_model_id, exec_dt
        )
        out_file = str(self.output_dir / "logs" / out_file)
        with open(out_file, "w") as f:
            sys.stdout = f  # Change the standard output to the file.
            mdl.summary()
            if coh_values:
                print("<Topic Coherence Scores>")
                for cm, cv in coh_values.items():
                    print(f"| {cm}: {cv}")
            sys.stdout = original_stdout  # Reset the standard output.

        if model_path is None:
            model_path = "{}-{}.mdl".format(self.model_name, model_id)
            model_path = self.model_dir / model_path
        else:
            model_path = Path(model_path)
        if save:
            mdl.save(str(model_path), full=save_full)

        self.save_document_topic_dists()

        entry = ModelSummary(
            exec_dt,
            model_path.name,
            self.model_name,
            self.active_model_id,
            model_type,
            self.sample_ratio,
            len(mdl.docs),
            mdl.num_words,
            len(mdl.vocabs),
            len(mdl.used_vocabs),
            iterations,
            interval,
            mdl.burn_in,
            round(mdl.ll_per_word, 2),
            str(tw),
            min_cf,
            min_df,
            rm_top,
            mdl.k,
            k1,
            k2,
            alpha,
            eta,
            seed,
            round(mdl.perplexity),
            round(coh_values["u_mass"], 3) if "u_mass" in coh_values else None,
            round(coh_values["c_uci"], 3) if "c_uci" in coh_values else None,
            round(coh_values["c_npmi"], 3) if "c_npmi" in coh_values else None,
            round(coh_values["c_v"], 3) if "c_v" in coh_values else None,
        )
        self.summaries.append(entry)
        df = pd.DataFrame(self.summaries)
        eKonf.save_data(df, self.summary_file, index=True)
        return df_ll, entry

    def eval_coherence_value(
        self, coherence_metrics=["u_mass", "c_uci", "c_npmi", "c_v"]
    ):
        assert self.model, "Model not found"
        mdl = self.model
        coh_values = {}
        for metric in coherence_metrics:
            coh = tp.coherence.Coherence(mdl, coherence=metric)
            average_coherence = coh.get_score()
            coh_values[metric] = average_coherence
            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
            print("==== Coherence : {} ====".format(metric))
            print("Average:", average_coherence, "\nPer Topic:", coherence_per_topic)
            print()
        return coh_values

    def load_model(self, model_id=None, model_file=None, reload_model=False, **kwargs):
        if model_id:
            self.active_model_id = model_id
        if self.active_model_id in self.models and not reload_model:
            print("The model is already loaded.")
            return True

        if model_file is None:
            model_file = "{}-{}.mdl".format(self.model_name, self.active_model_id)
        model_path = self.model_dir / model_file
        print("Loading a model from {}".format(model_path))
        if model_path.is_file():
            if not self.active_model_id:
                self.active_model_id = model_path.stem.split("-")[-1]
            model_type = self.active_model_id.split(".")[0]
            model_path = str(model_path)
            with elapsed_timer() as elapsed:
                if model_type == "LDA":
                    mdl = tp.LDAModel.load(model_path)
                elif model_type == "HPA":
                    mdl = tp.HPAModel.load(model_path)
                elif model_type == "HDP":
                    mdl = tp.HDPModel.load(model_path)
                else:
                    print("{} is not supported".format(model_type))
                    return False
                self.models[self.active_model_id] = mdl
                self.model = mdl
                print("Elapsed time is %.2f seconds" % elapsed())
        else:
            self.model = None
            print("Model file not found")

    def save_labels(self, names=None, **kwargs):
        if names is None:
            print("No names are given")
            return

        if not self.labels:
            self.label_topics()
        for k in names:
            self.labels[int(k)]["topic_name"] = names[k]
            if self.verbose:
                print(f"{k}: {names[k]}")
        label_file = "{}-labels.csv".format(self.active_model_id)
        label_file = self.output_dir / label_file
        df = pd.DataFrame(self.labels)
        eKonf.save_data(df, label_file, index=False, verbose=self.verbose)

    def label_topics(
        self,
        rebuild=False,
        use_pmiextractor=False,
        min_cf=10,
        min_df=5,
        max_len=5,
        max_cand=100,
        smoothing=1e-2,
        mu=0.25,
        window_size=100,
        top_n=10,
        **kwargs,
    ):

        label_file = "{}-labels.csv".format(self.active_model_id)
        label_file = self.output_dir / label_file
        if label_file.is_file() and not rebuild:
            print("loading labels from {}".format(label_file))
            df = eKonf.load_data(label_file)
            self.labels = df.to_dict("records")
        else:
            assert self.model, "Model not found"
            mdl = self.model
            if use_pmiextractor:
                # extract candidates for auto topic labeling
                print("extract candidates for auto topic labeling")
                extractor = tp.label.PMIExtractor(
                    min_cf=min_cf, min_df=min_df, max_len=max_len, max_cand=max_cand
                )
                with elapsed_timer() as elapsed:
                    cands = extractor.extract(mdl)
                    print("Elapsed time is %.2f seconds" % elapsed())
                    labeler = tp.label.FoRelevance(
                        mdl,
                        cands,
                        min_df=min_df,
                        smoothing=smoothing,
                        mu=mu,
                        window_size=window_size,
                    )
                    print("Elapsed time is %.2f seconds" % elapsed())
                self.labeler = labeler

            labels = []
            for k in range(mdl.k):
                print("== Topic #{} ==".format(k))
                name = f"Topic #{k}"
                if use_pmiextractor:
                    lbls = ",".join(
                        label
                        for label, score in labeler.get_topic_labels(k, top_n=top_n)
                    )
                    print(
                        "Labels:",
                        ", ".join(
                            label
                            for label, score in labeler.get_topic_labels(k, top_n=top_n)
                        ),
                    )
                wrds = ",".join(
                    word for word, prob in mdl.get_topic_words(k, top_n=top_n)
                )
                if use_pmiextractor:
                    label = {
                        "topic_no": k,
                        "topic_num": f"topic{k}",
                        "topic_name": name,
                        "topic_labels": lbls,
                        "topic_words": wrds,
                    }
                else:
                    label = {
                        "topic_no": k,
                        "topic_num": f"topic{k}",
                        "topic_name": name,
                        "topic_words": wrds,
                    }
                labels.append(label)
                for word, prob in mdl.get_topic_words(k, top_n=top_n):
                    print(word, prob, sep="\t")
                print()

            self.labels = labels
            df = pd.DataFrame(self.labels)
            eKonf.save_data(df, label_file, index=False, verbose=self.verbose)

    def visualize(self, **kwargs):
        import pyLDAvis

        assert self.model, "Model not found"
        mdl = self.model
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack(
            [
                doc.get_topic_dist()
                for doc in mdl.docs
                if np.sum(doc.get_topic_dist()) == 1
            ]
        )
        doc_lengths = np.array(
            [len(doc.words) for doc in mdl.docs if np.sum(doc.get_topic_dist()) == 1]
        )
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
        )
        out_file = "{}-{}-ldavis.html".format(self.model_name, self.active_model_id)
        out_file = str(self.output_dir / "output" / out_file)
        pyLDAvis.save_html(prepared_data, out_file)

    def get_topic_words(self, top_n=10):
        """Wrapper function to extract topics from trained tomotopy HDP model

        ** Inputs **
        top_n: int -> top n words in topic based on frequencies

        ** Returns **
        topics: dict -> per topic, an arrays with top words and associated frequencies
        """
        assert self.model, "Model not found"
        mdl = self.model

        # Get most important topics by # of times they were assigned (i.e. counts)
        sorted_topics = [
            k
            for k, v in sorted(
                enumerate(mdl.get_count_by_topics()), key=lambda x: x[1], reverse=True
            )
        ]

        topics = dict()
        # For topics found, extract only those that are still assigned
        for k in sorted_topics:
            if type(mdl) in ["tomotopy.HPAModel", "tomotopy.HDPModel"]:
                if not mdl.is_live_topic(k):
                    continue  # remove un-assigned topics at the end (i.e. not alive)
            topic_wp = []
            for word, prob in mdl.get_topic_words(k, top_n=top_n):
                topic_wp.append((word, prob))

            topics[k] = topic_wp  # store topic word/frequency array

        return topics

    def topic_wordclouds(
        self,
        title_fontsize=20,
        title_color="green",
        top_n=100,
        ncols=5,
        nrows=1,
        dpi=300,
        figsize=(10, 10),
        save=True,
        mask_dir=None,
        wordclouds=None,
        save_each=False,
        save_masked=False,
        fontpath=None,
        colormap="PuBu",
        **kwargs,
    ):
        """Wrapper function that generates wordclouds for ALL topics of a tomotopy model
        ** Inputs **
        topic_dic: dict -> per topic, an arrays with top words and associated frequencies
        save: bool -> If the user would like to save the images

        ** Returns **
        wordclouds as plots
        """
        assert self.model, "Model not found"
        num_topics = self.model.k

        if figsize is not None and isinstance(figsize, str):
            figsize = eval(figsize)
        if mask_dir is None:
            mask_dir = str(self.output_dir / "figures/masks")
        fig_output_dir = str(self.output_dir / "figures/wc")
        fig_filename_format = "{}-{}-wc_topic".format(
            self.model_name, self.active_model_id
        )
        if wordclouds is None:
            wordclouds_args = {}
        else:
            wordclouds_args = eKonf.to_dict(wordclouds)
        for k in range(num_topics):
            topic_freq = dict(self.model.get_topic_words(k, top_n=top_n))
            if k in wordclouds_args:
                wc_args = wordclouds_args[k]
            else:
                wc_args = {}
            title = wc_args.get("title", None)
            if title is None:
                if self.labels:
                    topic_name = self.labels[k]["name"]
                    if topic_name.startswith("Topic #"):
                        topic_name = None
                else:
                    topic_name = None
                if topic_name:
                    title = f"Topic #{k} - {topic_name}"
                else:
                    title = f"Topic #{k}"
                wc_args["title"] = title
            wc_args["word_freq"] = topic_freq
            wordclouds_args[k] = wc_args

        generate_wordclouds(
            wordclouds_args,
            fig_output_dir,
            fig_filename_format,
            title_fontsize=title_fontsize,
            title_color=title_color,
            ncols=ncols,
            nrows=nrows,
            dpi=dpi,
            figsize=figsize,
            save=save,
            mask_dir=mask_dir,
            save_each=save_each,
            save_masked=save_masked,
            fontpath=fontpath,
            colormap=colormap,
            verbose=self.verbose,
            **kwargs,
        )


class SimpleTokenizer:
    """Class to tokenize texts for a corpus"""

    def __init__(
        self,
        stopwords=[],
        min_word_len=2,
        min_num_words=5,
        verbose=False,
        ngrams=[],
        ngram_delimiter="_",
        **kwargs,
    ):
        self.stopwords = stopwords if stopwords else []
        self.min_word_len = min_word_len
        self.min_num_words = min_num_words
        self.ngram_delimiter = ngram_delimiter
        if ngrams:
            self.ngrams = {ngram_delimiter.join(ngram): ngram for ngram in ngrams}
        else:
            self.ngrams = {}
        self.verbose = verbose
        self.verbose = verbose
        if verbose:
            print(f"{self.__class__.__name__} initialized with:")
            print(f"\tstopwords: {len(self.stopwords)}")
            print(f"\tmin_word_len: {self.min_word_len}")
            print(f"\tmin_num_words: {self.min_num_words}")
            print(f"\tngrams: {len(self.ngrams)}")
            print(f"\tngram_delimiter: {self.ngram_delimiter}")

    def tokenize(self, text):
        if text is None:
            return None
        if len(self.ngrams) > 0:
            words = text.split()
            for repl, ngram in self.ngrams.items():
                words = self.replace_seq(words, ngram, repl)
        else:
            words = text.split()
        words = [
            w for w in words if w not in self.stopwords and len(w) >= self.min_word_len
        ]
        if len(set(words)) > self.min_num_words:
            return words
        else:
            return None

    @staticmethod
    def replace_seq(sequence, subseq, repl):
        if len(sequence) < len(subseq):
            return sequence
        return eval(str(list(sequence)).replace(str(list(subseq))[1:-1], f"'{repl}'"))
