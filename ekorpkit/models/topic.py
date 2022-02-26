import argparse
import os
import sys
import platform
import random
import itertools
import pandas as pd
import orjson as json
import tomotopy as tp
from pathlib import Path
from collections import namedtuple
from datetime import datetime
from tqdm import tqdm
import numpy as np
from contextlib import contextmanager
from timeit import default_timer
from ..io.load.list import load_wordlist, save_wordlist
# from ..corpora.loader import load_corpus_paths

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pyLDAvis
from wordcloud import WordCloud



@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def _get_font_name(set_font_for_matplot=True, font_path=None):
    if not font_path:
        if platform.system() == 'Darwin':
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        elif platform.system() == 'Windows':
            font_path = 'c:/Windows/Fonts/malgun.ttf'
        elif platform.system() == 'Linux':
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

    if not Path(font_path).is_file():
        font_name = None
        font_path = None
        print(f'Font file does not exist at {font_path}')
        if platform.system() == 'Linux':
            font_install_help = '''
            apt install fontconfig 
            apt install fonts-nanum* 
            fc-list | grep -i nanum
            '''
            print(font_install_help)
    else:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        if set_font_for_matplot and font_name:
            rc('font', family=font_name)
            plt.rcParams['axes.unicode_minus'] = False
    print("font family: ", plt.rcParams['font.family'])
    print("font path: ", font_path)
    return font_name, font_path


ModelSummary = namedtuple(
    'ModelSummary', [
        'train_dt',
        'filename',
        'corpus',
        'model_id',
        'model_type',
        'sample_ratio',
        'num_docs',
        'num_words',
        'total_vocabs',
        'used_vocabs',
        'iterations',
        'interval',
        'burn_in',
        'll_per_word',
        'tw',
        'min_cf',
        'min_df',
        'rm_top',
        'k',
        'k1',
        'k2',
        'alpha',
        'eta',
        'seed',
        'perplexity',
        'u_mass',
        'c_uci',
        'c_npmi',
        'c_v'
    ], defaults=[None]*29)

IDF = tp.TermWeight.IDF
ONE = tp.TermWeight.ONE
PMI = tp.TermWeight.PMI


class TopicModel():
    def __init__(
        self, model_name, 
        model_dir,
        output_dir,
        corpus_names=None,
        corpus_dir=None,
        text_key='text', id_keys=['id', 'chunk_no'],
        corpus_key_filename=None,
        num_workers=0,
        ngram_path=None,
        ngram_min_cf=10, ngram_min_df=10, ngram_max_len=5, 
        ngram_max_cand=5000, ngram_min_score=-1, ngram_normalized=True,
        ngram_delimiter='_', ngram_doc_path=None,
        stopwords_path=None, min_word_len=2,
        word_prior_path=None
    ):
        self._raw_corpus = tp.utils.Corpus()
        self._raw_corpus_keys = None
        self.ngrams = None
        self.ngram_min_cf = ngram_min_cf
        self.ngram_min_df = ngram_min_df
        self.ngram_max_len = ngram_max_len
        self.ngram_max_cand = ngram_max_cand
        self.ngram_min_score = ngram_min_score
        self.ngram_normalized = ngram_normalized
        self.ngram_delimiter = ngram_delimiter
        self.num_workers = num_workers
        self._ngram_docs = None
        self.min_word_len = min_word_len
        self.stopwords = []
        self.docs = None
        self.corpus = None
        self.sample_ratio = 1.0
        self.active_model_id = None
        self.model_name = model_name
        self.model_dir = Path(str(model_dir))/model_name
        self.output_dir = Path(str(output_dir))/model_name
        self.corpus_names = corpus_names
        self.corpus_dir = Path(str(corpus_dir))
        self.text_key = text_key
        self.corpus_filename_col = 'corpus_filename'
        self.id_keys = [self.corpus_filename_col] + list(id_keys)
        self.model = None
        self.models = {}
        self.labels = []

        self.summary_file = self.model_dir/'model_summary.csv'
        self.summaries = []
        if self.summary_file.is_file():
            df = pd.read_csv(self.summary_file, index_col=0)
            for row in df.itertuples():
                self.summaries.append(ModelSummary(*row[1:]))

        if corpus_key_filename is None:
            self.corpus_key_path = self.model_dir/'{}-keys.csv'.format(self.model_name)
            self._raw_corpus_key_path = self.model_dir/'{}-raw_keys.csv'.format(self.model_name)
        else:
            self.corpus_key_path = self.model_dir/'{}-keys.csv'.format(self.model_name)
            self._raw_corpus_key_path = self.model_dir/'{}-raw_keys.csv'.format(self.model_name)

        if ngram_path is None:
            self.ngram_path = self.output_dir/'{}-ngram-candidates.csv'.format(self.model_name)
        else:
            self.ngram_path = Path(ngram_path)

        if ngram_doc_path is None:
            self.ngram_doc_path = self.model_dir/'{}-ngram-docs.pkl'.format(self.model_name)
        else:
            self.ngram_doc_path = Path(ngram_doc_path)

        self.stopwords_path = self.output_dir/'{}-stopwords.txt'.format(self.model_name)
        if stopwords_path is None:
            self._stopwords_path = self.stopwords_path
        else:
            self._stopwords_path = Path(stopwords_path)

        self.word_prior_path = self.output_dir/'{}-word_prior.json'.format(self.model_name)
        if word_prior_path is None:
            self._word_prior_path = self.word_prior_path
        else:
            self._word_prior_path = Path(word_prior_path)

        (self.model_dir).mkdir(exist_ok=True, parents=True)
        (self.output_dir/'figures/wc').mkdir(exist_ok=True, parents=True)
        (self.output_dir/'figures/train').mkdir(exist_ok=True, parents=True)
        (self.output_dir/'output/train').mkdir(exist_ok=True, parents=True)
        (self.output_dir/'figures/tune').mkdir(exist_ok=True, parents=True)
        (self.output_dir/'output/tune').mkdir(exist_ok=True, parents=True)
        (self.output_dir/'logs').mkdir(exist_ok=True, parents=True)
        
    def _load_raw_corpus(self, reload=False):
        def convert_to_token_list(row):
            text = row[self.text_key]
            if not isinstance(text, str):
                return None
            tokens = [w.strip() for w in text.split()] 
            return tokens

        if not self._raw_corpus or reload:
            # corpus_paths = load_corpus_paths(self.corpus_dir, self.corpus_names, corpus_type='dataframe', corpus_filetype='csv')
            self._raw_corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
            # with elapsed_timer() as elapsed:
            #     key_dfs = []
            #     for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
            #         print('Starting to load documents from {}'.format(corpus_file))
            #         df = pd.read_csv(corpus_file, index_col=0)
            #         df[self.corpus_filename_col] = Path(corpus_file).name
            #         # df[self.text_key] = df.apply(convert_to_token_list, axis=1)
            #         df = df.dropna(subset=[self.text_key])
            #         key_dfs.append(df)

            #     df = pd.concat(key_dfs, ignore_index=True, axis=0)
            #     # self._raw_corpus = df[self.text_key].to_list()
            #     self._raw_corpus_keys = df[self.id_keys].values.tolist()
            #     self._raw_corpus.process(df[self.text_key].to_list())

            #     df[self.id_keys].to_csv(self._raw_corpus_key_path, header=True)
            #     print("Elapsed time is %.2f seconds" % elapsed() )

    def extract_ngrams(
        self, 
    ):
        if not self.ngrams:
            self._load_raw_corpus()
            assert self._raw_corpus, 'Load a corpus first'
            with elapsed_timer() as elapsed:
                print('Extracting ngram candidates')    
                self.ngrams = self._raw_corpus.extract_ngrams(
                    min_cf=self.ngram_min_cf, min_df=self.ngram_min_df, 
                    max_len=self.ngram_max_len, max_cand=self.ngram_max_cand, 
                    min_score=self.ngram_min_score, normalized=self.ngram_normalized, 
                    workers=self.num_workers
                )
                # print(self.ngrams)
                ngram_list = [{'words': ','.join(cand.words), 'score': cand.score} 
                                for cand in self.ngrams]
                df = pd.DataFrame(ngram_list)
                df.to_csv(self.ngram_path, header=True)
                print("Elapsed time is %.2f seconds" % elapsed() )

    def _load_ngram_docs(self, rebuild=False):
        if self.ngram_doc_path.is_file() and not rebuild:
            with elapsed_timer() as elapsed:
                print('Starting to load ngram documents from {}'.format(self.ngram_doc_path))
                self._raw_corpus = tp.utils.Corpus().load(self.ngram_doc_path)
                df = pd.read_csv(self._raw_corpus_key_path)
                self._raw_corpus_keys = df[self.id_keys].values.tolist()
                # self._raw_corpus.load(self.ngram_doc_path)
                print(f'{len(self._raw_corpus)} documents are loaded.')
                print("Elapsed time is %.2f seconds" % elapsed() )
        else:
            self.extract_ngrams()
            assert self.ngrams, 'Load a ngrams first'
            print('Building ngram docs by concatenaing words in ngram list')
            self._raw_corpus.concat_ngrams(self.ngrams, self.ngram_delimiter)
            self._raw_corpus.save(self.ngram_doc_path)

    def _load_stopwords(self):
        if self.stopwords_path.is_file():
            self.stopwords = load_wordlist(self.stopwords_path)
        else:
            if self._stopwords_path.is_file():
                self.stopwords = load_wordlist(self._stopwords_path)
            else:
                self.stopwords = ['.']
        save_wordlist(self.stopwords, self.stopwords_path)


    def _load_word_prior(self):
        if self.word_prior_path.is_file():
            with open(self.word_prior_path, "r") as fp:
                self.word_prior = json.load(fp)
            print(self.word_prior)
        else:
            if self._word_prior_path.is_file():
                with open(self._word_prior_path, "r") as fp:
                    self.word_prior = json.load(fp)
                print(self.word_prior)
            else:
                self.word_prior = {}
        with open(self.word_prior_path, 'w') as fp:
            json.dump(self.word_prior, fp, ensure_ascii=False, indent=4)

    def load_corpus(self, sample_ratio=None, reload=False, min_df=5):
        sample_ratio = sample_ratio if sample_ratio else self.sample_ratio
        if self.corpus and self.sample_ratio == sample_ratio and not reload:
            print('Corpus is already loaded w/ sample_ratio: {}'.format(sample_ratio))
            return True
        else:
            print('Start loading corpus w/ sample_ratio: {}'.format(sample_ratio))
        if not self._raw_corpus: 
            self._load_ngram_docs()
        self._load_stopwords()
        assert self._raw_corpus, 'Load ngram documents first'
        assert self.stopwords, 'Load stopwords first'
        if sample_ratio and sample_ratio < 1.0 and sample_ratio > 0.0:
            docs = random.sample(self._raw_corpus, 
                                 int(len(self._raw_corpus) * sample_ratio))
            self.sample_ratio = sample_ratio
        else:
            docs = self._raw_corpus
            self.sample_ratio = 1.0
        self.corpus = tp.utils.Corpus()
        self.corpus_keys = []

        n_skipped = 0
        for i_doc, doc in tqdm(enumerate(docs)):
            words =[w for w in doc if w not in self.stopwords and len(w) >= self.min_word_len]
            if len(words) > min_df:
                self.corpus.add_doc(words=words)
                self.corpus_keys.append(self._raw_corpus_keys[i_doc])
            else:
                print(f'Skipped - index={i_doc}, key={self._raw_corpus_keys[i_doc]}, words={list(doc)}')
                n_skipped += 1
        print(f'Total {i_doc-n_skipped+1} documents are loaded.')
        print(f'Total {n_skipped} documents are removed from the corpus.')
        df = pd.DataFrame(self.corpus_keys, columns=self.id_keys)
        df[self.id_keys].to_csv(self.corpus_key_path, header=True)

    def transform_topic_dists(self, input_dir, corpus_names, export_dir, compress=False,
        column_evals=[], aggregations=None, groupby_cols=['id']):

        print('Transform topic distributions by performing evals and aggregations')
        os.makedirs(os.path.abspath(export_dir), exist_ok=True)

        # corpus_paths = load_corpus_paths(input_dir, corpus_names, corpus_type='dataframe', corpus_filetype='csv')
        # for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
        #     print('Inferring topic distribution of documents from {}'.format(corpus_file))

        #     df = pd.read_csv(corpus_file, index_col=0)
        #     print(df.tail())
        #     for eval in column_evals:
        #         print('Performing eval: ', eval)
        #         df.eval(eval, inplace=True)
        #         print(df.tail())

        #     if aggregations:
        #         print('Aggregations: ', aggregations)
        #         # function_str = "lambda x: 'a' + x"    
        #         # fn = eval(function_str)
        #         df = df.groupby(groupby_cols).agg(aggregations)
        #         # df.columns = [col[0] for col in df.columns]
        #         df = df.reset_index()
        #         print(df.tail())

        #     corpus_filename = Path(corpus_file).name
        #     corpus_filename = corpus_filename[:corpus_filename.find('.csv')]
        #     filename = f'{corpus_filename}.csv' + ('.bz2' if compress else '')
        #     output_path = f'{export_dir}/{filename}'
        #     df.to_csv(output_path, header=True)
        #     print(f'Corpus is saved as {output_path}')

    def infer_corpus(self, input_dir, corpus_names, export_dir, compress=False, 
        iter=100, min_df=5, text_key='text', id_keys=['id', 'chunk_no'], num_workers=0):

        self._load_stopwords()
        assert self.stopwords, 'Load stopwords first'
        assert self.model, 'Model not found'
        print('Infer document out of the model')
        os.makedirs(os.path.abspath(export_dir), exist_ok=True)

        num_workers = num_workers if num_workers else 1

        def convert_to_token_list(row):
            doc = str(getattr(row, text_key))
            words =[w for w in doc.split() 
                if w not in self.stopwords and len(w) >= self.min_word_len]
            if len(words) > min_df:
                return words
            else:
                return None

        # corpus_paths = load_corpus_paths(input_dir, corpus_names, corpus_type='dataframe', corpus_filetype='csv')
        # for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
        #     print('Inferring topic distribution of documents from {}'.format(corpus_file))

        #     df = pd.read_csv(corpus_file, index_col=0)
        #     df.dropna(subset=[text_key], inplace=True)
        #     # df[text_key] = df.apply(convert_to_token_list, axis=1)
        #     # df = df.dropna(subset=[text_key]).reset_index(drop=True)
        #     # print(df.tail())
        #     tp_docs = []
        #     for row in df.itertuples():
        #         text = convert_to_token_list(row)
        #         doc = None
        #         if text:
        #             doc = self.model.make_doc(text)
        #         else:
        #             print('Empty text!\n', row)
        #             # df.drop(row.Index, inplace = True)
        #         if doc:
        #             tp_docs.append(doc)
        #         else:
        #             if text:
        #                 print(':::::::::::: Empty doc\n', text)
        #             df.drop(row.Index, inplace = True)
        #     df = df.dropna(subset=[text_key]).reset_index(drop=True)
        #     print(df.tail())
        #     # topic_distributions, ll = self.model.infer(docs)
        #     # for doc, topic_dist in zip(docs, topic_distributions):
        #     #     print(doc)
        #     #     print(topic_dist)

        #     # tp_corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer())
        #     # tp_corpus.process(df[text_key].to_list())
        #     topic_dists, ll = self.model.infer(tp_docs, workers=num_workers, iter=iter)
        #     # topic_dist = []
        #     # # print topic distributions of each document
        #     # for doc in inferred_corpus:
        #     #     topic_dist.append(doc.get_topic_dist())
        #     print(topic_dists[-1:])

        #     print(f'Total inferred: {len(topic_dists)}, from: {len(df.index)}')
        #     if len(topic_dists) == len(df.index): 
        #         idx = range(len(topic_dists[0]))
        #         df_infer = pd.DataFrame(topic_dists, columns=[f'topic{i}' for i in idx])
        #         # print(df_infer.tail())
        #         df_infer = pd.concat([df[id_keys], df_infer], axis=1)
        #         print(df_infer.tail())

        #         corpus_filename = Path(corpus_file).name
        #         corpus_filename = corpus_filename[:corpus_filename.find('.csv')]
        #         filename = f'{corpus_filename}.csv' + ('.bz2' if compress else '')
        #         output_path = f'{export_dir}/{filename}'
        #         df_infer.to_csv(output_path, header=True)
        #         print(f'Corpus is saved as {output_path}')
        #     else:
        #         print('The number of inferred is not same as the number of input.')

    def save_document_topic_dists(self):
        assert self.model, 'Model not found'

        topic_dists = []
        for doc in self.model.docs:
            topic_dists.append(doc.get_topic_dist())
        # print(topic_dists[-1:])

        if self.corpus_keys:
            df = pd.DataFrame(self.corpus_keys, columns=self.id_keys)
        elif self.corpus_key_path.is_file():
            df = pd.read_csv(self.corpus_key_path, index_col=0)
        else:
            print('Corpus keys do not exist')
            return

        print(f'Total inferred: {len(topic_dists)}, from: {len(df.index)}')
        if len(topic_dists) == len(df.index): 
            idx = range(len(topic_dists[0]))
            df_infer = pd.DataFrame(topic_dists, columns=[f'topic{i}' for i in idx])
            # print(df_infer.tail())
            df_infer = pd.concat([df[self.id_keys], df_infer], axis=1)
            print(df_infer.tail())

            filename = '{}-{}-topic_dists.csv'.format(self.model_name, self.active_model_id)
            output_path = f'{self.model_dir}/{filename}'
            df_infer.to_csv(output_path, header=True)
            print(f'Corpus is saved as {output_path}')
        else:
            print('The number of inferred is not same as the number of input.')

    def export_samples(self, corpus_names, topic_nums, model_id, 
        groupby_cols=['corpus_filename', 'id'],
        num_samples_per_topic = 100, num_tests_per_topic = 20, 
        num_exps_per_topic = 2, min_topic_ratio = 0.5):

        groupby_cols = list(groupby_cols)
        if model_id:
            self.active_model_id = model_id
        # model_type = self.active_model_id.split('.')[0]
        k = int(self.active_model_id.split('.')[1][1:])

        filename = '{}-{}-topic_dists.csv'.format(self.model_name, self.active_model_id)
        topic_dists_path = f'{self.model_dir}/{filename}'

        df_dists = pd.read_csv(topic_dists_path, index_col=0)

        topics = [f'topic{i}' for i in topic_nums]
        cols = [f'topic{i}' for i in range(k)]
        df_id = df_dists.groupby(groupby_cols)[cols].mean()
        df_id = df_id.reset_index()

        dfs_train = []
        dfs_test = []
        dfs_exp = []
        num_train = num_samples_per_topic - num_tests_per_topic

        for corpus_name in corpus_names: 

            df_cp = df_id.query('corpus_filename.str.startswith(@corpus_name)', engine='python')

            for topic in topics:
                df_sample = df_cp.query(f'{topic} > @min_topic_ratio', engine='python')
                n = len(df_sample.index) if len(df_sample.index) < num_samples_per_topic else num_samples_per_topic
                df_sample = df_sample.sample(n=n)
                dfs_train.append(df_sample[:num_train])
                dfs_test.append(df_sample[num_train:])
                dfs_exp.append(df_sample[:num_exps_per_topic]) 

        df_train = pd.concat(dfs_train)
        df_test = pd.concat(dfs_test)
        df_exp = pd.concat(dfs_exp)
        # df_train = df_train[groupby_cols]
        # df_train = df_train.merge(df_dists, on=groupby_cols)
        # df_test = df_test[groupby_cols]
        # df_test = df_test.merge(df_dists, on=groupby_cols)
        # df_exp = df_exp[groupby_cols]
        # df_exp = df_exp.merge(df_dists, on=groupby_cols)

        print(len(df_train.index), len(df_test.index), len(df_exp.index))

        if not self.labels:
            self.label_topics()
        labels = {'topic{}'.format(label['no']): label['name'] for label in self.labels}

        def top_topics(row):
            top_t = row[topics].sort_values(ascending = False).head(5)
            return ', '.join([f'{labels[s]}[{round(top_t[s]*100,0):.0f}%]' for s in top_t.index])

        top_topic_col = 'top_topics'
        out_cols  = groupby_cols[:] + [top_topic_col]
        df_train[top_topic_col] = df_train.apply(top_topics, axis=1)
        df_train = df_train[out_cols]
        df_test[top_topic_col] = df_test.apply(top_topics, axis=1)
        df_test = df_test[out_cols]
        df_exp[top_topic_col] = df_exp.apply(top_topics, axis=1)
        df_exp = df_exp[out_cols]

        filename = '{}-{}-train.csv'.format(self.model_name, self.active_model_id)
        export_path = f'{self.output_dir}/{filename}'
        df_train.to_csv(export_path, index=False)
        filename = '{}-{}-test.csv'.format(self.model_name, self.active_model_id)
        export_path = f'{self.output_dir}/{filename}'
        df_test.to_csv(export_path, index=False)
        filename = '{}-{}-exp.csv'.format(self.model_name, self.active_model_id)
        export_path = f'{self.output_dir}/{filename}'
        df_exp.to_csv(export_path, index=False)
        print(df_exp.head())


    def tune_params(self, model_type='LDA',
                    topics=[20], alphas=[0.1], etas=[0.01], sample_ratios=[0.1],
                    tws=[IDF], 
                    min_cf=5, rm_top=0, min_df=0, 
                    burn_in=0, interval = 10, iterations=100, seed=None,
                    eval_coherence=True,
                    save=False, save_full=False
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
        total_iters = (len(etas)*len(alphas)*len(topics)*len(tws)*len(sample_ratios))
        exec_dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_lls = None
        ys = []
        for i, (sr, k, a, e, tw) in tqdm(enumerate(
            itertools.product(sample_ratios, topics, alphas, etas, tws), 
            ), total=total_iters):
                            # train a model
            print(
                'sample_ratio: {}, k:{}, alpha:{}, eta:{}, tw:{}'.format(
                    sr, k, a, e, str(tw)
                )
            )
            # self.load_corpus(sample_ratio=sr)
            df_ll, m_sum = self.train_model(
                model_type=model_type,
                sample_ratio=sr,
                k=k, tw=tw,
                alpha=a, eta=e,
                min_cf=min_cf, rm_top=rm_top, 
                min_df=min_df, burn_in=burn_in,
                interval=interval, iterations=iterations,
                seed=seed, eval_coherence=eval_coherence,
                save=save, save_full=save_full
            )
            margs = []
            if len(topics)>1: margs.append('k={}'.format(k))
            if len(alphas)>1: margs.append('a={}'.format(a))
            if len(etas)>1: margs.append('e={}'.format(e))
            if len(tws)>1: margs.append('tw={}'.format(tw))
            if len(sample_ratios)>1: margs.append('sr={}'.format(sr))
            y = ','.join(margs) if len(margs)>0 else 'll_{}'.format(i)
            ys.append(y)
            
            df_ll.rename(columns={'ll_per_word': y}, inplace=True)
            if df_lls is not None:
                df_lls = df_lls.merge(df_ll, on='iter')
            else:
                df_lls = df_ll
                
        out_file = '{}-{}-ll_per_word-{}.csv'.format(
            self.model_name, model_type, exec_dt)
        out_file = str(self.model_dir/'output/tune'/out_file)
        df_lls.to_csv(out_file)
        ax = df_lls.plot(x ='iter', y=ys, kind = 'line')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        out_file = '{}-{}-ll_per_word-{}.png'.format(
            self.model_name, model_type, exec_dt)
        out_file = str(self.model_dir/'figures/tune'/out_file)
        plt.savefig(out_file, transparent=False, dpi=300)
    
    def train_model(self, model_id=None, model_type='LDA',
                    model_path=None, sample_ratio=None,
                    k=None, k1=None, k2=None,
                    tw=IDF, gamma=2, alpha=0.1, eta=0.01,
                    min_cf=5, rm_top=0, min_df=0, 
                    burn_in=0, interval = 10, iterations=100, seed=None,
                    eval_coherence=False, set_word_prior=False,
                    save=True, save_full=True):

        self.load_corpus(sample_ratio=sample_ratio)
        assert self.corpus, 'Load corpus first'

        exec_dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = model_type.upper()
        if model_id is None:
            margs = [model_type]
            if k: margs.append('k{}'.format(k))
            if k1: margs.append('k{}'.format(k1))
            if k2: margs.append('k{}'.format(k2))
            if self.sample_ratio < 1.0:
                margs.append('sr{}'.format(int(self.sample_ratio*100)))
            model_id = '.'.join(margs)
        self.active_model_id = model_id
            
        if not seed:
            random.seed()
            seed = random.randint(0, 32767)

        if model_type == 'LDA':
            mdl = tp.LDAModel(
                tw=tw, k=k, min_cf=min_cf, rm_top=rm_top, 
                alpha=alpha, eta=eta,
                corpus=self.corpus, seed=seed
            )
        elif model_type == 'HPA':
            mdl = tp.HPAModel(
                tw=tw, k1=k1, k2=k2, min_cf=min_cf, rm_top=rm_top, 
                alpha=alpha, eta=eta,
                corpus=self.corpus, seed=seed
            )
        elif model_type == 'HDP':
            mdl = tp.HDPModel(
                tw=tw, min_cf=min_cf, rm_top=rm_top, 
                gamma=gamma, alpha=alpha, initial_k=k, 
                corpus=self.corpus, seed=seed
            )
        elif model_type == 'CTM':
            mdl = tp.CTModel(
                tw=tw, k=k, min_cf=min_cf, rm_top=rm_top, 
                eta=eta, smoothing_alpha=alpha,
                corpus=self.corpus, seed=seed
            )
        else:
            print('{} is not supported'.format(model_type))
            return False

        if set_word_prior:
            self._load_word_prior()
            for tno, words in self.word_prior.items():
                print(f'Set words {words} to topic #{tno} as prior.')
                for word in words:
                    mdl.set_word_prior(word, [1.0 if i == int(tno) else 0.1 for i in range(k)])
        
        mdl.burn_in = burn_in
        mdl.train(0)
        print('Num docs:', len(mdl.docs), ', Vocab size:', mdl.num_vocabs,
            ', Num words:', mdl.num_words)
        print('Removed top words:', mdl.removed_top_words)

        print('Training model by iterating over the corpus {} times, {} iterations at a time'.format(
            iterations, interval))
        ll_per_words = []
        for i in range(0, iterations, interval):
            mdl.train(interval)
            if model_type == 'HDP':
                print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, mdl.ll_per_word, mdl.live_k))
            else:
                print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
            ll_per_words.append((i, mdl.ll_per_word))
        df_ll = pd.DataFrame(ll_per_words, columns=['iter', 'll_per_word'])
        out_file = '{}-{}-ll_per_word-{}.csv'.format(
            self.model_name, self.active_model_id, exec_dt)
        out_file = str(self.output_dir/'output/train'/out_file)
        df_ll.to_csv(out_file)
        ax = df_ll.plot(x ='iter', y='ll_per_word', kind = 'line')
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Log-likelihood per word")
        ax.invert_yaxis()
        out_file = '{}-{}-ll_per_word-{}.png'.format(
            self.model_name, self.active_model_id, exec_dt)
        out_file = str(self.output_dir/'figures/train'/out_file)
        plt.savefig(out_file, transparent=False, dpi=300)

        mdl.summary()
        self.model = mdl
        self.models[model_id] = mdl        

        coh_values = self.eval_coherence_value() if eval_coherence else {}
        # calculate coherence using preset

        original_stdout = sys.stdout
        out_file = '{}-{}-summary-{}.txt'.format(
            self.model_name, self.active_model_id, exec_dt)
        out_file = str(self.output_dir/'logs'/out_file)
        with open(out_file, 'w') as f:
            sys.stdout = f # Change the standard output to the file.
            mdl.summary()
            if coh_values:
                print('<Topic Coherence Scores>')
                for cm, cv in coh_values.items():
                    print(f'| {cm}: {cv}')
            sys.stdout = original_stdout # Reset the standard output.

        if model_path is None:
            model_path = '{}-{}.mdl'.format(self.model_name, model_id)
            model_path = self.model_dir/model_path
        else:
            model_path = Path(model_path)
        if save:
            mdl.save(str(model_path), full=save_full)

        self.save_document_topic_dists()

        entry = ModelSummary (
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
            round(mdl.ll_per_word,2),
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
            round(coh_values['u_mass'], 3) if 'u_mass' in coh_values else None,
            round(coh_values['c_uci'], 3) if 'c_uci' in coh_values else None,
            round(coh_values['c_npmi'], 3) if 'c_npmi' in coh_values else None,
            round(coh_values['c_v'], 3) if 'c_v' in coh_values else None,
        )
        self.summaries.append(entry)
        df = pd.DataFrame(self.summaries)
        df.to_csv(self.summary_file)
        return df_ll, entry

    def eval_coherence_value(self,  coherence_metrics=['u_mass', 'c_uci', 'c_npmi', 'c_v']):
        assert self.model, 'Model not found'
        mdl = self.model
        coh_values = {}
        for metric in coherence_metrics:
            coh = tp.coherence.Coherence(mdl, coherence=metric)
            average_coherence = coh.get_score()
            coh_values[metric] = average_coherence
            coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
            print('==== Coherence : {} ===='.format(metric))
            print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
            print()
        return coh_values
        
    def load_model(self, model_id=None, model_file=None, reload=False):
        if model_id:
            self.active_model_id = model_id
        if self.active_model_id in self.models and not reload:
            print('The model is already loaded.')
            return True
        
        if model_file is None:
            model_file = '{}-{}.mdl'.format(self.model_name, self.active_model_id)
        model_path = self.model_dir / model_file
        print('Loading a model from {}'.format(model_path))
        if model_path.is_file():
            if not self.active_model_id:
                self.active_model_id = model_path.stem.split('-')[-1]
            model_type = self.active_model_id.split('.')[0]
            model_path = str(model_path)
            with elapsed_timer() as elapsed:
                if model_type == 'LDA':
                    mdl = tp.LDAModel.load(model_path)
                elif model_type == 'HPA':
                    mdl = tp.HPAModel.load(model_path)
                elif model_type == 'HDP':
                    mdl = tp.HDPModel.load(model_path)
                else:
                    print('{} is not supported'.format(model_type))
                    return False
                self.models[self.active_model_id] = mdl
                self.model = mdl
                print("Elapsed time is %.2f seconds" % elapsed() )
        else:
            self.model = None
            print('Model file not found')


    def save_labels(self, names=None):
        if not self.labels:
            self.label_topics()
        if isinstance(names, dict):
            for k in names:
                self.labels[int(k)]['name'] = names[k]
        label_file = '{}-{}-labels.csv'.format(self.model_name, self.active_model_id)
        label_file = self.output_dir/label_file
        df = pd.DataFrame(self.labels)
        df.to_csv(label_file, index=False)        
                

    def label_topics(
        self, rebuild=False, use_pmiextractor=False, 
        min_cf=10, min_df=5, 
        max_len=5, max_cand=100,
        smoothing=1e-2, mu=0.25,
        window_size=100, top_n=10
    ):

        label_file = '{}-{}-labels.csv'.format(self.model_name, self.active_model_id)
        label_file = self.output_dir/label_file
        if label_file.is_file() and not rebuild:
            print('loading labels from {}'.format(label_file))
            df = pd.read_csv(label_file, index_col=None)
            self.labels = df.to_dict('records')
        else:
            assert self.model, 'Model not found'
            mdl = self.model
            if use_pmiextractor:
                # extract candidates for auto topic labeling
                print('extract candidates for auto topic labeling')
                extractor = tp.label.PMIExtractor(
                    min_cf=min_cf, min_df=min_df, 
                    max_len=max_len, max_cand=max_cand
                )
                with elapsed_timer() as elapsed:
                    cands = extractor.extract(mdl)
                    print("Elapsed time is %.2f seconds" % elapsed() )            
                    labeler = tp.label.FoRelevance(
                        mdl, cands, min_df=min_df, 
                        smoothing=smoothing, mu=mu,
                        window_size=window_size
                    )
                    print("Elapsed time is %.2f seconds" % elapsed() )
                self.labeler = labeler

            labels = []
            for k in range(mdl.k):
                print("== Topic #{} ==".format(k))
                name = f'Topic #{k}'
                if use_pmiextractor:
                    lbls = ','.join(label for label, score in labeler.get_topic_labels(k, top_n=top_n))
                    print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=top_n)))
                wrds = ','.join(word for word, prob in mdl.get_topic_words(k, top_n=top_n)) 
                if use_pmiextractor:
                    label = {'no': k, 'name': name, 'labels': lbls, 'words': wrds}
                else:
                    label = {'no': k, 'name': name, 'words': wrds}
                labels.append(label)
                for word, prob in mdl.get_topic_words(k, top_n=top_n):
                    print(word, prob, sep='\t')
                print()

            self.labels = labels
            df = pd.DataFrame(self.labels)
            df.to_csv(label_file, index=False)        

    def visualize(self):
        assert self.model, 'Model not found'
        mdl = self.model
        topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
        doc_topic_dists = np.stack([doc.get_topic_dist() 
                                    for doc in mdl.docs 
                                    if np.sum(doc.get_topic_dist()) == 1])
        doc_lengths = np.array([len(doc.words) 
                                for doc in mdl.docs 
                                if np.sum(doc.get_topic_dist()) == 1])
        vocab = list(mdl.used_vocabs)
        term_frequency = mdl.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists, 
            doc_topic_dists, 
            doc_lengths, 
            vocab, 
            term_frequency
        )
        out_file = '{}-{}-ldavis.html'.format(self.model_name, self.active_model_id)
        out_file = str(self.output_dir/'output'/out_file)
        pyLDAvis.save_html(prepared_data, out_file)

    def get_topic_words(self, top_n=10):
        '''Wrapper function to extract topics from trained tomotopy HDP model 
        
        ** Inputs **
        top_n: int -> top n words in topic based on frequencies
        
        ** Returns **
        topics: dict -> per topic, an arrays with top words and associated frequencies 
        '''
        assert self.model, 'Model not found'
        mdl = self.model
        
        # Get most important topics by # of times they were assigned (i.e. counts)
        sorted_topics = [k for k, v in sorted(enumerate(mdl.get_count_by_topics()), key=lambda x:x[1], reverse=True)]

        topics=dict()        
        # For topics found, extract only those that are still assigned
        for k in sorted_topics:
            if type(mdl) in ['tomotopy.HPAModel', 'tomotopy.HDPModel']:
                if not mdl.is_live_topic(k): continue # remove un-assigned topics at the end (i.e. not alive)
            topic_wp =[]
            for word, prob in mdl.get_topic_words(k, top_n=top_n):
                topic_wp.append((word, prob))

            topics[k] = topic_wp # store topic word/frequency array
            
        return topics

    def topic_wordclouds(
        self, topic_dict=None, 
        title_fontsize=20, title_color='green',
        top_n=100,
        ncols=5, nrows=1, dpi=300,
        save=True
    ):
        '''Wrapper function that generates wordclouds for ALL topics of a tomotopy model
        ** Inputs **
        topic_dic: dict -> per topic, an arrays with top words and associated frequencies
        save: bool -> If the user would like to save the images
        
        ** Returns **
        wordclouds as plots
        '''
        assert self.model, 'Model not found'
        # if topic_dict is None:
        #     topic_dict = self.get_topic_words()
        num_topics = self.model.k
        # wc = WordCloud(background_color="white")
        def save_fig():
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                                wspace=0.00, hspace=0.00)  # make the figure look better
            fig.tight_layout()
            out_file = '{}-{}-wc_topic_p{}.png'.format(self.model_name, self.active_model_id, p)
            out_file = str(self.output_dir/'figures/wc'/out_file)
            plt.savefig(out_file, transparent=True, dpi=dpi)
        
        fontname, _ = _get_font_name()
        plt.rcParams["font.family"] = fontname
        figsize=(nrows*4, ncols*5)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        cnt = 0
        p = 1
        for i in range(num_topics):
            r, c = divmod(cnt, ncols)
            k = i
            print(f'Creating topic wordcloud for Topic #{k}')
            self.create_wordcloud(k, fig, axes[r, c], top_n=top_n, save=False, fontname=fontname)
            if self.labels:
                topic_name = self.labels[k]['name']
                if topic_name.startswith('Topic #'):
                    topic_name = None
            else:
                topic_name = None
            if topic_name:
                title = f'Topic #{k} - {topic_name}'
            else:
                title = f'Topic #{k}'
            axes[r, c].set_title(title, fontsize=title_fontsize, color=title_color)
            cnt += 1
            if cnt == nrows*ncols:
                if save: save_fig()
                if i < num_topics-1:  
                    p += 1
                    cnt = 0
                    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if save and cnt < nrows*ncols: 
            while cnt < nrows*ncols:
                r, c = divmod(cnt, ncols)
                axes[r, c].set_visible(False)
                cnt += 1
            save_fig()

    def create_wordcloud(self, topic_idx, fig, ax, top_n=100, save=False, fontname=None):
        '''Wrapper function that generates individual wordclouds from topics in a tomotopy model
        
        ** Inputs **
        topic_idx: int -> topic index
        fig, ax: obj -> pyplot objects from subplots method
        save: bool -> If the user would like to save the images
        
        ** Returns **
        wordclouds as plots'''
        assert self.model, 'Model not found'
        mdl = self.model
        if not fontname:
            fontname, _ = _get_font_name()
        wc = WordCloud(font_path=fontname, background_color='white')
        
        topic_freq = dict(mdl.get_topic_words(topic_idx, top_n=top_n))
        img = wc.generate_from_frequencies(topic_freq)
        ax.imshow(img, interpolation='bilinear')
        ax.axis('off')        
        if save:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            out_file = '{}-{}-wc_topic_{}.png'.format(self.model_name, self.active_model_id, topic_idx)
            out_file = str(self.output_dir/'figures/wc'/out_file)
            plt.savefig(out_file, bbox_inches=extent.expanded(1.1, 1.2))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_mode', type=str, help='process mode')
    parser.add_argument('--corpus_dir', type=str, help='Location of corpus files')
    parser.add_argument('--model_dir', type=str, help='Location of model files')
    parser.add_argument('--model_type', type=str, help='Type of the model')
    parser.add_argument('--model_id', type=str, help='Name of the model')
    parser.add_argument('--corpus', type=str, help='Name of the corpus')
    parser.add_argument('--k', help='No. of topics', type=int, default=20)

    args = parser.parse_args()
    print(args)
    
    tm = TopicModel(model_name=args.corpus)
    if args.process_mode == "train-topic-model":
        tm.train_model(
            model_type=args.model_type, 
            k=args.k,
            burn_in=10,
            iterations=200
        )
        # tm.visualize()
    elif args.process_mode == "tune-topic-model":
        tm.tune_params(
            model_type=args.model_type,
            topics=[30], 
            alphas=[0.01, 0.1],
            etas=[0.01], 
            sample_ratios=[1.0],
            tws=[IDF], 
            min_cf=5, rm_top=0, min_df=0, 
            burn_in=0, interval = 50, iterations=1000, seed=None,
            eval_coherence=True,
            save=False, save_full=False
        )
    elif args.process_mode == "analyse-topic-model":
        tm.active_model_id =  args.model_id
        tm.load_model()
        # tm.label_topics()
        # tm.visualize()
        tm.topic_wordclouds()
    