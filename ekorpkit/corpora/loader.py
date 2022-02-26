from hydra.utils import instantiate
from wasabi import msg		
from ekorpkit.pipelines.pipe import apply_pipeline
import pandas as pd
from omegaconf import OmegaConf
from ekorpkit.utils.func import ordinal, elapsed_timer
from .corpus import Corpus


class Corpora:
    def __init__(self, **args):
        args = OmegaConf.create(args)
        self.args = args
        self.names = args.name
        if isinstance(self.names, str):
            self.names = [self.names]
        self.data_dir = args.data_dir
        self.corpora = {}
        self._data = None

        with elapsed_timer(format_time=True) as elapsed:
            for name in self.names:
                print(f'processing {name}')
                data_dir = f'{self.data_dir}/{name}'
                args['data_dir'] = data_dir
                args['name'] = name
                corpus = Corpus(**args)
                self.corpora[name] = corpus
            print(f"\n >>> Elapsed time: {elapsed()} <<< ")

    def concat_corpora(self):
        self._data = pd.concat(
                [corpus._data for corpus in self.corpora.values()], 
                ignore_index=True
            )

    def __iter__(self):
        for corpus in self.corpora.values():
            yield corpus


def do_corpus_tasks(
		corpus, 
		pipeline=None,
		**kwargs
	):
	verbose = kwargs.get('verbose', False)
	merge_metadata = kwargs.get('merge_metadata', False)
	if merge_metadata:
		df = pd.merge(corpus._metadata, corpus._data, on=corpus._id_key)
	else:
		df = corpus._data
	update_args = {'corpus_name': corpus.name}
	_pipeline_ = pipeline.get('_pipeline_', {})
	df = apply_pipeline(df, _pipeline_, pipeline, update_args=update_args)


# class eKorpkit:
#     """
#     Examples::
#         >>> from eKorpkit import eKorpkit
#         >>> corpus = eKorpkit.load('esg_reports')
#         >>> len(corpus.train.texts)
#     """
#     @classmethod
#     def load(cls, name, corpus_dir=None, 
#             filename_pattern=None, load_light=False, 
#             id_keys=ID_KEYS, text_key=TEXT_KEY, meta_cols=None,
#             **kwargs
#             ):
        
#         id_keys = id_keys if id_keys else ID_KEYS
#         text_key = text_key if text_key else TEXT_KEY

#         print(f'Loading corpus [{name}] from [{corpus_dir}]')

#         corpus = DocCorpus(
#                     name=name, corpus_dir=corpus_dir, 
#                     filename_pattern=filename_pattern,
#                     load_light=load_light, 
#                     id_keys=id_keys, text_key=text_key, meta_cols=meta_cols
#                 )
#         return corpus

#     @classmethod
#     def exists(cls, name, 
#             corpus_dir=None, 
#             filename_pattern='.', 
#             ):
                
#         paths = get_corpus_paths(corpus_dir, filename_pattern)
#         return len(paths) > 0


# def load_corpus_paths(corpus_names, corpus_dir, 
#         corpus_filetype=None, filename_pattern=None, 
#         **kwargs):

#     corpus_dir = str(corpus_dir)
#     if isinstance(corpus_names, str):
#         corpus_names = [corpus_names]
#     else:
#         corpus_names = list(corpus_names)
#     available = []
#     if not corpus_filetype:
#         corpus_filetype = 'csv'            
#     if corpus_names[0] == CORPUS_ALL:
#         if not filename_pattern:
#             filename_pattern = '.'
#         for corpus_file in get_corpus_paths(corpus_dir, filename_pattern=filename_pattern, corpus_filetype=corpus_filetype):
#             available.append((CORPUS_ALL, str(corpus_file)))
#     else:
#         for corpus_name in corpus_names:
#             if not filename_pattern or filename_pattern == '.':
#                 f_pattern = corpus_name
#             else:
#                 if filename_pattern.startswith(corpus_name):
#                     f_pattern = filename_pattern
#                 else:
#                     f_pattern = corpus_name + filename_pattern
#             for corpus_file in get_corpus_paths(corpus_dir, filename_pattern=f_pattern, corpus_filetype=corpus_filetype):
#                 available.append((corpus_name, str(corpus_file)))
#     if not available:
#         raise ValueError(
#             'Not found any proper corpus name. Check the `corpus` argument')
#     print(f'{len(available)} corpus files are found.')
#     for path in available:
#         print(path)
#     return sorted(available)
