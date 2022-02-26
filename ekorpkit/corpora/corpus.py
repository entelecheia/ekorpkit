import os
import codecs
from pathlib import Path
from pprint import pprint
# from hydra.utils import instantiate
from omegaconf import OmegaConf
# from omegaconf.dictconfig import DictConfig

from wasabi import msg
# from ekorpkit.utils.func import ordinal, elapsed_timer
from ekorpkit.io.file import load_dataframe
import pandas as pd
# import swifter


DESCRIPTION = "Corpus for Language Models"
LICENSE = "Copyright of the corpus is owned by the authors."

class Corpus():

	def __init__(self, **args):
		self.args = OmegaConf.create(args)
		self.name = self.args.name
		self.data_dir = Path(self.args.data_dir)
		self.info_file = self.data_dir / f'info-{self.name}.yaml'        
		self.info = OmegaConf.load(self.info_file) if self.info_file.is_file() else {}
		if self.info:
			self.args = OmegaConf.merge(self.args, self.info)
		self.verbose = self.args.get('verbose', False)
		self.autoload = self.args.get('autoload', False)

		if self.verbose:
			msg.info(f'Intantiating a corpus {self.name} with a config:')
			pprint(OmegaConf.to_container(self.args))

		self.filetype = self.args.filetype
		self.data_files = self.args.data_files
		self.meta_files = self.args.get('meta_files', None)
		if self.data_files is None:
			raise ValueError("Column info can't be None")

		self.segment_separator = self.args.get('segment_separator', '\n\n')
		self.sentence_separator = self.args.get('sentence_separator', '\n')
		self.segment_separator = codecs.decode(self.segment_separator, 'unicode_escape')
		self.sentence_separator = codecs.decode(self.sentence_separator, 'unicode_escape')
		self.description = self.args.description
		self.license = self.args.license
		self.column_info = OmegaConf.to_container(self.args.column_info)
		self.split_info = OmegaConf.to_container(self.args.splits)
		if self.column_info is None:
			raise ValueError("Column info can't be None")
		self._keys = self.column_info['keys']
		
		for k in ['id', 'text']:
			if isinstance(self._keys[k], str):
				self._keys[k] = [self._keys[k]]
			else:
				self._keys[k] = list(self._keys[k])

		self._text_key = 'text'
		self._id_key = 'id'
		self._id_separator = "_"
		self._data_keys = self.column_info['data']
		self._meta_kyes = self.column_info.get('meta', None)

		self._data = None
		self._metadata = None

		if self.autoload:
			self.load()
			self.load_metadata()

	# @classmethod
	# def exists(cls, corpus_dir=None, filename_pattern='.'):
	#     paths = get_corpus_paths(corpus_dir_or_paths=corpus_dir, filename_pattern=filename_pattern)
	#     return len(paths) > 0
	@property
	def data(self):
		return self._data

	@property
	def metadata(self):
		return self._metadata

	@property
	def num_rows(self) -> int:
		"""Number of rows in the dataset (same as :meth:`Dataset.__len__`)."""
		if self._data.index is not None:
			return len(self._data.index)
		return len(self._data)

	# def __repr__(self):
	# 	return f"Dataset({{\n    features: {list(self.features.keys())},\n    num_rows: {self.num_rows}\n}})"

	def load(self):
		dfs = []
		_text_keys = self._keys['text']
		_id_keys = self._keys['id']

		if len(_text_keys) > 1:
			self._data_keys = {k:v for k,v in self._data_keys.items() if k not in _text_keys}
			self._data_keys[self._text_key] = 'str'
		
		for split, data_file in self.data_files.items():
			data_file = self.data_dir / data_file
			df = load_dataframe(data_file)

			df[_text_keys] = df[_text_keys].fillna('')
			if len(_text_keys) > 1:
				df[self._text_key] = df[_text_keys].apply(
					lambda row: self.segment_separator.join(row.values.astype(str)), axis=1)
			
			_id_prefix = f'{split}_' if len(self.data_files) > 1 else ''
			if len(_id_keys) > 1 or len(self.data_files) > 1:
				df[self._id_key] = df[_id_keys].apply(
					lambda row: _id_prefix + self._id_separator.join(row.values.astype(str)), axis=1)
			dfs.append(df[list(self._data_keys.keys())])
		self._data = pd.concat(dfs)
		print(self._data.head(3))
		print(self._data.tail(3))

	def load_metadata(self):
		if self.meta_files is None:
			return
		dfs = []
		_id_keys = self._keys['id']
		for split, data_file in self.meta_files.items():
			data_file = self.data_dir / data_file
			df = load_dataframe(data_file)
			
			_id_prefix = f'{split}_' if len(self.data_files) > 1 else ''
			if len(_id_keys) > 1 or len(self.data_files) > 1:
				df[self._id_key] = df[_id_keys].apply(
					lambda row: _id_prefix + self._id_separator.join(row.values.astype(str)), axis=1)
			dfs.append(df)
		self._metadata = pd.concat(dfs)
		print(self._metadata.head(3))
		print(self._metadata.tail(3))

