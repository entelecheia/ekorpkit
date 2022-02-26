import os
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import codecs
import ekorpkit.config as config
from ekorpkit.utils.func import elapsed_timer, check_min_len, check_max_len
from functools import reduce
from hydra.utils import instantiate
from ekorpkit.utils.batch import decorator_apply
from functools import partial
from wasabi import msg
from collections import OrderedDict
from omegaconf.listconfig import ListConfig
from tqdm.auto import tqdm
from ekorpkit.utils import ordinal, print_status
from ekorpkit.io.file import get_filepaths, save_dataframe, load_dataframe


def apply(func, series, description=None, verbose=False, use_batcher=True, minibatch_size=None, **kwargs):
	batcher = config.batcher
	if use_batcher and  batcher is not None:
		batcher_minibatch_size = batcher.minibatch_size
		if minibatch_size is None:
			minibatch_size = batcher_minibatch_size
		if batcher.procs > 1:
			batcher.minibatch_size = min(int(len(series)/batcher.procs)+1, minibatch_size)
			if verbose:
				msg.info(f'Using batcher with minibatch size: {batcher.minibatch_size}')
			results = decorator_apply(func, batcher, description=description)(series)
			batcher.minibatch_size = batcher_minibatch_size
			return results

	if verbose and batcher is None:
		msg.warn('Warning: batcher not initialized')
	tqdm.pandas(desc=description)	
	return series.progress_apply(func)

def apply_pipe(df, pipe):
	fn = instantiate(pipe['method'], _recursive_=False)
	print(f'\nApplying pipe: {fn}')
	if isinstance(df, list):
		if 'concat_dataframes' in str(fn):
			return fn(df, pipe)
		else:
			dfs = []
			for df_no, df_each in enumerate(df):
				print(f'Applying pipe to dataframe {(df_no+1)}/{len(df)}')
				pipe['dataframe_no'] = df_no
				dfs.append(fn(df_each, pipe))
			return dfs
	else:
		return fn(df, pipe)
	# df = df.pipe(fn, pipe)
	# return df

def apply_pipeline(df, pipeline, pipeline_args, update_args={}, verbose=True):
	pipeline_targets = []
	pipes = OrderedDict()
	if isinstance(pipeline, (list, ListConfig)):
		for pipe in pipeline:
			pipes[pipe] = pipe
	elif isinstance(pipeline, str):
		pipes[pipeline] = pipeline
	else:
		pipes = pipeline
	if verbose:
		print(f'Applying pipeline: {pipes}')
	for pipe, pipe_arg_name in pipes.items():
		args = dict(pipeline_args.get(pipe_arg_name, {}))
		if args and isinstance(args, dict):
			args.update(update_args)
			pipeline_targets.append(args)

	return reduce(apply_pipe, pipeline_targets, df)

def dataframe_pipeline(**cfg):
	args = OmegaConf.create(cfg)
	pipeline_args = args.task.get('pipeline', {})
	process_pipeline = pipeline_args.get('_pipeline_', [])
	if process_pipeline is None:
		process_pipeline = []

	if len(process_pipeline) > 0:
		df = apply_pipeline(None, process_pipeline, pipeline_args)
		if df is not None:
			if isinstance(df, list):
				df = pd.concat(df)
			print(df.tail())
		else:
			print('No dataframe returned')


def eval_columns(df, args):
	verbose = args.get('verbose', False)
	expressions = args.get('expressions', None)
	if expressions is None:
		if verbose:
			print('No expressions specified')
		return df
	if verbose:
		print(f'Eval columns: {args}')
	for col, expr in expressions.items():
		df[col] = df.eval(expr)
	return df

def combine_columns(df, args):
	verbose = args.get('verbose', False)
	columns = args.get('columns', None)
	if columns is None:
		if verbose:
			print('No columns specified')
		return df
	if verbose:
		print(f'Combining columns: {args}')
	for col in columns:
		df[col].fillna('', inplace=True)
		df[col] = df[col].astype(str)
	separator = codecs.decode(args['separator'], 'unicode_escape')
	df[args['into']] = df[columns].agg(separator.join, axis=1)
	return df

def aggregate_columns(df, args):
	verbose = args.get('verbose', False)
	onto_column = args.get('onto', None)
	if onto_column is None:
		if verbose:
			print('No columns specified')
		return df
	groupby_cloumns =  args['groupby']
	if groupby_cloumns is None:
		if verbose:
			print('No groupby specified')
		return df
	if isinstance(groupby_cloumns, ListConfig):
		groupby_cloumns = list(groupby_cloumns)
	separator = codecs.decode(args['separator'], 'unicode_escape')

	num_docs = df.shape[0]
	if verbose:
		print(f'Aggregating columns: {args}')
	df[onto_column].fillna('', inplace=True)
	df = df.groupby(groupby_cloumns, as_index=False).agg({onto_column: separator.join})
	n_docs = df.shape[0]
	if verbose:
		print(f'{num_docs} documents aggregated into {n_docs} documents')
	return df

def explode_splits(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	separator = codecs.decode(args['separator'], 'unicode_escape')
	id_key = args.get('id_key', 'id')
	split_key = args.get('split_key', 'seg_id')

	if verbose:
		print(f'Exploding column: {args}')

	num_docs = df.shape[0]
	df[apply_to] = df[apply_to].str.split(separator)
	df = df.explode(apply_to)
	df[split_key] = df.groupby(id_key).cumcount()
	n_docs = df.shape[0]
	if verbose:
		print(f'{num_docs} documents exploded into {n_docs} documents')
	return df

def rename_columns(df, args):
	verbose = args.get('verbose', False)
	new_names = args.get('new_names', None)
	if new_names is None:
		if verbose:
			print('No columns specified')
		return df
	if verbose:
		print(f'Renaming columns: {args}')
	if new_names is not None:
		df.rename(columns=new_names, inplace=True)
	return df

def reset_index(df, args):
	verbose = args.get('verbose', False)
	index_column_name = args.get('index_column_name', 'id')
	if verbose:
		print(f'Resetting index: {args}')
	df = df.reset_index().rename(columns={'index': index_column_name})
	return df

def normalize(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	normalizer = args.get('normalizer', None)
	if normalizer is None:
		if verbose:
			print('No normalizer specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	if verbose:
		print(f'Normalizing text: {args}')
	if verbose:
		print('instantiating normalizer')
	normalizer = instantiate(normalizer)
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		with elapsed_timer(format_time=True) as elapsed:
			df[key] = apply(normalizer.normalize, df[key], description=f'Normalizing column: {key}', verbose=verbose)
			if verbose:
				msg.good('\n >> elapsed time to normalize: {}\n'.format(elapsed()))
	return df

def fillna(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	fill_with = args.get('fill_with', '')
	if verbose:
		print(f'Filling missing values: {args}')
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		df[key].fillna(fill_with, inplace=True)
	return df

def segment(df, args):
	verbose = args.get('verbose', False)
	use_batcher = args.get('use_batcher', True)
	minibatch_size = args.get('minibatch_size', None)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	segmenter = args.get('segmenter', None)
	if segmenter is None:
		if verbose:
			print('No segmenter specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	if verbose:
		print(f'Splitting text: {args}')
		print('instantiating segmenter')
	segmenter = instantiate(segmenter)
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		with elapsed_timer(format_time=True) as elapsed:
			df[key] = apply(
				segmenter.segment_article, df[key], 
				description=f'Splitting column: {key}', 
				verbose=verbose, use_batcher=use_batcher,
				minibatch_size=minibatch_size
			)
			if verbose:
				msg.good('\n >> elapsed time to segment: {}\n'.format(elapsed()))
	return df

def replace_whitespace(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	replace_with = args.get('replace_with', ' ')
	if verbose:
		print(f'Replacing whitespace with [{replace_with}]')
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		with elapsed_timer(format_time=True) as elapsed:
			df[key] = df[key].str.replace('\s+', replace_with)
			if verbose:
				msg.good('\n >> elapsed time to replace whitespace: {}\n'.format(elapsed()))
	return df

def replace_regex(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	patterns = args.get('patterns', {})
	if patterns is None:
		if verbose:
			print('No patterns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	if verbose:
		print(f'Replacing regex: {args}')
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		with elapsed_timer(format_time=True) as elapsed:
			for pat, repl in patterns.items():
				df[key] = df[key].str.replace(pat, repl, regex=True).str.strip()
			if verbose:
				msg.good('\n >> elapsed time to replace regex: {}\n'.format(elapsed()))
	return df

def remove_startswith(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	startswith = args.get('startswith', {})
	if startswith is None:
		if verbose:
			print('No startswith text specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	if verbose:
		print(f'Remove startswith: {args}')
	for key in apply_to:
		with elapsed_timer(format_time=True) as elapsed:
			for starting_text in startswith:
				print(f'Remove text starting with {starting_text} from [{key}]')
				idx = df[key].str.lower().str.startswith(starting_text, na=False)
				df.loc[idx, key] = df.loc[idx, key].str[len(starting_text) :].str.strip()
			if verbose:
				msg.good('\n >> elapsed time to remove startswith: {}\n'.format(elapsed()))
	return df


def filter_length(df, args, **kwargs):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	min_length = args.get('min_length', None)
	max_length = args.get('max_length', None)
	if min_length is None and max_length is None:
		if verbose:
			print('No length specified')
		return df
	len_func = args['method'].get('len_bytes', None)
	len_func = instantiate(len_func, _recursive_=False)
	_check_max_len = partial(check_max_len, max_len=max_length, len_func=len_func)
	_check_min_len = partial(check_min_len, min_len=min_length, len_func=len_func)

	if verbose:
		print(f'Filtering by length: {args}')
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		with elapsed_timer(format_time=True) as elapsed:
			if min_length and min_length > 0:
				n_docs = df.shape[0]
				idx = apply(_check_min_len, df[key].astype(str), verbose=verbose, description=f'min length: {min_length}')
				df = df[idx]
				if verbose:
					print(f'{(n_docs-df.shape[0])} documents removed due to length is less than {min_length}')
			if max_length and max_length > 0:
				n_docs = df.shape[0]
				idx = apply(_check_max_len, df[key].astype(str), verbose=verbose, description=f'max length: {max_length}')
				df = df[idx]
				if verbose:
					print(f'{(n_docs-df.shape[0])} documents removed due to length is greater than {max_length}')

	return df

def filter_query(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	query = args.get('query', None)
	if query is None:
		if verbose:
			print('No query specified')
		return df

	if verbose:
		print(f'Filtering by qeury: {args}')
	for key in apply_to:
		if verbose:
			print(f'\nPreprocessing column: {key}')
		with elapsed_timer(format_time=True) as elapsed:
			n_docs = df.shape[0]
			df = df.query(query, engine='python')
			if verbose:
				print(f'filtered {df.shape[0]} out of {n_docs} documents by {query}')

	return df


def drop_duplicates(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', None)
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	if verbose:
		print(f'Dropping duplicates: {args}')
	with elapsed_timer(format_time=True) as elapsed:
		for key in apply_to:
			num_docs = df.shape[0]
			df = df.drop_duplicates(subset=[key])
			n_docs = df.shape[0]
			if verbose:
				print(f'{n_docs} documents after dropping {(num_docs-n_docs)} duplicates from [{key}]')
		if verbose:
			msg.good('\n >> elapsed time to drop duplicates: {}\n'.format(elapsed()))
	return df

def save_samples(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	sample_length_to_print = args.get('sample_length_to_print', 1000)
	if apply_to is None:
		if verbose:
			print('No columns specified')
		return df
	if isinstance(apply_to, str):
		apply_to = [apply_to]
	num_samples_to_save = args.get('num_samples_to_save', None)
	smaple_file_prefix = args.get('sample_file_prefix', 'sample')
	if verbose:
		print(f'Saving samples: {args}')

	sample_separator = '-' * 100 + '\n'
	df_sample = df.sample(num_samples_to_save)[apply_to]
	sample_text = ''
	print_text = ''
	for i, row in df_sample.iterrows():
		for key in apply_to:
			stext = row[key]
			if len(stext) > sample_length_to_print:
				ptext = stext[:sample_length_to_print] + '...'
			else:
				ptext = stext
			sample_text += key + ': \n' + stext + '\n\n'
			print_text += key + ': \n' + ptext + '\n\n'
		sample_text += sample_separator
		print_text += sample_separator
	sample_text = sample_text.strip()
	print_text = print_text.strip()
	sample_file = smaple_file_prefix + '.txt'
	open(sample_file, 'w', encoding='utf-8').write(sample_text)

	if verbose:
		print(sample_separator)
		print(print_text)
		print(f'Saved {num_samples_to_save} samples to {sample_file}')

	return df

def save_as_text(df, args):
	verbose = args.get('verbose', False)
	apply_to = args.get('apply_to', 'text')
	corpus_name = args.get('corpus_name', 'corpus')
	output_dir = args.get('output_dir', '.')
	output_file = args.get('output_file', None)
	doc_separator = args.get('doc_separator', '\n\n')
	if isinstance(apply_to, (list, ListConfig)):
		apply_to = apply_to[0]

	os.makedirs(os.path.abspath(output_dir), exist_ok=True)
	if output_file:
		filename = output_file
	else:
		filename = f'{corpus_name}.txt'
	output_file_path = f'{output_dir}/{filename}'

	with elapsed_timer(format_time=True) as elapsed:
		n_loaded = len(df.index)
		doc_separator = str(doc_separator).encode('utf-8').decode('unicode_escape')
		with open(output_file_path, 'w') as fo:
			fo.write(
				doc_separator.join(df[apply_to].dropna().tolist())
			)
		print(f'Corpus is exported to {output_file_path}')
		n_sampled = len(df.index)
		status = [[' x ', corpus_name, n_loaded, n_sampled, elapsed(), filename]]

	if verbose:
		print_status(status)
	return df

def split_dataframe(df, args):
	verbose = args.get('verbose', False)
	num_splits = args.get('num_splits', 1)
	if num_splits <= 1:
		return df
	if verbose:
		print(f'Splitting dataframe into {num_splits} splits')
	return np.array_split(df, num_splits)

def concat_dataframes(dfs, args):
	verbose = args.get('verbose', False)
	if isinstance(dfs, list):
		if verbose:
			print(f'Concatenating {len(dfs)} dataframes')
		return pd.concat(dfs)
	else:
		if verbose:
			print('Returning original dataframe')
		return dfs

def save_metadata(df, args):
	verbose = args.get('verbose', False)
	filepath = args.get('filepath', None)
	filetype = args.get('filetype', None)
	column_info = args.get('column_info', None)
	split_name = args.get('split_name', None)

	if verbose:
		print(f'Saving metadata: {args}')

	meta_info = column_info.get('meta', None)
	if isinstance(meta_info , (dict, DictConfig)):
		meta_columns = list(meta_info.keys())		
		if 'split' in meta_columns and 'split' not in df.columns:
			df['split'] = split_name
		df_meta = df[meta_columns]
		save_dataframe(df_meta, filepath, filetype, verbose)

	data_info = column_info.get('data', None)
	if isinstance(data_info , (dict, DictConfig)):
		data_columns = list(data_info.keys())
		if 'split' in data_columns and 'split' not in df.columns:
			df['split'] = split_name
		df = df[data_columns]

	return df

def save_dataframe_pipe(df, args):
	verbose = args.get('verbose', False)
	filepath = args.get('filepath', None)
	filetype = args.get('filetype', None)
	corpus_name = args.get('corpus_name', 'corpus')
	output_dir = args.get('output_dir', '.')
	output_file = args.get('output_file', None)
	dataframe_no = args.get('dataframe_no', None)

	if df is None:
		msg.warn('Dataframe is None')
		return df
	if verbose:
		print(f'Saving dataframe: {args}')

	if filepath:
		output_dir = os.path.dirname(filepath)
		output_file = os.path.basename(filepath)
	if output_file:
		fileinfo = os.path.splitext(output_file)
		filename = fileinfo[0]
		if not filetype:
			filetype = fileinfo[1] if len(fileinfo) > 1 else 'csv'
	else:
		filename = f'{corpus_name}'
		if not filetype:
			filetype = 'csv'
	filetype = '.' + filetype.replace('.', '')
	if dataframe_no is not None:
		filename = f'{filename}-{dataframe_no:0>3d}{filetype}'
	else:
		filename = f'{filename}{filetype}'
	filepath = f'{output_dir}/{filename}'
	
	save_dataframe(df, filepath, filetype, verbose)
	return df


def load_dataframe_pipe(df=None, args=None):
	if args is None:
		raise ValueError('args must be specified')
	verbose = args.get('verbose', False)
	filepath = args.get('filepath', None)
	data_dir = args.get('data_dir', None)
	data_file = args.get('data_file', None)

	if filepath:
		filepaths = get_filepaths(filepath)
	else:
		filepaths = get_filepaths(data_file, data_dir)
	if verbose:
		print(f'Loading {len(filepaths)} dataframes from {filepaths}')
	if len(filepaths) == 1:
		return load_dataframe(filepaths[0], verbose=verbose)
	else:
		df = pd.concat([load_dataframe(f, verbose=verbose) for f in filepaths])
		return df


def save_as_json(df, args):
	verbose = args.get('verbose', False)
	corpus_name = args.get('corpus_name', 'corpus')
	output_dir = args.get('output_dir', '.')
	output_file = args.get('output_file', None)
	force_ascii = args.get('force_ascii', False)

	os.makedirs(os.path.abspath(output_dir), exist_ok=True)
	if output_file:
		filename = output_file
	else:
		filename = f'{corpus_name}.json'
	output_file_path = f'{output_dir}/{filename}'

	df.to_json(output_file_path, orient='records', lines=True, force_ascii=force_ascii)
	if verbose:
		print(f'Corpus is exported to {output_file_path}')
	return df