import pandas as pd
from omegaconf import OmegaConf
from ekorpkit.utils.func import ordinal, elapsed_timer
from .dataset import Dataset


class Datasets:
	def __init__(self, **args):
		args = OmegaConf.create(args)
		self.args = args
		self.names = args.name
		if isinstance(self.names, str):
			self.names = [self.names]
		self.data_dir = args.data_dir
		self.split_info = self.args.splits
		self.datasets = {}
		self._id_key = 'id'
		self._id_separator = "_"

		with elapsed_timer(format_time=True) as elapsed:
			for name in self.names:
				print(f'processing {name}')
				data_dir = f'{self.data_dir}/{name}'
				args['data_dir'] = data_dir
				args['name'] = name
				dataset = Dataset(**args)
				self.datasets[name] = dataset
			print(f"\n >>> Elapsed time: {elapsed()} <<< ")

	def concat_datasets(self):
		self.splits = {}
		for split in self.split_info:
			self.splits[split] = pd.concat(
					[ds.splits[split] for ds in self.datasets.values()], 
					ignore_index=True
				)


# class DatasetLoader:
# 	def __init__(self, **args):
# 		self.args = OmegaConf.create(args)
# 		self.datasets = {}
# 		self.load()

# 	def load(self):
# 		print('loading dataset')
# 		os.makedirs(self.args.cache_dir, exist_ok=True)
# 		os.makedirs(self.args.archive_dir, exist_ok=True)

# 		self.datasets = load_archive(self.args)
# 		if len(self.datasets) == 0:
# 			args = self.args
# 			self.datasets = hydra.utils.instantiate(args.loader, _recursive_=False)
# 			for pp in args.postprocess:
# 				pp_args = args.postprocess[pp] 
# 				if pp_args:
# 					print(f'\npostprocessing: {pp} with {pp_args}\n')
# 					getattr(self, pp)(pp_args)

# 			for name in self.datasets:
# 				df = self.datasets[name]
# 				file_path = f'{args.archive_dir}/{name}.csv'
# 				df.to_csv(file_path, index=False)
# 				if args.verbose:
# 					print(f'{name}: {df.shape}')
# 					print(df.head())
# 					print(df.dtypes)
# 					print(f'[{name}] is archived to {file_path}\n')

# 	def pretokenize(self, args):
# 		datasets = self.datasets
# 		for name in datasets:
# 			df = datasets[name]
# 			df = tokenize_dataframe(df, **args)
# 			df = extract_tokens_dataframe(df, **args)

# 			print(df.tail())
# 			print(f'{len(df.index)} documents are tokenized.')
# 			datasets[name] = df

# 		self.datasets = datasets

# 	def split_test_dev(self, args):
# 		from sklearn.model_selection import train_test_split
# 		datasets = self.datasets
# 		if args.ratio.test > 0 and args.ratio.test < 1:
# 			train, test = train_test_split(
# 				datasets['train'], 
# 				test_size=args.ratio.test, 
# 				random_state=args.random_state, 
# 				shuffle=args.shuffle)
# 			datasets['train'] = train
# 			datasets['test'] = test
# 		if args.ratio.dev > 0 and args.ratio.dev < 1:
# 			train, dev = train_test_split(
# 				datasets['train'], 
# 				test_size=args.ratio.dev, 
# 				random_state=args.random_state, 
# 				shuffle=args.shuffle)
# 			datasets['train'] = train
# 			datasets['dev'] = dev		
# 		self.datasets = datasets


# def load_archive(args):
# 	args = OmegaConf.create(args)
# 	# args = args.loader
# 	datasets = {}
# 	if args.force_download:
# 		return datasets
	
# 	for file in os.listdir(args.archive_dir):
# 		if file.endswith('.csv'):
# 			file_path = f'{args.archive_dir}/{file}'
# 			df = pd.read_csv(file_path, index_col=None)
# 			if args.column_dtypes:
# 				df = df.astype(dict(args.column_dtypes), errors='raise')
# 			name = file.split('.')[0]
# 			datasets[name] = df
# 			if args.verbose:
# 				print(f'{name}: {df.shape}')
# 				print(df.head())
# 				print(df.dtypes)
# 				print(f'[{name}] is loaded from archived file {file_path}\n')

# 	return datasets

# def load_local_datasets(**cfg):
# 	args = OmegaConf.create(cfg)
# 	# args = args.loader
# 	datasets = {}
# 	for name, file in args.source_files.items():
# 		file_path = f'{args.source_dir}/{file}'
# 		datasets[name] = globals()[args.reader._target_](file_path, **args)
# 	return datasets


# def load_gdrive_datasets(**cfg):
# 	args = OmegaConf.create(cfg)
# 	os.makedirs(args.cache_dir, exist_ok=True)
# 	# args = args.loader 
# 	datasets = {}
# 	for name, url in args.source_urls.items():
# 		file_path = f'{args.cache_dir}/{args.source_files[name]}'
# 		google_drive_download(url, file_path, name, args.force_download)
# 		datasets[name] = globals()[args.reader._target_](file_path, **args)
# 	return datasets


# def load_hf_dataset(**args):
# 	from datasets import load_dataset

# 	args = OmegaConf.create(args)
# 	# print(args)
# 	# args = args.loader 
# 	datasets = {}
# 	dsets = load_dataset(args.name, args.subset)
# 	datasets = {}
# 	for name in dsets:
# 		dset = dsets[name]
# 		for old, new in args.column_mapping.items():
# 			if new is None:
# 				dset = dset.remove_columns(old)
# 			elif old != new:
# 				dset = dset.rename_column(old, new)
# 		df = dset.to_pandas()
# 		split_name = args.split[name]
# 		datasets[split_name] = df
# 	return datasets


