from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from hydra.utils import instantiate
from ..utils.func import elapsed_timer
from wasabi import msg		


def corpus_tasks(**cfg):
	args = OmegaConf.create(cfg)
	corpus_cfg = args.corpus
	subtasks = args.task.corpus
	_subtasks_ = args.task.corpus._subtasks_

	with elapsed_timer(format_time=True) as elapsed:
		corpora = instantiate(corpus_cfg, _recursive_=False)

		for corpus in corpora:
			print(f'::: processing {corpus.name}')
			for subtask in _subtasks_:
				subtask_cfg = subtasks[subtask]
				if '_target_' in subtask_cfg:
					msg.info(f'Running {subtask} ...')
					instantiate(subtask_cfg, corpus=corpus, _recursive_=False)
				else:
					msg.fail(f'{subtask} is not a valid subtask')
		print(f"\n >>> Elapsed time: {elapsed()} <<< ")


def transfomer_finetune(**cfg):
	args = OmegaConf.create(cfg)
	model_cfg = args.model.transformer.finetune
	dataset_cfg = args.dataset

	if model_cfg._target_:
		model = instantiate(model_cfg, dataset_cfg=dataset_cfg, _recursive_=False)
		model.apply_pipeline()
	else:
		raise Exception('Model instantiation target is missing')		

