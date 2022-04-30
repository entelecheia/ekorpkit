from pathlib import Path
from ekorpkit import eKonf
from hydra.utils import instantiate

from ..datasets import ShardingDataset
from ..models import train_tokenizer
from ..utils.func import elapsed_timer


def lmdata(**args):
    args = eKonf.to_config(args)

    with elapsed_timer(format_time=True) as elapsed:
        for subtask, task_args in args.task.lmdata.items():
            if "_target_" in task_args:
                instantiate(task_args, parent=args, _recursive_=False)
            else:
                print(f"{subtask} is not a valid subtask")
        print(f">>> Elapsed time: {elapsed()} <<< ")


def build_vocab(**args):
    cfg = eKonf.to_config(args)
    # print(cfg)
    args = cfg.parent.tokenizer

    if isinstance(args.input_dir, str):
        input_dirs = [args.input_dir]
    else:
        input_dirs = list(args.input_dir)

    if args.train_files is not None:
        # train_files = {args.input_dir + '/' + f:d for f, d in args.train_files.items()}
        train_files = []
        for input_dir in input_dirs:
            train_files = [input_dir + "/" + f for f in args.train_files]
    else:
        # train_files = {str(f):1 for f in Path(args.input_dir).glob('**/*') if f.is_file()}
        train_files = []
        for input_dir in input_dirs:
            train_files += [str(f) for f in Path(input_dir).glob("**/*") if f.is_file()]
    train_tokenizer(train_files=train_files, args=args, output_dir=args.output_dir)


def sharding(**args):
    args = eKonf.to_config(args)
    # print(cfg)

    sharding = ShardingDataset(**args)
    sharding.load_datasets()
    sharding.segment_articles_into_sentences()
    sharding.distribute_datasets_over_shards()
    sharding.write_shards_to_disk()
