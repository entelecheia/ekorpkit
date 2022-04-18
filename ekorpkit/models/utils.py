import os
import glob
import orjson as json
from pathlib import Path
from hydra.utils import instantiate
from ekorpkit import eKonf

from ..utils.convert import convert_tf_checkpoint_to_pytorch


def convert_electra_ckpt_to_pytorch(**cfg):
    cfg = eKonf.to_config(cfg)

    model_config = eKonf.to_dict(cfg.model.electra, resolve=True)
    args = cfg.task.train.convert_electra
    tokenizer_config = dict(args.tokenizer)

    print(f"Converting tensorflow checkpoints to pytoch")
    os.makedirs(args.torch_output_dir, exist_ok=True)
    ckpt_file = f"{args.checkpoints_dir}/checkpoint"
    org_contents = open(ckpt_file, "r").read()

    if args.checkpoint_files:
        if isinstance(args.checkpoint_files, str):
            args.checkpoint_files = [args.checkpoint_files]
        checkpoint_files = args.checkpoint_files
    else:
        checkpoint_files = [
            Path(f).stem for f in glob.glob(f"{args.checkpoints_dir}/*.index")
        ]

    for ckpt in checkpoint_files:

        print("processing checkpoint: {}".format(ckpt))
        ckpt_contents = f'model_checkpoint_path: "{ckpt}"\n'
        open(ckpt_file, "w").write(ckpt_contents)
        for model_type in ["discriminator", "generator"]:
            torch_config = model_config[model_type]
            output_dir = f"{args.torch_output_dir}/{ckpt}/{model_type}"
            os.makedirs(output_dir, exist_ok=True)
            # create config
            config_file = f"{output_dir}/config.json"
            with open(config_file, "w") as f:
                json.dump(torch_config, f, ensure_ascii=False, indent=4)
            with open(f"{output_dir}/tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, ensure_ascii=False, indent=4)
            # copy vocab
            os.system(f"cp -f {args.vocab_file} {output_dir}/vocab.txt")
            # model path
            pytorch_dump_path = f"{output_dir}/pytorch_model.bin"
            print(f"Saving {model_type} to {pytorch_dump_path}")
            convert_tf_checkpoint_to_pytorch(
                args.checkpoints_dir,
                config_file,
                pytorch_dump_path,
                model_type,
                args.load_weights_func._target_,
            )
    # restore original contents
    open(ckpt_file, "w").write(org_contents)
