import os
import codecs
import subprocess
from omegaconf import OmegaConf


def rsync(**cfg):
    args = OmegaConf.create(cfg)
    os.makedirs(args.output_dir, exist_ok=True)

    options = codecs.decode(args.options, "unicode_escape")
    cmd = ["rsync", options, args.source, args.output_dir]
    if not os.listdir(args.output_dir) or args.force_download:
        print(f" >> rsyncing {args.source} to {args.output_dir}")
        subprocess.run(" ".join(cmd), shell=True)
