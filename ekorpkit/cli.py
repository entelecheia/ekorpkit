import os
import logging
import hydra
from ekorpkit import eKonf
from .tasks.info import make_table


log = logging.getLogger(__name__)


def cmd(**args):
    eKonf.run(args)


def listup(**args):
    ekorpkit_dir = eKonf.__ekorpkit_path__
    corpus_info = {}
    for name in args["corpus"]["preset"]["corpus"]:
        info_path = f"{ekorpkit_dir}/resources/corpora/{name}.yaml"
        info = eKonf.load(info_path)
        corpus_info[name] = info
    make_table(corpus_info.values(), args["info"]["table"])


def about(**args):
    from . import __version__

    print()
    for k, v in args.items():
        print(f"{k:11} : {v}")
    print(f"{'version':11} : {__version__}")
    print("\nExecute `ekorpkit --help` to see what eKorpkit provides")


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg) -> None:
    verbose = cfg.verbose
    if verbose:
        log.info("## eKorpkit Command Line Interface for Hydra ##")
    if cfg.get("print_config"):
        log.info("## hydra configuration ##")
        print(eKonf.to_yaml(cfg))

    if cfg.get("print_resolved_config"):
        log.info("## hydra configuration resolved ##")
        eKonf.pprint(cfg)

    if verbose:
        log.info(f"Hydra working directory : {os.getcwd()}")

    if cfg.get("_target_"):
        eKonf._init_env_(cfg, verbose)

        eKonf.instantiate(cfg)

        eKonf._stop_env_(cfg, verbose)


if __name__ == "__main__":
    hydra_main()
