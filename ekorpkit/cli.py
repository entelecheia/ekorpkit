import os
import logging
import hydra
from ekorpkit import eKonf, __hydra_version_base__
from ekorpkit.info.docs import make_table


log = logging.getLogger(__name__)


def cmd(**args):
    eKonf.run(args)


def listup(**args):
    ekorpkit_dir = eKonf.__ekorpkit_path__
    corpus_info = {}
    for name in args[eKonf.Keys.CORPUS]["preset"][eKonf.Keys.CORPUS]:
        info_path = f"{ekorpkit_dir}/resources/corpora/{name}.yaml"
        info = eKonf.load(info_path)
        corpus_info[name] = info
    make_table(corpus_info.values(), args["info"]["table"])


def about(**args):
    from . import __version__

    name = args.get("name")
    print()
    for k, v in args.items():
        print(f"{k:11} : {v}")
    print(f"\nExecute `{name} --help` to see what eKorpkit provides")


@hydra.main(config_path="conf", config_name="config", version_base=__hydra_version_base__)
def hydra_main(cfg) -> None:
    verbose = cfg.verbose

    if verbose:
        log.info("## eKorpkit Command Line Interface for Hydra ##")
    eKonf._init_env_(cfg, verbose)

    if cfg.get("print_config"):
        if cfg.get("print_resolved_config"):
            log.info("## hydra configuration resolved ##")
            eKonf.pprint(cfg)
        else:
            log.info("## hydra configuration ##")
            print(eKonf.to_yaml(cfg))

    if verbose:
        log.info(f"Hydra working directory : {os.getcwd()}")
        log.info(f"Orig working directory  : {hydra.utils.get_original_cwd()}")

    eKonf.instantiate(cfg)

    eKonf._stop_env_(cfg, verbose)


if __name__ == "__main__":
    hydra_main()
