import os
import logging
import hydra
from .base import __hydra_version_base__
from .hconf import HyFI


log = logging.getLogger(__name__)


def cmd(**args):
    HyFI.run(args)


def about(**args):
    # from . import __version__
    args = args["about"]
    name = args.get("name")
    print()
    for k, v in args.items():
        print(f"{k:11} : {v}")
    print(f"\nExecute `{name} --help` to see what eKorpkit provides")


@hydra.main(
    config_path="conf", config_name="config", version_base=__hydra_version_base__
)
def hydra_main(cfg) -> None:
    verbose = cfg.verbose
    app_name = cfg.about.name
    print_config = cfg.print_config
    print_resolved_config = cfg.print_resolved_config

    if verbose:
        log.info("## Command Line Interface for %s ##" % app_name)
    HyFI.initialize(cfg)

    if print_config:
        if print_resolved_config:
            log.info("## hydra configuration resolved ##")
            HyFI.pprint(cfg)
        else:
            log.info("## hydra configuration ##")
            print(HyFI.to_yaml(cfg))

    if verbose:
        log.info(f"Hydra working directory : {os.getcwd()}")
        log.info(f"Orig working directory  : {hydra.utils.get_original_cwd()}")

    HyFI.instantiate(cfg)

    HyFI.terminate()


if __name__ == "__main__":
    # hydra.initialize_config_module
    hydra_main()
