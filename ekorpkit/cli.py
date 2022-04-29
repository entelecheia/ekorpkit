import os
import logging
import hydra
from ekorpkit import eKonf
from pprint import pprint
from .tasks.info import make_table


logger = logging.getLogger(__name__)
logging.basicConfig(format="[ekorpkit]: %(message)s", level=logging.WARNING)


def cmd(**args):
    cfg = eKonf.to_config(args)
    if cfg._key_:
        cfg = eKonf.select(cfg, cfg._key_)
        eKonf.instantiate(cfg)


def listup(**args):
    ekorpkit_dir = os.path.dirname(os.path.abspath(__file__))
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
    # print(cfg)


def listfiles(**args):
    cfg = eKonf.to_config(args)
    args = cfg.corpus
    # corpus_paths = load_corpus_paths(args.corpus_dir, args.name, corpus_type=args.corpus_type,
    #     corpus_filetype=args.corpus_filetype, filename_pattern=args.filename_pattern)
    # for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
    #     print(f'{i_corpus:3} - {corpus_name} : {corpus_file}')


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg) -> None:
    # log.info("eKorpkit Command Line Interface for Hydra")
    verbose = cfg.verbose
    if verbose:
        print("\n## eKorpkit Command Line Interface for Hydra ##\n")
    if cfg.get("print_config"):
        print("## hydra configuration ##")
        print(eKonf.to_yaml(cfg))

    if cfg.get("print_resolved_config"):
        print("## hydra configuration resolved ##")
        args = eKonf.to_dict(cfg)
        pprint(args)
        print()

    if verbose:
        print(f"Hydra working directory : {os.getcwd()}\n")

    if cfg.get("_target_"):
        eKonf._init_env_(cfg, verbose)

        eKonf.instantiate(cfg)

        eKonf._stop_env_(cfg, verbose)
    # print(HydraConfig.get())


if __name__ == "__main__":
    hydra_main()
