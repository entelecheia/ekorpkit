import random
import os
import logging
import hydra

# from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from wasabi import msg
from ekorpkit.utils.batch.batcher import Batcher

import ekorpkit.config as config
from .tasks.info import make_table
from ekorpkit.utils.func import lower_case_with_underscores


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("iif", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("randint", random.randint, use_cache=True)
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver(
    "lower_case_with_underscores", lower_case_with_underscores
)


def listup(**args):
    ekorpkit_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_info = {}
    for name in args["corpus"]["preset"]["corpora"]:
        info_path = f"{ekorpkit_dir}/resource/corpora/{name}.yaml"
        info = OmegaConf.load(info_path)
        corpus_info[name] = info
    make_table(corpus_info.values(), args["info"]["table"])


def about(**args):
    cfg = OmegaConf.create(args)
    args = cfg.about.app
    print(f"\neKorpkit=={args.version}")
    print("Execute `ekorpkit --help` to see what eKorpkit provides")
    # print(cfg)


def listfiles(**args):
    cfg = OmegaConf.create(args)
    args = cfg.corpus
    # corpus_paths = load_corpus_paths(args.corpus_dir, args.name, corpus_type=args.corpus_type,
    #     corpus_filetype=args.corpus_filetype, filename_pattern=args.filename_pattern)
    # for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
    #     print(f'{i_corpus:3} - {corpus_name} : {corpus_file}')


def init_environ(cfg, verbose=False):
    env = cfg.env
    backend = env.distributed_framework.backend
    for env_name, env_value in env.get("os", {}).items():
        if env_value:
            if verbose:
                msg.info(f"setting environment variable {env_name} to {env_value}")
            os.environ[env_name] = str(env_value)

    if env.distributed_framework.initialize:
        backend_handle = None
        if backend == "ray":
            import ray

            ray_cfg = env.get("ray", None)
            ray_cfg = OmegaConf.to_container(ray_cfg, resolve=True)
            if verbose:
                msg.info(f"initializing ray with {ray_cfg}")
            ray.init(**ray_cfg)
            backend_handle = ray

        elif backend == "dask":
            from dask.distributed import Client

            dask_cfg = env.get("dask", None)
            dask_cfg = OmegaConf.to_container(dask_cfg, resolve=True)
            if verbose:
                msg.info(f"initializing dask client with {dask_cfg}")
            client = Client(**dask_cfg)
            if verbose:
                print(client)

        config.batcher = Batcher(backend_handle=backend_handle, **env.batcher)
        if verbose:
            print(config.batcher)
    if verbose:
        print()


def stop_environ(cfg, verbose=False):
    env = cfg.env
    backend = env.distributed_framework.backend

    if env.distributed_framework.initialize:
        if backend == "ray":
            import ray

            if ray.is_initialized():
                ray.shutdown()
                if verbose:
                    msg.info(f"shutting down ray")

        # elif modin_engine == 'dask':
        #     from dask.distributed import Client

        #     if Client.initialized():
        #         client.close()
        #         msg.info(f'shutting down dask client')


@hydra.main(config_path="conf", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # log.info("eKorpkit Command Line Interface for Hydra")
    verbose = cfg.verbose
    if verbose:
        print("\n## eKorpkit Command Line Interface for Hydra ##\n")
    if cfg.get("print_config"):
        print("## hydra configuration ##")
        print(OmegaConf.to_yaml(cfg))

    if cfg.get("print_resolved_config"):
        print("## hydra configuration resolved ##")
        args = OmegaConf.to_container(cfg, resolve=True)
        pprint(args)
        print()

    if verbose:
        print(f"Hydra working directory : {os.getcwd()}\n")

    if cfg.get("_target_"):
        init_environ(cfg, verbose)

        hydra.utils.instantiate(cfg, _recursive_=False)

        stop_environ(cfg, verbose)
    # print(HydraConfig.get())


if __name__ == "__main__":
    hydra_main()
