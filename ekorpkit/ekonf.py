import logging
import os
import functools
from pkgutil import extend_path
import random
import hydra
import pathlib
import ekorpkit.utils.batch.batcher as batcher
from pprint import pprint
from enum import Enum
from hydra.core.config_store import ConfigStore
from hydra.utils import get_method
from omegaconf import OmegaConf, SCMode, DictConfig, ListConfig
from typing import Any, List, IO, Dict, Union, Tuple, Optional
from cached_path import cached_path
from ekorpkit.io.gdown import cached_gdown, extractall
from ekorpkit.utils.func import lower_case_with_underscores
from . import _version


log = logging.getLogger(__name__)


def __ekorpkit_path__():
    return pathlib.Path(__file__).parent.as_posix()


def __home_path__():
    return pathlib.Path.home().as_posix()


def path(
    url_or_filename,
    extract_archive: bool = False,
    force_extract: bool = False,
    cache_dir=None,
    verbose: bool = False,
):
    if verbose:
        log.info(
            "caching path: {}, extract_archive: {}, force_extract: {}, cache_dir: {}".format(
                url_or_filename, extract_archive, force_extract, cache_dir
            )
        )

    if url_or_filename:
        try:
            if url_or_filename.startswith("gd://"):
                if extract_archive:
                    postprocess = extractall
                else:
                    postprocess = None

                _path = cached_gdown(
                    url_or_filename,
                    verbose=verbose,
                    postprocess=postprocess,
                    cache_dir=cache_dir,
                )
            else:
                if cache_dir is None:
                    cache_dir = (
                        pathlib.Path.home() / ".ekorpkit" / ".cache" / "cached_path"
                    )
                else:
                    cache_dir = pathlib.Path(cache_dir) / "cached_path"

                _path = cached_path(
                    url_or_filename,
                    extract_archive=extract_archive,
                    force_extract=force_extract,
                    cache_dir=cache_dir,
                ).as_posix()

            if verbose:
                log.info(f"cached path: {_path}")

            return _path

        except Exception as e:
            log.error(e)
            return None


def compose(
    overrides: List[str] = [],
    config_group: str = None,
    *,
    return_as_dict: bool = False,
    throw_on_resolution_failure: bool = True,
    throw_on_missing: bool = False,
    config_name="ekonf",
    verbose: bool = False,
):
    """
    Compose your configuration from config groups and overrides (overrides=["override_name"])

    :param overrides: List of overrides to apply
    :param config_group: Config group name to select ('config_group=name')
    :param return_as_dict: Return the composed config as a dict
    :param throw_on_resolution_failure: Throw if resolution fails
    :param throw_on_missing: Throw if a config is missing
    :param config_name: Name of the config to compose
    :param verbose: Print the composed config

    :return: The composed config
    """
    if config_group:
        _task = config_group.split("=")
        if len(_task) == 2:
            key, value = _task
        else:
            key = _task[0]
            value = None
    else:
        key = None
        value = None
    if key and value:
        with hydra.initialize_config_module(config_module="ekorpkit.conf"):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            cfg = select(
                cfg,
                key=key,
                default=None,
                throw_on_missing=False,
                throw_on_resolution_failure=False,
            )
            if cfg:
                overide = config_group
            else:
                overide = f"+{config_group}"
            if overrides:
                overrides.append(overide)
            else:
                overrides = [overide]
    if verbose:
        print(f"compose config with overrides: {overrides}")
    with hydra.initialize_config_module(config_module="ekorpkit.conf"):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        if key:
            cfg = select(
                cfg,
                key=key,
                default=None,
                throw_on_missing=throw_on_missing,
                throw_on_resolution_failure=throw_on_resolution_failure,
            )
        if verbose:
            print(cfg)
        if return_as_dict and isinstance(cfg, DictConfig):
            return to_dict(cfg)
        return cfg


config = compose()

DictKeyType = Union[str, int, Enum, float, bool]

OmegaConf.register_new_resolver("__ekorpkit_path__", __ekorpkit_path__)
OmegaConf.register_new_resolver("__home_path__", __home_path__)
OmegaConf.register_new_resolver("iif", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("randint", random.randint, use_cache=True)
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("cached_path", path)
OmegaConf.register_new_resolver(
    "lower_case_with_underscores", lower_case_with_underscores
)


def partial(_partial_, *args, **kwargs):
    return functools.partial(get_method(_partial_), *args, **kwargs)


class _Keys(str, Enum):
    """Special keys in configs used by instantiate."""

    TARGET = "_target_"
    CONVERT = "_convert_"
    RECURSIVE = "_recursive_"
    ARGS = "_args_"
    PARTIAL = "__partial__"


class eKonf:
    """ekorpkit config primary class"""

    __version__ = _version.get_versions()["version"]
    __ekorpkit_path__ = __ekorpkit_path__()
    __home_path__ = __home_path__()
    config = compose()

    def __init__(self) -> None:
        raise NotImplementedError("Use one of the static construction functions")

    @staticmethod
    def compose(
        overrides: List[str] = [],
        config_group: str = None,
        *,
        return_as_dict: bool = False,
        throw_on_resolution_failure: bool = True,
        throw_on_missing: bool = False,
        config_name="ekonf",
        verbose: bool = False,
    ):
        return compose(
            overrides,
            config_group=config_group,
            return_as_dict=return_as_dict,
            throw_on_resolution_failure=throw_on_resolution_failure,
            throw_on_missing=throw_on_missing,
            config_name=config_name,
            verbose=verbose,
        )

    @staticmethod
    def select(
        cfg: Any,
        key: str,
        *,
        default: Any = None,
        throw_on_resolution_failure: bool = True,
        throw_on_missing: bool = False,
    ):
        return select(
            cfg,
            key,
            default=default,
            throw_on_resolution_failure=throw_on_resolution_failure,
            throw_on_missing=throw_on_missing,
        )

    @staticmethod
    def to_dict(
        cfg: Any,
    ):
        return to_dict(cfg)

    @staticmethod
    def to_config(
        cfg: Any,
    ):
        return to_config(cfg)

    @staticmethod
    def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
        if resolve:
            cfg = to_dict(cfg)
        return to_yaml(cfg, resolve=resolve, sort_keys=sort_keys)

    @staticmethod
    def to_container(
        cfg: Any,
        *,
        resolve: bool = False,
        throw_on_missing: bool = False,
        enum_to_str: bool = False,
        structured_config_mode: SCMode = SCMode.DICT,
    ):
        return to_container(
            cfg,
            resolve,
            throw_on_missing,
            enum_to_str,
            structured_config_mode,
        )

    @staticmethod
    def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
        return instantiate(config, *args, **kwargs)

    @staticmethod
    def is_config(
        cfg: Any,
    ):
        return is_config(cfg)

    @staticmethod
    def is_instantiatable(cfg: Any):
        return is_instantiatable(cfg)

    @staticmethod
    def load(file_: Union[str, pathlib.Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
        return OmegaConf.load(file_)

    @staticmethod
    def merge(
        *configs: Union[
            DictConfig,
            ListConfig,
            Dict[DictKeyType, Any],
            List[Any],
            Tuple[Any, ...],
            Any,
        ],
    ) -> Union[ListConfig, DictConfig]:
        """
        Merge a list of previously created configs into a single one
        :param configs: Input configs
        :return: the merged config object.
        """
        return OmegaConf.merge(*configs)

    @staticmethod
    def save(
        config: Any, f: Union[str, pathlib.Path, IO[Any]], resolve: bool = False
    ) -> None:
        OmegaConf.save(config, f, resolve=resolve)

    @staticmethod
    def pprint(cfg: Any, **kwargs):
        pprint(cfg, **kwargs)

    @staticmethod
    def print(cfg: Any, **kwargs):
        pprint(cfg, **kwargs)

    @staticmethod
    def call(cfg: Any, obj: object):
        call(cfg, obj)

    @staticmethod
    def _init_env_(cfg, verbose=False):
        _init_env_(cfg, verbose=verbose)

    @staticmethod
    def _stop_env_(cfg, verbose=False):
        _stop_env_(cfg, verbose=verbose)

    @staticmethod
    def path(
        url_or_filename,
        extract_archive: bool = False,
        force_extract: bool = False,
        cache_dir=None,
    ):
        """
        Given something that might be a URL or local path, determine which.
        If it's a remote resource, download the file and cache it, and
        then return the path to the cached file. If it's already a local path,
        make sure the file exists and return the path.

        For URLs, the following schemes are all supported out-of-the-box:

        * ``http`` and ``https``,
        * ``s3`` for objects on `AWS S3`_,
        * ``gs`` for objects on `Google Cloud Storage (GCS)`_, and
        * ``hf`` for objects or repositories on `HuggingFace Hub`_.

        You can also extend ``cached_path()`` to handle more schemes with :func:`add_scheme_client()`.

        .. _AWS S3: https://aws.amazon.com/s3/
        .. _Google Cloud Storage (GCS): https://cloud.google.com/storage
        .. _HuggingFace Hub: https://huggingface.co/

        Examples
        --------

        To download a file over ``https``::

            cached_path("https://github.com/allenai/cached_path/blob/main/README.md")

        To download an object on GCS::

            cached_path("gs://allennlp-public-models/lerc-2020-11-18.tar.gz")

        To download the PyTorch weights for the model `epwalsh/bert-xsmall-dummy`_
        on HuggingFace, you could do::

            cached_path("hf://epwalsh/bert-xsmall-dummy/pytorch_model.bin")

        For paths or URLs that point to a tarfile or zipfile, you can append the path
        to a specific file within the archive to the ``url_or_filename``, preceeded by a "!".
        The archive will be automatically extracted (provided you set ``extract_archive`` to ``True``),
        returning the local path to the specific file. For example::

            cached_path("model.tar.gz!weights.th", extract_archive=True)

        .. _epwalsh/bert-xsmall-dummy: https://huggingface.co/epwalsh/bert-xsmall-dummy

        Parameters
        ----------

        url_or_filename :
            A URL or path to parse and possibly download.

        extract_archive :
            If ``True``, then zip or tar.gz archives will be automatically extracted.
            In which case the directory is returned.

        force_extract :
            If ``True`` and the file is an archive file, it will be extracted regardless
            of whether or not the extracted directory already exists.

            .. caution::
                Use this flag with caution! This can lead to race conditions if used
                from multiple processes on the same file.

        cache_dir :
            The directory to cache downloads. If not specified, the global default cache directory
            will be used (``~/.cache/cached_path``). This can be set to something else with
            :func:`set_cache_dir()`.

        Returns
        -------
        :class:`pathlib.Path`
            The local path to the (potentially cached) resource.

        Raises
        ------
        ``FileNotFoundError``

            If the resource cannot be found locally or remotely.

        ``ValueError``
            When the URL is invalid.

        ``Other errors``
            Other error types are possible as well depending on the client used to fetch
            the resource.

        """
        return path(
            url_or_filename,
            extract_archive=extract_archive,
            force_extract=force_extract,
            cache_dir=cache_dir,
        )


def call(cfg: Any, obj: object):
    if isinstance(cfg, list):
        for _run in cfg:
            if isinstance(_run, str):
                getattr(obj, _run)()
            elif isinstance(_run, dict):
                _run = eKonf.to_dict(_run)
                getattr(obj, _run["name"])(**_run["args"])


def pprint(cfg: Any, **kwargs):
    import pprint

    if is_config(cfg):
        pprint.pprint(to_dict(cfg), **kwargs)
    else:
        print(cfg)


def select(
    cfg: Any,
    key: str,
    *,
    default: Any = None,
    throw_on_resolution_failure: bool = True,
    throw_on_missing: bool = False,
):
    key = key.replace("/", ".")
    return OmegaConf.select(
        cfg,
        key=key,
        default=default,
        throw_on_resolution_failure=throw_on_resolution_failure,
        throw_on_missing=throw_on_missing,
    )


def to_dict(
    cfg: Any,
):
    if isinstance(cfg, dict):
        cfg = to_config(cfg)
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=False,
            structured_config_mode=SCMode.DICT,
        )
    return cfg


def is_config(
    cfg: Any,
):
    return isinstance(cfg, (DictConfig, dict))


def is_instantiatable(cfg: Any):
    return is_config(cfg) and "_target_" in cfg


def to_config(
    cfg: Any,
):
    return OmegaConf.create(cfg)


def load(file_: Union[str, pathlib.Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
    return OmegaConf.load(file_)


def save(
    config: Any, f: Union[str, pathlib.Path, IO[Any]], resolve: bool = False
) -> None:
    eKonf.save(config, f, resolve=resolve)


def merge(
    *configs: Union[
        DictConfig,
        ListConfig,
        Dict[DictKeyType, Any],
        List[Any],
        Tuple[Any, ...],
        Any,
    ],
) -> Union[ListConfig, DictConfig]:
    return eKonf.merge(*configs)


def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
    return OmegaConf.to_yaml(cfg, resolve=resolve, sort_keys=sort_keys)


def to_container(
    cfg: Any,
    *,
    resolve: bool = False,
    throw_on_missing: bool = False,
    enum_to_str: bool = False,
    structured_config_mode: SCMode = SCMode.DICT,
):
    return OmegaConf.to_container(
        cfg,
        resolve=resolve,
        throw_on_missing=throw_on_missing,
        enum_to_str=enum_to_str,
        structured_config_mode=structured_config_mode,
    )


def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    :param config: An config object describing what to call and what params to use.
                   In addition to the parameters, the config must contain:
                   _target_ : target class or callable name (str)
                   And may contain:
                   _args_: List-like of positional arguments to pass to the target
                   _recursive_: Construct nested objects as well (bool).
                                False by default.
                                may be overridden via a _recursive_ key in
                                the kwargs
                   _convert_: Conversion strategy
                        none    : Passed objects are DictConfig and ListConfig, default
                        partial : Passed objects are converted to dict and list, with
                                  the exception of Structured Configs (and their fields).
                        all     : Passed objects are dicts, lists and primitives without
                                  a trace of OmegaConf containers
                   _partial_: If True, return functools.partial wrapped method or object
                              False by default. Configure per target.
                   _args_: List-like of positional arguments
    :param args: Optional positional parameters pass-through
    :param kwargs: Optional named parameters to override
                   parameters in the config object. Parameters not present
                   in the config objects are being passed as is to the target.
                   IMPORTANT: dataclasses instances in kwargs are interpreted as config
                              and cannot be used as passthrough
    :return: if _target_ is a class name: the instantiated object
             if _target_ is a callable: the return value of the call
    """
    if config.get("_target_") is None:
        log.warning("No target specified in config")
        return None
    _recursive_ = config.get(_Keys.RECURSIVE, False)
    if _Keys.RECURSIVE not in kwargs:
        kwargs[_Keys.RECURSIVE] = _recursive_
    return hydra.utils.instantiate(config, *args, **kwargs)


def _init_env_(cfg, verbose=False):
    env = cfg.env
    backend = env.distributed_framework.backend
    for env_name, env_value in env.get("os", {}).items():
        if env_value:
            if verbose:
                log.info(f"setting environment variable {env_name} to {env_value}")
            os.environ[env_name] = str(env_value)

    if env.distributed_framework.initialize:
        backend_handle = None
        if backend == "ray":
            import ray

            ray_cfg = env.get("ray", None)
            ray_cfg = eKonf.to_container(ray_cfg, resolve=True)
            if verbose:
                log.info(f"initializing ray with {ray_cfg}")
            ray.init(**ray_cfg)
            backend_handle = ray

        elif backend == "dask":
            from dask.distributed import Client

            dask_cfg = env.get("dask", None)
            dask_cfg = eKonf.to_container(dask_cfg, resolve=True)
            if verbose:
                log.info(f"initializing dask client with {dask_cfg}")
            client = Client(**dask_cfg)
            if verbose:
                log.info(client)

        batcher.batcher_instance = batcher.Batcher(
            backend_handle=backend_handle, **env.batcher
        )
        if verbose:
            log.info(batcher.batcher_instance)


def _stop_env_(cfg, verbose=False):
    env = cfg.env
    backend = env.distributed_framework.backend

    if env.distributed_framework.initialize:
        if backend == "ray":
            import ray

            if ray.is_initialized():
                ray.shutdown()
                if verbose:
                    log.info("shutting down ray")

        # elif modin_engine == 'dask':
        #     from dask.distributed import Client

        #     if Client.initialized():
        #         client.close()
        #         log.info(f'shutting down dask client')
