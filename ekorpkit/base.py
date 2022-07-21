import logging
import warnings
import os
import sys
import random
import hydra
import dotenv
import functools
import inspect
import importlib
import pandas as pd
import ekorpkit.utils.batch.batcher as batcher
from pandas import DataFrame
from enum import Enum
from tqdm.auto import tqdm
from pathlib import Path
from omegaconf import OmegaConf, SCMode, DictConfig, ListConfig
from typing import Any, List, IO, Dict, Union, Tuple
from ekorpkit.utils.batch import decorator_apply
from ekorpkit.io.cached_path import cached_path
from ekorpkit.utils.func import lower_case_with_underscores
from . import _version


def _setLogger(level=None, force=True, filterwarnings_action="ignore", **kwargs):
    level = level or os.environ.get("EKORPKIT_LOG_LEVEL", "INFO")
    os.environ["EKORPKIT_LOG_LEVEL"] = level
    if filterwarnings_action is not None:
        warnings.filterwarnings(filterwarnings_action)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if sys.version_info >= (3, 8):
        logging.basicConfig(level=level, force=force, **kwargs)
    else:
        logging.basicConfig(level=level, **kwargs)


def _getLogger(
    _name=None,
    _log_level=None,
    _fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    _name = _name or __name__
    logger = logging.getLogger(_name)
    _log_level = _log_level or os.environ.get("EKORPKIT_LOG_LEVEL", "INFO")
    logger.setLevel(_log_level)
    return logger


logger = _getLogger()

__hydra_version_base__ = "1.2"


def __ekorpkit_path__():
    return Path(__file__).parent.as_posix()


def __home_path__():
    return Path.home().as_posix()


def __version__():
    return _version.get_versions()["version"]


def _check_path(_path: str, alt_path: str = None):
    if os.path.exists(_path):
        return _path
    else:
        return alt_path


def _mkdir(_path: str):
    if _path is None:
        return None
    Path(_path).mkdir(parents=True, exist_ok=True)
    return _path


def _exists(a, *p):
    _path = os.path.join(a, *p)
    return os.path.exists(_path)


def _is_file(a, *p):
    _path = os.path.join(a, *p)
    return Path(_path).is_file()


def _is_dir(a, *p):
    _path = os.path.join(a, *p)
    return Path(_path).is_dir()


def _join_path(a, *p):
    if p and p[0] is not None:
        p = [str(_p) for _p in p]
        return os.path.join(a, *p)
    else:
        return a


def _today(_format="%Y-%m-%d"):
    from datetime import datetime

    if _format is None:
        return datetime.today().date()
    else:
        return datetime.today().strftime(_format)


def _now(_format="%Y-%m-%d %H:%M:%S"):
    from datetime import datetime

    if _format is None:
        return datetime.now()
    else:
        return datetime.now().strftime(_format)


def _strptime(
    _date_str: str,
    _format: str = "%Y-%m-%d",
):
    from datetime import datetime

    return datetime.strptime(_date_str, _format)


def _to_dateparm(_date, _format="%Y-%m-%d"):
    from datetime import datetime

    _dtstr = datetime.strftime(_date, _format)
    _dtstr = "${to_datetime:" + _dtstr + "," + _format + "}"
    return _dtstr


def _to_datetime(data, _format=None, _columns=None, **kwargs):
    from datetime import datetime

    if isinstance(data, datetime):
        return data
    elif isinstance(data, str):
        if _format is None:
            _format = "%Y-%m-%d"
        return datetime.strptime(data, _format)
    elif isinstance(data, int):
        return datetime.fromtimestamp(data)
    elif isinstance(data, DataFrame):
        if _columns:
            if isinstance(_columns, str):
                _columns = [_columns]
            for _col in _columns:
                data[_col] = pd.to_datetime(data[_col], format=_format, **kwargs)
        return data
    else:
        return data


def _to_numeric(data, _columns=None, errors="coerce", downcast=None, **kwargs):
    if isinstance(data, str):
        return float(data)
    elif isinstance(data, int):
        return data
    elif isinstance(data, float):
        return data
    elif isinstance(data, DataFrame):
        if _columns:
            if isinstance(_columns, str):
                _columns = [_columns]
            for _col in _columns:
                data[_col] = pd.to_numeric(data[_col], errors=errors, downcast=downcast)
        return data
    else:
        return data


def _path(
    url_or_filename,
    extract_archive: bool = False,
    force_extract: bool = False,
    return_parent_dir: bool = False,
    cache_dir=None,
    verbose: bool = False,
):
    return cached_path(
        url_or_filename,
        extract_archive=extract_archive,
        force_extract=force_extract,
        return_parent_dir=return_parent_dir,
        cache_dir=cache_dir,
        verbose=verbose,
    )


def _compose(
    config_group: str = None,
    overrides: List[str] = [],
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
            value = "default"
        config_group = f"{key}={value}"
    else:
        key = None
        value = None
    if key and value:
        with hydra.initialize_config_module(
            config_module="ekorpkit.conf", version_base=__hydra_version_base__
        ):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            cfg = _select(
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
    with hydra.initialize_config_module(
        config_module="ekorpkit.conf", version_base=__hydra_version_base__
    ):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
        if key:
            cfg = _select(
                cfg,
                key=key,
                default=None,
                throw_on_missing=throw_on_missing,
                throw_on_resolution_failure=throw_on_resolution_failure,
            )
        if verbose:
            print(cfg)
        if return_as_dict and isinstance(cfg, DictConfig):
            return _to_dict(cfg)
        return cfg


def _to_dict(
    cfg: Any,
):
    if isinstance(cfg, dict):
        cfg = _to_config(cfg)
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=False,
            structured_config_mode=SCMode.DICT,
        )
    return cfg


def _to_config(
    cfg: Any,
):
    return OmegaConf.create(cfg)


def dotenv_values(dotenv_path=None, **kwargs):
    config = dotenv.dotenv_values(dotenv_path=dotenv_path, **kwargs)
    return dict(config)


def getcwd():
    try:
        return hydra.utils.get_original_cwd()
    except:
        return os.getcwd()


_env_initialized_ = False

_config = _compose().copy()

DictKeyType = Union[str, int, Enum, float, bool]

OmegaConf.register_new_resolver("__ekorpkit_path__", __ekorpkit_path__)
OmegaConf.register_new_resolver("__home_path__", __home_path__)
OmegaConf.register_new_resolver("__version__", __version__)
OmegaConf.register_new_resolver("today", _today)
OmegaConf.register_new_resolver("to_datetime", _strptime)
OmegaConf.register_new_resolver("iif", lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver("alt", lambda val, alt: val if val else alt)
OmegaConf.register_new_resolver("randint", random.randint, use_cache=True)
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("get_original_cwd", getcwd)
OmegaConf.register_new_resolver("exists", _exists)
OmegaConf.register_new_resolver("join_path", _join_path)
OmegaConf.register_new_resolver("mkdir", _mkdir)
OmegaConf.register_new_resolver("dirname", os.path.dirname)
OmegaConf.register_new_resolver("basename", os.path.basename)
OmegaConf.register_new_resolver("check_path", _check_path)
OmegaConf.register_new_resolver("cached_path", _path)
OmegaConf.register_new_resolver(
    "lower_case_with_underscores", lower_case_with_underscores
)
OmegaConf.register_new_resolver("dotenv_values", dotenv_values)


class _SPLITS(str, Enum):
    """Split keys in configs used by Dataset."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class _Keys(str, Enum):
    """Special keys in configs used by ekorpkit."""

    TARGET = "_target_"
    CONVERT = "_convert_"
    RECURSIVE = "_recursive_"
    ARGS = "_args_"
    PARTIAL = "_partial_"
    CONFIG = "_config_"
    CONFIG_GROUP = "_config_group_"
    PIPELINE = "_pipeline_"
    TASK = "_task_"
    CALL = "_call_"
    EXEC = "_exec_"
    rcPARAMS = "rcParams"
    METHOD = "_method_"
    METHOD_NAME = "_name_"
    FUNC = "_func_"
    NAME = "name"
    SPLIT = "split"
    CORPUS = "corpus"
    DATASET = "dataset"
    PATH = "path"
    OUTPUT = "output"
    ID = "id"
    _ID = "_id_"
    META_MERGE_ON = "meta_merge_on"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    DATETIME = "datetime"
    X = "x"
    Y = "y"
    INDEX = "index"
    COLUMNS = "columns"
    KEY = "key"
    KEYS = "_keys_"
    DATA = "data"
    META = "meta"
    FORMAT = "format"
    VERBOSE = "verbose"
    FILE = "file"
    FILENAME = "filename"
    SUFFIX = "suffix"
    MODEL = "model"
    LOG = "log"
    PRED = "pred"
    DEFAULT = "_default_"
    EVAL = "_eval_"
    TRAIN = "_train_"
    PREDICT = "_predict_"
    PREDICTED = "predicted"
    PRED_PROBS = "pred_probs"
    ACTUAL = "actual"
    INPUT = "input"
    TARGET_TEXT = "target_text"
    MODEL_OUTPUTS = "model_outputs"
    LABELS = "labels"
    PREFIX = "prefix"
    FEATURES = "features"
    COUNT = "count"
    CLASSIFICATION = "classification"


class _Defaults(str, Enum):
    ID_SEP = "_"
    SENT_SEP = "\n"
    SEG_SEP = "\n\n"
    POS_DELIM = "\\"
    NGRAM_DELIM = ";"


def _methods(cfg: Any, obj: object, return_function=False):
    cfg = _to_dict(cfg)
    if not cfg:
        logger.info("No method defined to call")
        return

    if isinstance(cfg, dict) and _Keys.METHOD in cfg:
        _method_ = cfg[_Keys.METHOD]
    else:
        _method_ = cfg
    if isinstance(_method_, str):
        _fn = getattr(obj, _method_)
        if return_function:
            logger.info(f"Returning function {_fn}")
            return _fn
        logger.info(f"Calling {_method_}")
        return _fn(**cfg)
    elif isinstance(_method_, dict):
        if _Keys.CALL in _method_:
            _call_ = _method_.pop(_Keys.CALL)
        else:
            _call_ = True
        if _call_:
            _fn = getattr(obj, _method_[_Keys.METHOD_NAME])
            _parms = _method_.pop(_Keys.rcPARAMS, {})
            if return_function:
                if not _parms:
                    logger.info(f"Returning function {_fn}")
                    return _fn
                logger.info(f"Returning function {_fn} with params {_parms}")
                return functools.partial(_fn, **_parms)
            logger.info(f"Calling {_method_}")
            return _fn(**_parms)
        else:
            logger.info(f"Skipping call to {_method_}")
    elif isinstance(_method_, list):
        for _each_method in _method_:
            logger.info(f"Calling {_each_method}")
            if isinstance(_each_method, str):
                getattr(obj, _each_method)()
            elif isinstance(_each_method, dict):
                if _Keys.CALL in _each_method:
                    _call_ = _each_method.pop(_Keys.CALL)
                else:
                    _call_ = True
                if _call_:
                    getattr(obj, _each_method[_Keys.METHOD_NAME])(
                        **_each_method[_Keys.rcPARAMS]
                    )
                else:
                    logger.info(f"Skipping call to {_each_method}")


def _function(cfg: Any, _name_, return_function=False, **parms):
    cfg = _to_dict(cfg)
    if not isinstance(cfg, dict):
        logger.info("No function defined to execute")
        return None

    if _Keys.FUNC not in cfg:
        logger.info("No function defined to execute")
        return None

    _functions_ = cfg[_Keys.FUNC]
    fn = _partial(_functions_[_name_])
    if _name_ in cfg:
        _parms = cfg[_name_]
        _parms = {**_parms, **parms}
    else:
        _parms = parms
    if _Keys.EXEC in _parms:
        _exec_ = _parms.pop(_Keys.EXEC)
    else:
        _exec_ = True
    if _exec_:
        if callable(fn):
            if return_function:
                logger.info(f"Returning function {fn}")
                return fn
            logger.info(f"Executing function {fn} with parms {_parms}")
            return fn(**_parms)
        else:
            logger.info(f"Function {_name_} not callable")
            return None
    else:
        logger.info(f"Skipping execute of {fn}")
        return None


def _print(cfg: Any, resolve: bool = True, **kwargs):
    import pprint

    if _is_config(cfg):
        if resolve:
            pprint.pprint(_to_dict(cfg), **kwargs)
        else:
            pprint.pprint(cfg, **kwargs)
    else:
        print(cfg)


def _select(
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


def _is_config(
    cfg: Any,
):
    return isinstance(cfg, (DictConfig, dict))


def _is_list(
    cfg: Any,
):
    return isinstance(cfg, (ListConfig, list))


def _is_instantiatable(cfg: Any):
    return _is_config(cfg) and _Keys.TARGET in cfg


def _load(file_: Union[str, Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
    return OmegaConf.load(file_)


def _save(config: Any, f: Union[str, Path, IO[Any]], resolve: bool = False) -> None:
    os.makedirs(os.path.dirname(f), exist_ok=True)
    OmegaConf.save(config, f, resolve=resolve)


def _update(_dict, _overrides):
    import collections.abc

    for k, v in _overrides.items():
        if isinstance(v, collections.abc.Mapping):
            _dict[k] = _update((_dict.get(k) or {}), v)
        else:
            _dict[k] = v
    return _dict


def _merge(
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


def _to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
    return OmegaConf.to_yaml(cfg, resolve=resolve, sort_keys=sort_keys)


def _to_container(
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


def _run(config: Any, **kwargs: Any) -> Any:
    config = _merge(config, kwargs)
    _config_ = config.get(_Keys.CONFIG)
    if _config_ is None:
        logger.warning("No _config_ specified in config")
        return None
    if isinstance(_config_, str):
        _config_ = [_config_]
    for _cfg_ in _config_:
        cfg = _select(config, _cfg_)
        _instantiate(cfg)


def _partial(
    config: Any = None, config_group: str = None, *args: Any, **kwargs: Any
) -> Any:
    if config is None and config_group is None:
        logger.warning("No config specified")
        return None
    elif config_group is not None:
        config = _compose(config_group=config_group)
    kwargs[_Keys.PARTIAL] = True
    return _instantiate(config, *args, **kwargs)


def _instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
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
    if not _env_initialized_:
        _init_env_()
    verbose = config.get(_Keys.VERBOSE, False)
    if not _is_instantiatable(config):
        if verbose:
            logger.info(f"Config is not instantiatable, returning config")
        return config
    _recursive_ = config.get(_Keys.RECURSIVE, False)
    if _Keys.RECURSIVE not in kwargs:
        kwargs[_Keys.RECURSIVE.value] = _recursive_
    if verbose:
        logger.info(f"instantiating {config.get(_Keys.TARGET)}...")
    return hydra.utils.instantiate(config, *args, **kwargs)


def _load_dotenv(verbose=False):
    original_cwd = getcwd()
    config_dir = os.environ.get("EKORPKIT_CONFIG_DIR")
    dotenv_dir = config_dir or original_cwd
    dotenv_path = Path(dotenv_dir, ".env")
    if dotenv_path.is_file():
        dotenv.load_dotenv(dotenv_path=dotenv_path, verbose=verbose)
        logger.info(f"Loaded .env from {dotenv_path}")
    else:
        logger.info(f"No .env file found in {dotenv_path}")


def _osenv(key: str = None) -> Any:
    if key:
        return os.environ.get(key)
    return os.environ


def _env_set(key: str, value: Any) -> None:
    if _is_dir(value):
        value = os.path.abspath(value)
    os.environ[key] = value


def _init_env_(cfg=None, verbose=False):
    global _env_initialized_

    _load_dotenv(verbose=verbose)
    if _is_notebook():
        _log_level = os.environ.get("EKORPKIT_LOG_LEVEL", "INFO")
        logging.basicConfig(level=_log_level, force=True)

    if cfg is None:
        cfg = _config
    env = cfg.env

    backend = env.distributed_framework.backend
    for env_name, env_value in env.get("os", {}).items():
        if env_value:
            logger.info(f"setting environment variable {env_name} to {env_value}")
            os.environ[env_name] = str(env_value)

    if env.distributed_framework.initialize:
        backend_handle = None
        if backend == "ray":
            import ray

            ray_cfg = env.get("ray", None)
            ray_cfg = _to_container(ray_cfg, resolve=True)
            if verbose:
                logger.info(f"initializing ray with {ray_cfg}")
            ray.init(**ray_cfg)
            backend_handle = ray

        elif backend == "dask":
            from dask.distributed import Client

            dask_cfg = env.get("dask", None)
            dask_cfg = _to_container(dask_cfg, resolve=True)
            if verbose:
                logger.info(f"initializing dask client with {dask_cfg}")
            client = Client(**dask_cfg)
            if verbose:
                logger.info(client)

        batcher.batcher_instance = batcher.Batcher(
            backend_handle=backend_handle, **env.batcher
        )
        if verbose:
            logger.info(f"initialized batcher with {batcher.batcher_instance}")
    _env_initialized_ = True


def _stop_env_(cfg, verbose=False):
    env = cfg.env
    backend = env.distributed_framework.backend
    if verbose:
        logger.info(f"stopping {backend}, if running")

    if env.distributed_framework.initialize:
        if backend == "ray":
            import ray

            if ray.is_initialized():
                ray.shutdown()
                if verbose:
                    logger.info("shutting down ray")

        # elif modin_engine == 'dask':
        #     from dask.distributed import Client

        #     if Client.initialized():
        #         client.close()
        #         log.info(f'shutting down dask client')


def _pipe(data, pipe):
    _func_ = pipe.get(_Keys.FUNC)
    fn = _partial(_func_)
    logger.info(f"Applying pipe: {fn}")
    if isinstance(data, dict):
        if "concat_dataframes" in str(fn):
            return fn(data, pipe)
        else:
            dfs = {}
            for df_no, df_name in enumerate(data):
                df_each = data[df_name]
                logger.info(
                    f"Applying pipe to dataframe [{df_name}], {(df_no+1)}/{len(data)}"
                )
                pipe[_Keys.SUFFIX.value] = df_name
                dfs[df_name] = fn(df_each, pipe)
            return dfs
    else:
        return fn(data, pipe)


def _dependencies(_key=None, _path=None):
    import re
    from collections import defaultdict

    if _path is None:
        _path = os.path.join(
            os.path.dirname(__file__), "resources", "requirements-extra.yaml"
        )

    with open(_path) as fp:
        extra_deps = defaultdict(set)
        for k in fp:
            if k.strip() and not k.startswith("#"):
                tags = set()
                if ":" in k:
                    k, v = k.split(":")
                    tags.update(vv.strip() for vv in v.split(","))
                tags.add(re.split("[<=>]", k.strip())[0])
                for t in tags:
                    extra_deps[t].add(k.strip())

        # add tag `exhaustive` at the end
        extra_deps["exhaustive"] = set(vv for v in extra_deps.values() for vv in v)

    if _key is None or _key == "keys":
        tags = []
        for tag, deps in extra_deps.items():
            if len(deps) > 1:
                tags.append(tag)
        return tags
    else:
        return extra_deps[_key]


def _ensure_list(value):
    if not value:
        return []
    elif isinstance(value, str):
        return [value]
    return _to_dict(value)


def _ensure_kwargs(_kwargs, _fn):
    from inspect import getfullargspec as getargspec

    if callable(_fn):
        kwargs = {}
        args = getargspec(_fn).args
        logger.info(f"args of {_fn}: {args}")
        for k, v in _kwargs.items():
            if k in args:
                kwargs[k] = v
        return kwargs
    return _kwargs


def _apply(
    func,
    series,
    description=None,
    use_batcher=True,
    minibatch_size=None,
    num_workers=None,
    verbose=False,
    **kwargs,
):
    batcher_instance = batcher.batcher_instance
    if use_batcher and batcher_instance is not None:
        if batcher_instance is not None:
            batcher_minibatch_size = batcher_instance.minibatch_size
        else:
            batcher_minibatch_size = 1000
        if minibatch_size is None:
            minibatch_size = batcher_minibatch_size
        if num_workers is not None:
            batcher_instance.procs = int(num_workers)
        if batcher_instance.procs > 1:
            batcher_instance.minibatch_size = min(
                int(len(series) / batcher_instance.procs) + 1, minibatch_size
            )
            logger.info(
                f"Using batcher with minibatch size: {batcher_instance.minibatch_size}"
            )
            results = decorator_apply(func, batcher_instance, description=description)(
                series
            )
            if batcher_instance is not None:
                batcher_instance.minibatch_size = batcher_minibatch_size
            return results

    if batcher_instance is None:
        logger.warning("Warning: batcher not initialized")
    tqdm.pandas(desc=description)
    return series.progress_apply(func)


def _is_colab():
    is_colab = "google.colab" in sys.modules
    if is_colab:
        logger.info("Google Colab detected.")
    else:
        logger.info("Google Colab not detected.")
    return is_colab


def _is_notebook():
    is_notebook = "ipykernel" in sys.modules
    if is_notebook:
        ip = sys.modules["ipykernel"]
        ip_version = ip.version_info
        ip_client = ip.write_connection_file.__module__.split(".")[0]
        # logger.info(f"IPython version: {ip_version}, client: {ip_client}")
    else:
        logger.info("IPython not detected.")
    return is_notebook


def _nvidia_smi():
    import subprocess

    nvidiasmi_output = subprocess.run(
        ["nvidia-smi", "-L"], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    return nvidiasmi_output


def _is_cuda_available():
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _set_cuda(device=0):
    try:
        import torch

        _names = []
        if isinstance(device, str):
            device = device.replace("cuda:", "")
            ids = device.split(",")
        else:
            ids = [str(device)]
        for id in ids:
            _device_name = torch.cuda.get_device_name(int(id))
            _names.append(_device_name)
        logger.info(f"Setting cuda device to {_names}")
        device = ", ".join(ids)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        raise Exception("Cuda device not found")


def _getsource(obj):
    """Return the source code of the object."""
    try:
        if _is_config(obj):
            if _Keys.TARGET in obj:
                target_string = obj[_Keys.TARGET]
                mod_name, object_name = target_string.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                obj = getattr(mod, object_name)
        elif isinstance(obj, str):
            mod_name, object_name = obj.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, object_name)
        return inspect.getsource(obj)
    except:
        return ""


def _viewsource(obj):
    """Print the source code of the object."""
    print(_getsource(obj))

    # from ipywidgets import Output


def _clear_output(wait=False):
    from IPython import display

    if _is_notebook():
        display.clear_output(wait=True)


def _display(
    *objs,
    include=None,
    exclude=None,
    metadata=None,
    transient=None,
    display_id=None,
    raw=False,
    clear=False,
    **kwargs,
):
    from IPython import display

    if _is_notebook():
        return display.display(
            *objs,
            include=include,
            exclude=exclude,
            metadata=metadata,
            transient=transient,
            display_id=display_id,
            raw=raw,
            clear=clear,
            **kwargs,
        )


def _display_image(
    data=None,
    url=None,
    filename=None,
    format=None,
    embed=None,
    width=None,
    height=None,
    retina=False,
    unconfined=False,
    metadata=None,
    alt=None,
):
    from IPython import display

    if _is_notebook():
        img = display.Image(
            data=data,
            url=url,
            filename=filename,
            format=format,
            embed=embed,
            width=width,
            height=height,
            retina=retina,
            unconfined=unconfined,
            metadata=metadata,
            alt=alt,
        )
        return display.display(img)
