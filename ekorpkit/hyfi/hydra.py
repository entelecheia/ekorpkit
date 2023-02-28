import functools
import importlib
import inspect
import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import IO, Any, Dict, List, Tuple, Union

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf, SCMode

from .env import (
    ProjectConfig,
    __global_config__,
    __hydra_version_base__,
    __version__,
    _compose,
    _select,
    _to_config,
    _to_dict,
)
from .io.cached_path import _path
from .io.file import _check_path, _exists, _join_path, _mkdir
from .utils.batch import batcher
from .utils.env import dotenv_values, getcwd
from .utils.func import _strptime, _today, lower_case_with_underscores
from .utils.logging import getLogger
from .utils.notebook import is_notebook

logger = getLogger(__name__)


DictKeyType = Union[str, int, Enum, float, bool]


class _SpecialKeys(str, Enum):
    """Special keys in configs used by hyfi."""

    CALL = "_call_"
    CONFIG = "_config_"
    CONFIG_GROUP = "_config_group_"
    EXEC = "_exec_"
    FUNC = "_func_"
    METHOD = "_method_"
    METHOD_NAME = "_name_"
    NAME = "name"
    PARTIAL = "_partial_"
    rcPARAMS = "rcParams"
    RECURSIVE = "_recursive_"
    TARGET = "_target_"
    VERBOSE = "verbose"


_config_ = _compose().copy()


def _print(cfg: Any, resolve: bool = True, **kwargs):
    import pprint

    if _is_config(cfg):
        if resolve:
            pprint.pprint(_to_dict(cfg), **kwargs)
        else:
            pprint.pprint(cfg, **kwargs)
    else:
        print(cfg)


def _is_config(
    cfg: Any,
):
    return isinstance(cfg, (DictConfig, dict))


def _is_list(
    cfg: Any,
):
    return isinstance(cfg, (ListConfig, list))


def _is_instantiatable(cfg: Any):
    return _is_config(cfg) and _SpecialKeys.TARGET in cfg


def _load(file_: Union[str, Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
    return OmegaConf.load(file_)


def _save(config: Any, f: Union[str, Path, IO[Any]], resolve: bool = False) -> None:
    os.makedirs(os.path.dirname(f), exist_ok=True)
    OmegaConf.save(config, f, resolve=resolve)


def _save_json(
    json_dict: dict,
    f: Union[str, Path, IO[Any]],
    indent=4,
    ensure_ascii=False,
    default=None,
    **kwargs,
):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as f:
        json.dump(
            json_dict,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=default,
            **kwargs,
        )


def _load_json(f: Union[str, Path, IO[Any]], **kwargs) -> dict:
    with open(f, "r") as f:
        return json.load(f, **kwargs)


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
    _config_ = config.get(_SpecialKeys.CONFIG)
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
    kwargs[_SpecialKeys.PARTIAL] = True
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
    verbose = config.get(_SpecialKeys.VERBOSE, False)
    if not __global_config__.__initilized__:
        __global_config__.initialize()
    if not _is_instantiatable(config):
        if verbose:
            logger.info("Config is not instantiatable, returning config")
        return config
    _recursive_ = config.get(_SpecialKeys.RECURSIVE, False)
    if _SpecialKeys.RECURSIVE not in kwargs:
        kwargs[_SpecialKeys.RECURSIVE.value] = _recursive_
    if verbose:
        logger.info("instantiating %s ...", config.get(_SpecialKeys.TARGET))
    return hydra.utils.instantiate(config, *args, **kwargs)


def _methods(cfg: Any, obj: object, return_function=False):
    cfg = _to_dict(cfg)
    if not cfg:
        logger.info("No method defined to call")
        return

    if isinstance(cfg, dict) and _SpecialKeys.METHOD in cfg:
        _method_ = cfg[_SpecialKeys.METHOD]
    elif isinstance(cfg, dict):
        _method_ = cfg
    elif isinstance(cfg, str):
        _method_ = cfg
        cfg = {}
    else:
        raise ValueError(f"Invalid method: {cfg}")

    if isinstance(_method_, str):
        _fn = getattr(obj, _method_)
        if return_function:
            logger.info(f"Returning function {_fn}")
            return _fn
        logger.info(f"Calling {_method_}")
        return _fn(**cfg)
    elif isinstance(_method_, dict):
        if _SpecialKeys.CALL in _method_:
            _call_ = _method_.pop(_SpecialKeys.CALL)
        else:
            _call_ = True
        if _call_:
            _fn = getattr(obj, _method_[_SpecialKeys.METHOD_NAME])
            _parms = _method_.pop(_SpecialKeys.rcPARAMS, {})
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
                if _SpecialKeys.CALL in _each_method:
                    _call_ = _each_method.pop(_SpecialKeys.CALL)
                else:
                    _call_ = True
                if _call_:
                    getattr(obj, _each_method[_SpecialKeys.METHOD_NAME])(
                        **_each_method[_SpecialKeys.rcPARAMS]
                    )
                else:
                    logger.info(f"Skipping call to {_each_method}")


def _function(cfg: Any, _name_, return_function=False, **parms):
    cfg = _to_dict(cfg)
    if not isinstance(cfg, dict):
        logger.info("No function defined to execute")
        return None

    if _SpecialKeys.FUNC not in cfg:
        logger.info("No function defined to execute")
        return None

    _functions_ = cfg[_SpecialKeys.FUNC]
    fn = _partial(_functions_[_name_])
    if _name_ in cfg:
        _parms = cfg[_name_]
        _parms = {**_parms, **parms}
    else:
        _parms = parms
    if _SpecialKeys.EXEC in _parms:
        _exec_ = _parms.pop(_SpecialKeys.EXEC)
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


def __hyfi_path__():
    return Path(__file__).parent.as_posix()


def __home_path__():
    return Path.home().as_posix()


def __search_package_path__():
    return __global_config__.hyfi_package_config_path


OmegaConf.register_new_resolver("__hyfi_path__", __hyfi_path__)
OmegaConf.register_new_resolver("__search_package_path__", __search_package_path__)
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


def _getsource(obj):
    """Return the source code of the object."""
    try:
        if _is_config(obj):
            if _SpecialKeys.TARGET in obj:
                target_string = obj[_SpecialKeys.TARGET]
                mod_name, object_name = target_string.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                obj = getattr(mod, object_name)
        elif isinstance(obj, str):
            mod_name, object_name = obj.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, object_name)
        return inspect.getsource(obj)
    except Exception as e:
        logger.error(f"Error getting source: {e}")
        return ""


def _viewsource(obj):
    """Print the source code of the object."""
    print(_getsource(obj))


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
