import logging
from pathlib import Path
from omegaconf import SCMode, DictConfig, ListConfig
from typing import Any, List, IO, Dict, Union, Tuple
from .base import (
    DictKeyType,
    Environments,
    apply_pipe,
    __version__,
    __ekorpkit_path__,
    __home_path__,
    _config,
    _Keys,
    _Defaults,
    _SPLITS,
    _compose,
    _select,
    _to_dict,
    _to_config,
    _to_yaml,
    _to_container,
    _partial,
    _instantiate,
    _is_config,
    _is_list,
    _is_instantiatable,
    _load,
    _update,
    _merge,
    _save,
    _print,
    _methods,
    _function,
    _run,
    _load_dotenv,
    _init_env_,
    _stop_env_,
    _path,
    _dependencies,
    _ensure_list,
    _to_dateparm,
    _exists,
    _is_file,
    _is_dir,
    _mkdir,
    _join_path,
    _apply,
    _ensure_kwargs,
)

log = logging.getLogger(__name__)


class eKonf:
    """ekorpkit config primary class"""

    __version__ = __version__()
    __ekorpkit_path__ = __ekorpkit_path__()
    __home_path__ = __home_path__()
    config = _config
    Keys = _Keys
    Defaults = _Defaults
    SPLITS = _SPLITS
    env = Environments()

    def __init__(self) -> None:
        raise NotImplementedError("Use one of the static construction functions")

    @staticmethod
    def compose(
        config_group: str = None,
        overrides: List[str] = [],
        *,
        return_as_dict: bool = False,
        throw_on_resolution_failure: bool = True,
        throw_on_missing: bool = False,
        config_name="ekonf",
        verbose: bool = False,
    ):
        return _compose(
            config_group=config_group,
            overrides=overrides,
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
        return _select(
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
        return _to_dict(cfg)

    @staticmethod
    def to_config(
        cfg: Any,
    ):
        return _to_config(cfg)

    @staticmethod
    def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
        if resolve:
            cfg = _to_dict(cfg)
        return _to_yaml(cfg, resolve=resolve, sort_keys=sort_keys)

    @staticmethod
    def to_container(
        cfg: Any,
        *,
        resolve: bool = False,
        throw_on_missing: bool = False,
        enum_to_str: bool = False,
        structured_config_mode: SCMode = SCMode.DICT,
    ):
        return _to_container(
            cfg,
            resolve,
            throw_on_missing,
            enum_to_str,
            structured_config_mode,
        )

    @staticmethod
    def partial(
        config: Any = None, config_group: str = None, *args: Any, **kwargs: Any
    ) -> Any:
        return _partial(config=config, config_group=config_group, *args, **kwargs)

    @staticmethod
    def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
        return _instantiate(config, *args, **kwargs)

    @staticmethod
    def is_config(
        cfg: Any,
    ):
        return _is_config(cfg)

    @staticmethod
    def is_list(
        cfg: Any,
    ):
        return _is_list(cfg)

    @staticmethod
    def is_instantiatable(cfg: Any):
        return _is_instantiatable(cfg)

    @staticmethod
    def load(file_: Union[str, Path, IO[Any]]) -> Union[DictConfig, ListConfig]:
        return _load(file_)

    @staticmethod
    def update(_dict, _overrides):
        return _update(_dict, _overrides)

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
        return _merge(*configs)

    @staticmethod
    def save(config: Any, f: Union[str, Path, IO[Any]], resolve: bool = False) -> None:
        _save(config, f, resolve)

    @staticmethod
    def pprint(cfg: Any, resolve: bool = True, **kwargs):
        _print(cfg, resolve=resolve, **kwargs)

    @staticmethod
    def print(cfg: Any, resolve: bool = True, **kwargs):
        _print(cfg, resolve=resolve, **kwargs)

    @staticmethod
    def methods(cfg: Any, obj: object):
        _methods(cfg, obj)

    @staticmethod
    def function(cfg: Any, _name_, return_function=False, **parms):
        return _function(cfg, _name_, return_function, **parms)

    @staticmethod
    def run(config: Any, **kwargs: Any) -> Any:
        _run(config, **kwargs)

    @staticmethod
    def load_dotenv(verbose: bool = False):
        _load_dotenv(verbose)

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
        return_parent_dir: bool = False,
        cache_dir=None,
        verbose: bool = False,
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
        * ``gd`` for objects on `Google Drive`_, and
        * ``hf`` for objects or repositories on `HuggingFace Hub`_.

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
        return _path(
            url_or_filename,
            extract_archive=extract_archive,
            force_extract=force_extract,
            return_parent_dir=return_parent_dir,
            cache_dir=cache_dir,
            verbose=verbose,
        )

    @staticmethod
    def pipe(cfg, data=None):
        return apply_pipe(data, cfg)

    @staticmethod
    def dependencies(_key=None, path=None):
        return _dependencies(_key, path)

    @staticmethod
    def ensure_list(value):
        return _ensure_list(value)

    @staticmethod
    def to_dateparm(_date, _format="%Y-%m-%d"):
        return _to_dateparm(_date, _format)

    @staticmethod
    def exists(a, *p):
        return _exists(a, *p)

    @staticmethod
    def is_file(a, *p):
        return _is_file(a, *p)

    @staticmethod
    def is_dir(a, *p):
        return _is_dir(a, *p)

    @staticmethod
    def mkdir(_path: str):
        return _mkdir(_path)

    @staticmethod
    def join_path(a, *p):
        return _join_path(a, *p)

    @staticmethod
    def apply(
        func,
        series,
        description=None,
        use_batcher=True,
        minibatch_size=None,
        num_workers=None,
        verbose=False,
        **kwargs,
    ):
        return _apply(
            func,
            series,
            description=description,
            use_batcher=use_batcher,
            minibatch_size=minibatch_size,
            num_workers=num_workers,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def ensure_kwargs(_kwargs, _fn):
        return _ensure_kwargs(_kwargs, _fn)

    @staticmethod
    def save_data(
        data,
        filename=None,
        base_dir=None,
        columns=None,
        index=False,
        filetype="parquet",
        suffix=None,
        verbose=False,
        **kwargs,
    ):
        from ekorpkit.io.file import save_data

        if filename is None:
            raise ValueError("filename must be specified")
        save_data(
            data,
            filename,
            base_dir=base_dir,
            columns=columns,
            index=index,
            filetype=filetype,
            suffix=suffix,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def load_data(filename=None, base_dir=None, verbose=False, **kwargs):
        from ekorpkit.io.file import load_data

        if filename is None:
            raise ValueError("filename must be specified")
        return load_data(filename, base_dir=base_dir, verbose=verbose, **kwargs)

    @staticmethod
    def get_filepaths(
        filename_patterns=None, base_dir=None, recursive=True, verbose=True, **kwargs
    ):
        from ekorpkit.io.file import get_filepaths

        if filename_patterns is None:
            raise ValueError("filename must be specified")
        return get_filepaths(
            filename_patterns,
            base_dir=base_dir,
            recursive=recursive,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def concat_data(
        data,
        columns=None,
        add_key_as_name=False,
        name_column=Keys.NAME_KEY.value,
        ignore_index=True,
        verbose=False,
        **kwargs,
    ):
        from ekorpkit.io.file import concat_data

        return concat_data(
            data,
            columns=columns,
            add_key_as_name=add_key_as_name,
            name_column=name_column,
            ignore_index=ignore_index,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def is_dataframe(data):
        from ekorpkit.io.file import is_dataframe

        return is_dataframe(data)
