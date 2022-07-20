from pathlib import Path
from omegaconf import SCMode, DictConfig, ListConfig
from typing import Any, List, IO, Dict, Union, Tuple
from .base import (
    __ekorpkit_path__,
    __home_path__,
    __version__,
    _apply,
    _clear_output,
    _compose,
    _config,
    _Defaults,
    _dependencies,
    _display,
    _display_image,
    _ensure_kwargs,
    _ensure_list,
    _env_set,
    _exists,
    _function,
    _getLogger,
    _getsource,
    _init_env_,
    _instantiate,
    _is_colab,
    _is_config,
    _is_dir,
    _is_file,
    _is_instantiatable,
    _is_list,
    _is_notebook,
    _join_path,
    _Keys,
    _load_dotenv,
    _load,
    _merge,
    _methods,
    _mkdir,
    _nvidia_smi,
    _osenv,
    _partial,
    _path,
    _pipe,
    _print,
    _run,
    _save,
    _select,
    _set_cuda,
    _setLogger,
    _SPLITS,
    _stop_env_,
    _to_config,
    _to_container,
    _to_dateparm,
    _to_datetime,
    _to_dict,
    _to_numeric,
    _to_yaml,
    _update,
    _viewsource,
    DictKeyType,
)
from ekorpkit.io.google import _mount_google_drive
from ekorpkit.config import Environments


logger = _getLogger(__name__)


class eKonf:
    """ekorpkit config primary class"""

    __version__ = __version__()
    __ekorpkit_path__ = __ekorpkit_path__()
    __home_path__ = __home_path__()
    config = _config
    Keys = _Keys
    Defaults = _Defaults
    SPLITS = _SPLITS

    def __init__(self) -> None:
        raise NotImplementedError("Use one of the static construction functions")

    @staticmethod
    def env() -> Environments:
        """Return the current environments"""
        return Environments()

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
    def methods(cfg: Any, obj: object, return_function=False):
        _methods(cfg, obj, return_function)

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
    def pipe(data=None, cfg=None):
        return _pipe(data, cfg)

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

        if _Keys.TARGET in kwargs:
            return eKonf.instantiate(
                kwargs, filename=filename, base_dir=base_dir, verbose=verbose
            )
        else:
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
        name_column="_name_",
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

    @staticmethod
    def is_colab():
        return _is_colab()

    @staticmethod
    def is_notebook():
        return _is_notebook()

    @staticmethod
    def osenv(key):
        return _osenv(key)

    @staticmethod
    def env_set(key, value):
        return _env_set(key, value)

    @staticmethod
    def nvidia_smi():
        return _nvidia_smi()

    @staticmethod
    def ensure_import_module(name, libpath, liburi, specname=None, syspath=None):
        from ekorpkit.utils.lib import ensure_import_module

        return ensure_import_module(name, libpath, liburi, specname, syspath)

    @staticmethod
    def collage(
        image_filepaths=None,
        filename_patterns=None,
        base_dir=None,
        output_filepath=None,
        ncols=2,
        num_images=None,
        figsize=(30, 20),
        dpi=300,
        title=None,
        title_fontsize=12,
        show_filename=False,
        filename_offset=(5, 5),
        fontname=None,
        fontsize=12,
        fontcolor=None,
    ):
        from ekorpkit.visualize.collage import collage as _collage

        return _collage(
            image_filepaths=image_filepaths,
            filename_patterns=filename_patterns,
            base_dir=base_dir,
            output_filepath=output_filepath,
            ncols=ncols,
            num_images=num_images,
            figsize=figsize,
            dpi=dpi,
            title=title,
            title_fontsize=title_fontsize,
            show_filename=show_filename,
            filename_offset=filename_offset,
            fontname=fontname,
            fontsize=fontsize,
            fontcolor=fontcolor,
        )

    @staticmethod
    def make_gif(
        image_filepaths=None,
        filename_patterns=None,
        base_dir=None,
        output_filepath=None,
        duration=100,
        loop=0,
        width=None,
        optimize=True,
        quality=50,
        show=False,
        force=False,
        **kwargs,
    ):
        from ekorpkit.visualize.motion import make_gif as _make_gif

        return _make_gif(
            image_filepaths=image_filepaths,
            filename_patterns=filename_patterns,
            base_dir=base_dir,
            output_filepath=output_filepath,
            duration=duration,
            loop=loop,
            width=width,
            optimize=optimize,
            quality=quality,
            show=show,
            force=force,
            **kwargs,
        )

    @staticmethod
    def to_datetime(data, _format=None, _columns=None, **kwargs):
        return _to_datetime(data, _format, _columns, **kwargs)

    @staticmethod
    def to_numeric(data, _columns=None, errors="coerce", downcast=None, **kwargs):
        return _to_numeric(data, _columns, errors, downcast, **kwargs)

    @staticmethod
    def getLogger(
        name=None,
        log_level=None,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ):
        return _getLogger(name, log_level, fmt)

    @staticmethod
    def setLogger(level=None, force=True, **kwargs):
        return _setLogger(level, force, **kwargs)

    @staticmethod
    def set_cuda(device=0):
        return _set_cuda(device)

    @staticmethod
    def mount_google_drive(
        workspace=None,
        project=None,
        mountpoint="/content/drive",
        force_remount=False,
        timeout_ms=120000,
    ):
        return _mount_google_drive(
            workspace, project, mountpoint, force_remount, timeout_ms
        )

    @staticmethod
    def getsource(obj):
        return _getsource(obj)

    @staticmethod
    def viewsource(obj):
        return _viewsource(obj)

    @staticmethod
    def clear_output(wait=False):
        return _clear_output(wait)

    @staticmethod
    def display(
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
        return _display(
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

    @staticmethod
    def display_image(
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
        return _display_image(
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
