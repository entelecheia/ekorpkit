from pathlib import Path
from omegaconf import SCMode, DictConfig, ListConfig
from typing import Any, List, IO, Dict, Union, Tuple
from ekorpkit.utils.notebook import (
    _clear_output,
    _cprint,
    _create_button,
    _create_dropdown,
    _create_floatslider,
    _create_image,
    _create_radiobutton,
    _create_textarea,
    _display,
    _display_image,
    _get_display,
    _hide_code_in_slideshow,
    _is_notebook,
    _is_colab,
)
from .base import (
    __ekorpkit_path__,
    __home_path__,
    __version__,
    _apply,
    _compose,
    _Defaults,
    _dependencies,
    _dict_product,
    _dict_to_dataframe,
    _ensure_kwargs,
    _ensure_list,
    _env_set,
    _exists,
    _function,
    _getLogger,
    _getsource,
    _init_env_,
    _instantiate,
    _is_config,
    _is_dir,
    _is_file,
    _is_instantiatable,
    _is_list,
    _join_path,
    _Keys,
    _load_dotenv,
    _load,
    _load_json,
    _merge,
    _methods,
    _mkdir,
    _nvidia_smi,
    _osenv,
    _partial,
    _path,
    _pipe,
    _print,
    _records_to_dataframe,
    _run,
    _save,
    _save_json,
    _select,
    _set_cuda,
    _set_workspace,
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
    Environments,
    ProjectConfig,
    Secrets,
)
from ekorpkit.io.google import _mount_google_drive


logger = _getLogger(__name__)


class eKonfConfig:
    @property
    def envs(self):
        return Environments()

    @property
    def environ(self):
        return _osenv()

    @property
    def secrets(self):
        return Secrets()


class eKonf:
    """ekorpkit config primary class"""

    __version__ = __version__()
    __ekorpkit_path__ = __ekorpkit_path__()
    __home_path__ = __home_path__()
    os: eKonfConfig = eKonfConfig()
    book_repo = "https://github.com/entelecheia/ekorpkit-book/raw/main/"
    book_repo_assets = book_repo + "assets/"
    book_url = "https://entelecheia.github.io/ekorpkit-book/"
    book_assets_url = book_url + "assets/"
    Keys = _Keys
    Defaults = _Defaults
    SPLITS = _SPLITS

    def __init__(self) -> None:
        raise NotImplementedError("Use one of the static construction functions")

    @staticmethod
    def envs() -> Environments:
        """Return the current environments"""
        return Environments()

    @staticmethod
    def secrets() -> Secrets:
        """Return the current secrets"""
        return Secrets()

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
    ) -> Union[DictConfig, Dict]:
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
    def save_json(
        json_dict: dict,
        f: Union[str, Path, IO[Any]],
        indent=4,
        ensure_ascii=False,
        default=None,
        **kwargs,
    ):
        _save_json(json_dict, f, indent, ensure_ascii, default, **kwargs)

    @staticmethod
    def load_json(f: Union[str, Path, IO[Any]], **kwargs) -> dict:
        return _load_json(f, **kwargs)

    @staticmethod
    def pprint(cfg: Any, resolve: bool = True, **kwargs):
        _print(cfg, resolve=resolve, **kwargs)

    @staticmethod
    def print(cfg: Any, resolve: bool = True, **kwargs):
        _print(cfg, resolve=resolve, **kwargs)

    @staticmethod
    def methods(cfg: Any, obj: object, return_function=False):
        return _methods(cfg, obj, return_function)

    @staticmethod
    def function(cfg: Any, _name_, return_function=False, **parms):
        return _function(cfg, _name_, return_function, **parms)

    @staticmethod
    def run(config: Any, **kwargs: Any) -> Any:
        _run(config, **kwargs)

    @staticmethod
    def load_dotenv(
        verbose: bool = False,
        override: bool = False,
    ):
        _load_dotenv(verbose, override)

    @staticmethod
    def _init_env_(cfg, verbose=False):
        return _init_env_(cfg, verbose=verbose)

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
    def load_data(filename=None, base_dir=None, filetype=None, verbose=False, **kwargs):
        from ekorpkit.io.file import load_data

        if filename is not None:
            filename = str(filename)
        if _Keys.TARGET in kwargs:
            return eKonf.instantiate(
                kwargs,
                filename=filename,
                base_dir=base_dir,
                verbose=verbose,
                filetype=filetype,
            )
        else:
            if filename is None:
                raise ValueError("filename must be specified")
            return load_data(
                filename,
                base_dir=base_dir,
                verbose=verbose,
                filetype=filetype,
                **kwargs,
            )

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
    def osenv(key, default=None):
        return _osenv(key, default)

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
        images_or_uris,
        collage_filepath=None,
        ncols=3,
        max_images=12,
        collage_width=1200,
        padding: int = 10,
        bg_color: str = "black",
        crop_to_min_size=False,
        show_filename=False,
        filename_offset=(5, 5),
        fontname=None,
        fontsize=12,
        fontcolor="#000",
        **kwargs,
    ):
        from ekorpkit.visualize.collage import collage as _collage

        return _collage(
            images_or_uris,
            collage_filepath=collage_filepath,
            ncols=ncols,
            max_images=max_images,
            collage_width=collage_width,
            padding=padding,
            bg_color=bg_color,
            crop_to_min_size=crop_to_min_size,
            show_filename=show_filename,
            filename_offset=filename_offset,
            fontname=fontname,
            fontsize=fontsize,
            fontcolor=fontcolor,
            **kwargs,
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
        **kwargs,
    ):
        return _display(
            *objs,
            include=include,
            exclude=exclude,
            metadata=metadata,
            transient=transient,
            display_id=display_id,
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
        **kwargs,
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
            **kwargs,
        )

    @staticmethod
    def pip(
        name,
        upgrade=False,
        prelease=False,
        editable=False,
        quiet=False,
        find_links=None,
        requirement=None,
        force_reinstall=False,
        verbose=False,
        **kwargs,
    ):
        from ekorpkit.utils.lib import pip as _pip

        return _pip(
            name,
            upgrade,
            prelease,
            editable,
            quiet,
            find_links,
            requirement,
            force_reinstall,
            verbose,
            **kwargs,
        )

    @staticmethod
    def upgrade(prelease=False, quiet=False, force_reinstall=False, **kwargs):
        from ekorpkit.utils.lib import pip

        return pip(
            name="ekorpkit",
            upgrade=True,
            prelease=prelease,
            quiet=quiet,
            force_reinstall=force_reinstall,
            **kwargs,
        )

    @staticmethod
    def dict_product(dicts):
        return _dict_product(dicts)

    @staticmethod
    def get_display():
        return _get_display()

    @staticmethod
    def hide_code_in_slideshow():
        return _hide_code_in_slideshow()

    @staticmethod
    def cprint(str_color_tuples, **kwargs):
        return _cprint(str_color_tuples)

    @staticmethod
    def dict_to_dataframe(data, orient="columns", dtype=None, columns=None):
        return _dict_to_dataframe(data, orient, dtype, columns)

    @staticmethod
    def records_to_dataframe(
        data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None
    ):
        return _records_to_dataframe(data, index, exclude, columns, coerce_float, nrows)

    @staticmethod
    def create_dropdown(
        options,
        value,
        description,
        disabled=False,
        style={"description_width": "initial"},
        layout=None,
        **kwargs,
    ):
        return _create_dropdown(
            options,
            value,
            description,
            disabled,
            style,
            layout,
            **kwargs,
        )

    @staticmethod
    def create_textarea(
        value,
        description,
        placeholder="",
        disabled=False,
        style={"description_width": "initial"},
        layout=None,
        **kwargs,
    ):
        return _create_textarea(
            value,
            description,
            placeholder,
            disabled,
            style,
            layout,
            **kwargs,
        )

    @staticmethod
    def create_button(
        description, button_style="", icon="check", layout=None, **kwargs
    ):
        return _create_button(description, button_style, icon, layout, **kwargs)

    @staticmethod
    def create_radiobutton(
        options,
        description,
        value=None,
        disabled=False,
        style={"description_width": "initial"},
        layout=None,
        **kwargs,
    ):
        return _create_radiobutton(
            options,
            description,
            value,
            disabled,
            style,
            layout,
            **kwargs,
        )

    @staticmethod
    def create_image(
        filename=None,
        format=None,
        width=None,
        height=None,
        **kwargs,
    ):
        return _create_image(filename, format, width, height, **kwargs)

    @staticmethod
    def create_floatslider(
        min=0.0,
        max=1.0,
        step=0.1,
        value=None,
        description="",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=None,
        **kwargs,
    ):
        return _create_floatslider(
            min,
            max,
            step,
            value,
            description,
            disabled,
            continuous_update,
            orientation,
            readout,
            readout_format,
            style,
            layout,
            **kwargs,
        )

    @staticmethod
    def get_image_font(fontname=None, fontsize=12):
        from ekorpkit.visualize.collage import get_image_font

        return get_image_font(fontname, fontsize)

    @staticmethod
    def read(uri, mode="rb", encoding=None, head=None, **kwargs):
        from ekorpkit.io.file import read as _read

        return _read(uri, mode, encoding, head, **kwargs)

    @staticmethod
    def load_image(
        image_or_uri,
        max_width: int = None,
        max_height: int = None,
        max_pixels: int = None,
        scale: float = 1.0,
        resize_to_multiple_of: int = None,
        crop_box=None,
        mode="RGB",
        **kwargs,
    ):
        from ekorpkit.visualize.utils import load_image as _load_image

        return _load_image(
            image_or_uri,
            max_width,
            max_height,
            max_pixels,
            scale,
            resize_to_multiple_of,
            crop_box,
            mode,
            **kwargs,
        )

    @staticmethod
    def set_workspace(
        workspace=None,
        project=None,
        task=None,
        log_level=None,
        autotime=True,
        retina=True,
        verbose=None,
        **kwargs,
    ) -> ProjectConfig:
        return _set_workspace(
            workspace,
            project,
            task,
            log_level,
            autotime,
            retina,
            verbose,
            **kwargs,
        )

    @staticmethod
    def scale_image(
        image,
        max_width: int = None,
        max_height: int = None,
        max_pixels: int = None,
        scale: float = 1.0,
        resize_to_multiple_of: int = 8,
        resample: int = 1,
    ):
        """
        Scale an image to a maximum width, height, or number of pixels.

        resample:   Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2),
                    Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
        """
        from ekorpkit.visualize.utils import scale_image as _scale_image

        return _scale_image(
            image,
            max_width,
            max_height,
            max_pixels,
            scale,
            resize_to_multiple_of,
            resample,
        )

    @staticmethod
    def copy(src, dst, *, follow_symlinks=True):
        import shutil

        src = str(src)
        dst = str(dst)
        _mkdir(dst)
        shutil.copy(src, dst, follow_symlinks=follow_symlinks)
        logger.info(f"copied {src} to {dst}")

    @staticmethod
    def copyfile(src, dst, *, follow_symlinks=True):
        import shutil

        src = str(src)
        dst = str(dst)
        shutil.copyfile(src, dst, follow_symlinks=follow_symlinks)
        logger.info(f"copied {src} to {dst}")

    @staticmethod
    def gpu_usage(all=False, attrList=None, useOldCode=False):
        from GPUtil import showUtilization

        return showUtilization(all, attrList, useOldCode)
