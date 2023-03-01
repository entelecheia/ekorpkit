import random
from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import DictConfig
from pydantic import BaseModel, validator

from ..env import ProjectConfig, _to_config, _to_dict, getLogger
from ..hydra import _compose, _load, _merge, _methods, _print, _save, _save_json
from ..utils.lib import ensure_import_module

logger = getLogger(__name__)


class PathConfig(BaseModel):
    task_name: str = "default-task"
    root: str = None
    batch_name: str = None
    verbose: bool = False

    class Config:
        extra = "ignore"

    def __init__(self, **data: Any):
        if not data:
            data = _compose("path=__batch__")
            logger.info(
                "There are no arguments to initilize a config, using default config."
            )
        super().__init__(**data)

    @property
    def root_dir(self):
        # return as an absolute path
        return Path(self.root).absolute()

    @property
    def output_dir(self):
        return self.root_dir / "outputs"

    @property
    def batch_dir(self):
        return self.output_dir / self.batch_name

    @property
    def library_dir(self):
        return self.root_dir / "libs"

    @property
    def data_dir(self):
        return self.root_dir / "data"

    @property
    def model_dir(self):
        return self.root_dir / "models"

    @property
    def cache_dir(self):
        return self.root_dir / "cache"

    @property
    def tmp_dir(self):
        return self.root_dir / "tmp"

    @property
    def log_dir(self):
        log_dir = self.root_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir


class BaseBatchConfig(BaseModel):
    batch_name: str
    batch_num: int = None
    output_dir: Path = Path.cwd() / "outputs"
    output_suffix: str = None
    output_extention: Optional[str] = ""
    random_seed: bool = True
    seed: int = None
    resume_run: bool = False
    resume_latest: bool = False
    num_workers: int = 1
    device: str = "cpu"
    num_devices: Optional[int] = None
    config_yaml = "config.yaml"
    config_json = "config.json"
    config_dirname = "configs"
    verbose: Union[bool, int] = False

    def __init__(self, **data):
        if not data:
            data = _compose("batch")
            logger.info(
                f"There is no batch in the config, using default batch: {data.batch_name}"
            )
        super().__init__(**data)
        self.init_batch_num()

    def init_batch_num(self):
        if self.batch_num is None:
            num_files = len(list(self.config_dir.glob(self.config_filepattern)))
            if self.resume_latest:
                self.batch_num = num_files - 1
            else:
                self.batch_num = num_files
        if self.verbose:
            logger.info(
                f"Init batch - Batch name: {self.batch_name}, Batch num: {self.batch_num}"
            )

    @validator("seed")
    def _validate_seed(cls, v, values):
        if values["random_seed"] or v is None or v < 0:
            random.seed()
            seed = random.randint(0, 2**32 - 1)
            if values.get("verbose"):
                logger.info(f"Setting seed to {seed}")
            return seed
        return v

    @validator("output_extention")
    def _validate_output_extention(cls, v):
        if v:
            return v.strip(".")
        else:
            return ""

    @property
    def batch_dir(self):
        batch_dir = self.output_dir / self.batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    @property
    def config_dir(self):
        config_dir = self.batch_dir / self.config_dirname
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    @property
    def file_prefix(self):
        return f"{self.batch_name}({self.batch_num})"

    @property
    def output_file(self):
        if self.output_suffix:
            return f"{self.file_prefix}_{self.output_suffix}.{self.output_extention}"
        else:
            return f"{self.file_prefix}.{self.output_extention}"

    @property
    def config_filename(self):
        return f"{self.file_prefix}_{self.config_yaml}"

    @property
    def config_jsonfile(self):
        return f"{self.file_prefix}_{self.config_json}"

    @property
    def config_filepattern(self):
        return f"{self.batch_name}(*)_{self.config_yaml}"

    @property
    def config_filepath(self):
        return self.config_dir / self.config_filename

    @property
    def config_jsonpath(self):
        return self.config_dir / self.config_jsonfile


class BaseConfigModel(BaseModel):
    config_name: str = None
    config_group: str = None
    name: str
    path: PathConfig = None
    project: ProjectConfig = None
    module: DictConfig = None
    auto: Union[DictConfig, str] = None
    force: Union[DictConfig, str] = None
    autoload: bool = False
    version: str = "0.0.0"
    _config_: DictConfig = None
    _initial_config_: DictConfig = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        validate_assignment = False
        exclude = {
            "_config_",
            "_initial_config_",
            "__data__",
            "path",
            "module",
            "secret",
            "auto",
            "project",
        }
        include = {}
        underscore_attrs_are_private = True
        property_set_methods = {
            "name": "set_name",
            "root_dir": "set_root_dir",
        }

    def __init__(
        self,
        config_name: str = None,
        config_group: str = None,
        root_dir: str = None,
        **args,
    ):
        if config_group is not None:
            args = _merge(_compose(config_group), args)
        else:
            args = _to_config(args)
        super().__init__(config_name=config_name, config_group=config_group, **args)

        object.__setattr__(self, "_config_", args)
        object.__setattr__(self, "_initial_config_", args.copy())
        self.initialize_configs(root_dir=root_dir)

    def __setattr__(self, key, val):
        super().__setattr__(key, val)
        method = self.__config__.property_set_methods.get(key)
        if method is not None:
            getattr(self, method)(val)

    def set_root_dir(self, root_dir: Union[str, Path]):
        path = self.config.path
        if path is None:
            path = _compose("path=_batch_")
            logger.info(
                f"There is no path in the config, using default path: {path.root}"
            )
            self._config_.path = path
        if root_dir is not None:
            path.root = str(root_dir)
        self.path = PathConfig(**path)

    def set_name(self, val):
        self._config_.name = val
        if self.name is None or self.name != val:
            self.name = val

    def initialize_configs(self, root_dir=None, **kwargs):
        self.root_dir = root_dir

    @property
    def config(self):
        return self._config_

    @property
    def root_dir(self) -> Path:
        return Path(self.path.root)

    @property
    def output_dir(self):
        return self.path.output_dir

    @property
    def project_name(self):
        return self.project.project_name

    @property
    def project_dir(self):
        return Path(self.project.project_dir)

    @property
    def workspace_dir(self):
        return Path(self.project.workspace_dir)

    @property
    def model_dir(self):
        return self.path.model_dir

    @property
    def log_dir(self):
        return self.project.path.log_dir

    @property
    def cache_dir(self):
        cache_dir = Path(self.project.path.cache)
        if cache_dir is None:
            cache_dir = self.output_dir / ".cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
        return Path(cache_dir)

    @property
    def library_dir(self):
        return self.path.library_dir

    @property
    def verbose(self):
        return self.project.verbose

    def autorun(self):
        return _methods(self.auto, self)

    def show_config(self):
        _print(self.dict())

    def load_modules(self):
        """Load the modules"""
        if self.module.get("modules") is None:
            logger.info("No modules to load")
            return
        library_dir = self.library_dir
        for module in self.module.modules:
            name = module.name
            libname = module.libname
            liburi = module.liburi
            specname = module.specname
            libpath = library_dir / libname
            syspath = module.get("syspath")
            if syspath is not None:
                syspath = library_dir / syspath
            ensure_import_module(name, libpath, liburi, specname, syspath)

    def reset(self, objects=None):
        """Reset the memory cache"""
        if isinstance(objects, list):
            for obj in objects:
                del obj
        try:
            from ..utils.gpu import GPUMon

            GPUMon.release_gpu_memory()
        except ImportError:
            pass


class BaseBatchModel(BaseConfigModel):
    batch: BaseBatchConfig = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        validate_assignment = False
        exclude = {
            "_config_",
            "_initial_config_",
            "__data__",
            "path",
            "module",
            "secret",
            "auto",
            "project",
        }
        include = {}
        underscore_attrs_are_private = True
        property_set_methods = {
            "name": "set_name",
            "batch_name": "set_name",
            "batch_num": "set_batch_num",
            "root_dir": "set_root_dir",
        }

    def __init__(self, **args):
        super().__init__(**args)

    def set_name(self, val):
        super().set_name(val)
        self._config_.batch.batch_name = val
        self.batch.batch_name = val
        self.initialize_configs(name=val)

    def set_batch_num(self, val):
        self._config_.batch.batch_num = val
        self.batch.batch_num = val

    def initialize_configs(
        self, root_dir=None, batch_config_class=BaseBatchConfig, **kwargs
    ):
        super().initialize_configs(root_dir=root_dir, **kwargs)

        self.batch = batch_config_class(**self.config.batch)
        self.batch_num = self.batch.batch_num
        # if self.project.use_huggingface_hub:
        #     self.secrets.init_huggingface_hub()
        if self.verbose:
            logger.info(
                f"Initalized batch: {self.batch_name}({self.batch_num}) in {self.root_dir}"
            )

    @property
    def batch_name(self):
        return self.batch.batch_name

    @property
    def batch_num(self):
        return self.batch.batch_num

    @property
    def seed(self):
        return self.batch.seed

    @property
    def batch_dir(self):
        return self.batch.batch_dir

    @property
    def data_dir(self):
        return self.path.data_dir

    @property
    def verbose(self):
        return self.batch.verbose

    @property
    def device(self):
        return self.batch.device

    @property
    def num_devices(self):
        return self.batch.num_devices

    def save_config(
        self,
        config=None,
        exclude=None,
        include=None,
    ):
        """Save the batch config"""
        if config is not None:
            self._config_ = config
        logger.info(f"Saving config to {self.batch.config_filepath}")
        cfg = _to_dict(self.config)
        if exclude is None:
            exclude = self.__config__.exclude

        if include:
            args = {}
            if isinstance(include, str):
                include = [include]
            for key in include:
                args[key] = cfg[key]
        else:
            args = cfg
            if exclude:
                if isinstance(exclude, str):
                    exclude = [exclude]
                for key in exclude:
                    args.pop(key, None)
        _save(args, self.batch.config_filepath)
        self.save_settings(exclude=exclude)
        return self.batch.config_filename

    def save_settings(self, exclude=None, exclude_none=True):
        def dumper(obj):
            if isinstance(obj, DictConfig):
                return _to_dict(obj)
            return str(obj)

        if exclude is None:
            exclude = self.__config__.exclude
        config = self.dict(exclude=exclude, exclude_none=exclude_none)
        if self.verbose:
            logger.info(f"Saving config to {self.batch.config_jsonpath}")
        _save_json(config, self.batch.config_jsonpath, default=dumper)

    def load_config(
        self,
        batch_name=None,
        batch_num=None,
        **args,
    ):
        """Load the config from the batch config file"""
        if self.verbose:
            logger.info(
                f"> Loading config for batch_name: {batch_name} batch_num: {batch_num}"
            )
        # self.config.batch.batch_num = batch_num
        if batch_name is None:
            batch_name = self.batch_name

        if batch_num is not None:
            cfg = self._initial_config_.copy()
            self.batch.batch_name = batch_name
            self.batch.batch_num = batch_num
            _path = self.batch.config_filepath
            if _path.is_file():
                logger.info(f"Loading config from {_path}")
                batch_cfg = _load(_path)
                if self.verbose:
                    logger.info("Merging config with the loaded config")
                cfg = _merge(cfg, batch_cfg)
            else:
                logger.info(f"No config file found at {_path}")
                batch_num = None
        else:
            cfg = self.config

        if self.verbose:
            logger.info(f"Merging config with args: {args}")
        self._config_ = _merge(cfg, args)

        self.batch_num = batch_num
        self.batch_name = batch_name

        return self.config

    def show_config(self, batch_name=None, batch_num=None):
        self.load_config(batch_name, batch_num)
        _print(self.dict())
