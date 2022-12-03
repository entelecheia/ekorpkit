import logging
import random
from omegaconf import DictConfig
from pathlib import Path
from pydantic import (
    BaseModel,
    BaseSettings,
    SecretStr,
    validator,
    root_validator,
)
from pydantic.utils import ROOT_KEY
from typing import (
    Any,
    Optional,
    Union,
)
from ekorpkit import eKonf
from .base import Environments, Secrets


log = logging.getLogger(__name__)


class BaseBatchConfig(BaseModel):
    output_dir: Path = Path.cwd() / "outputs"
    batch_name: str
    batch_num: int = None
    random_seed: bool = True
    seed: int = None
    resume_run: bool = False
    resume_latest: bool = False
    num_workers: int = 1
    config_yaml = "config.yaml"
    config_json = "config.json"
    config_dirname = "configs"
    verbose: Union[bool, int] = False

    def __init__(self, **values):
        super().__init__(**values)
        self.init_batch_num()

    def init_batch_num(self):
        if self.batch_num is None:
            num_files = len(list(self.config_dir.glob(self.config_filepattern)))
            if self.resume_latest:
                self.batch_num = num_files - 1
            else:
                self.batch_num = num_files
        if self.verbose:
            log.info(f"Batch name: {self.batch_name}, Batch num: {self.batch_num}")

    @validator("seed")
    def _validate_seed(cls, v, values):
        if values["random_seed"] or v is None or v < 0:
            random.seed()
            seed = random.randint(0, 2**32 - 1)
            if values.get("verbose"):
                log.info(f"Setting seed to {seed}")
            return seed
        return v

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


class ProjectConfig(BaseModel):
    project_name: str = "ekorpkit-project"
    project_dir: str = None
    task_name: str = "default-task"
    workspace_dir: str = None
    description: str = None
    path: DictConfig = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        validate_assignment = True

    @validator("project_name", allow_reuse=True)
    def _validate_project_name(cls, v):
        if v is None:
            raise ValueError("Project name must be specified.")
        if eKonf.osenv("WANDB_PROJECT") is None:
            eKonf.env_set("WANDB_PROJECT", v)
        return v


class BaseBatchModel(BaseModel):
    config_name: str = None
    config_group: str = None
    name: str
    path: DictConfig = None
    root_dir: Path = None
    batch: BaseBatchConfig = None
    envs: Environments = Environments()
    secrets: Secrets = Secrets()
    project: ProjectConfig = None
    dataset: DictConfig = None
    model: DictConfig = None
    module: DictConfig = None
    auto: Union[DictConfig, str] = None
    autoload: bool = False
    device: str = "cpu"
    num_devices: int = None
    version: str = "0.0.0"
    _config: DictConfig = None
    _initial_config: DictConfig = None

    def __init__(self, config_group=None, **args):
        if config_group is not None:
            args = eKonf.merge(eKonf.compose(config_group), args)
        else:
            args = eKonf.to_config(args)
        super().__init__(**args)

        object.__setattr__(self, "_config", args)
        object.__setattr__(self, "_initial_config", args.copy())
        self._init_configs()

    def __setattr__(self, key, val):
        super().__setattr__(key, val)
        if key == "name":
            self._init_configs(name=val)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
        validate_assignment = False
        exclude = {
            "_config",
            "_initial_config",
            "path",
            "module",
            "secret",
            "auto",
            "project",
        }
        underscore_attrs_are_private = True

    @property
    def config(self):
        return self._config

    @validator("path", pre=True)
    def _validate_path(cls, v):
        if v is None:
            v = eKonf.compose("path=_batch_")
            log.info(f"There is no path in the config, using default path: {v.root}")
        if v.verbose:
            eKonf.print(v)
        return v

    @validator("root_dir")
    def _validate_root_dir(cls, v, values):
        if v is None:
            v = Path(values.get("path").root)
        if isinstance(v, str):
            v = Path(v)
        return v

    @validator("batch", pre=True)
    def _validate_batch(cls, v):
        if v is None:
            v = eKonf.compose("batch")
            log.info(
                f"There is no batch in the config, using default batch: {v.batch_name}"
            )
        return v

    @root_validator(pre=False)
    def _validate_config(cls, values):
        name = values.get("name")
        batch_name = values.get("batch").batch_name
        if name != batch_name:
            raise ValueError(
                f"Model name {name} does not match batch name {batch_name}"
            )
        return values

    def _init_configs(self, name=None, path=None, root_dir=None, **kwargs):
        if name is None:
            name = self.name
        self.config.name = name
        self.config.batch.batch_name = name
        if path is None and self.path is not None:
            path = self.path
            log.info(f"Using existing path: {path.root}")
        if path is None:
            path = self.config.get("path")
            log.info(f"Using config path: {path.root}")
        if path is None:
            path = eKonf.compose("path=_batch_")
            log.info(f"There is no path in the config, using default path: {path.root}")
        if root_dir is not None:
            path.root = str(root_dir)

        if path.verbose:
            eKonf.print(path)
        self.root_dir = Path(path.root)
        self.path = path
        self.batch = BaseBatchConfig(output_dir=self.output_dir, **self.config.batch)
        self.config.batch.batch_num = self.batch.batch_num
        self.secrets.init_huggingface_hub()

    @property
    def batch_name(self):
        return self.batch.batch_name

    @property
    def batch_num(self):
        return self.batch.batch_num

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
    def seed(self):
        return self.batch.seed

    @property
    def batch_dir(self):
        return self.batch.batch_dir

    @property
    def output_dir(self):
        return Path(self.path.output_dir)

    @property
    def data_dir(self):
        return Path(self.path.data_dir)

    @property
    def model_dir(self):
        return Path(self.path.get("model_dir"))

    @property
    def cache_dir(self):
        cache_dir = self.path.get("cache_dir")
        if cache_dir is None:
            cache_dir = self.output_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
        return Path(cache_dir)

    @property
    def library_dir(self):
        return Path(self.path.library_dir)

    def autorun(self):
        return eKonf.methods(self.auto, self)

    def save_config(
        self,
        config=None,
        exclude=None,
        include=None,
    ):
        """Save the batch config"""
        if config is not None:
            self._config = config
        log.info(f"Saving config to {self.batch.config_filepath}")
        cfg = eKonf.to_dict(self.config)
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
        eKonf.save(args, self.batch.config_filepath)
        self.save_settings(exclude=exclude)
        return self.batch.config_filename

    def save_settings(self, exclude=None):
        def dumper(obj):
            if isinstance(obj, DictConfig):
                return eKonf.to_dict(obj)
            return str(obj)

        if exclude is None:
            exclude = self.__config__.exclude
        config = self.dict(exclude=exclude)
        log.info(f"Saving config to {self.batch.config_jsonpath}")
        eKonf.save_json(config, self.batch.config_jsonpath, default=dumper)

    def load_config(
        self,
        batch_name=None,
        batch_num=None,
        **args,
    ):
        """Load the config from the batch config file"""
        log.info(
            f"> Loading config for batch_name: {batch_name} batch_num: {batch_num}"
        )
        self.config.batch.batch_num = batch_num
        if batch_name is None:
            batch_name = self.batch_name
        else:
            self.name = batch_name

        if batch_num is not None:
            cfg = self._initial_config.copy()
            cfg.name = batch_name
            cfg.batch.batch_num = batch_num
            _path = self.batch.config_filepath
            if _path.is_file():
                log.info(f"Loading config from {_path}")
                batch_cfg = eKonf.load(_path)
                log.info("Merging config with the loaded config")
                cfg = eKonf.merge(cfg, batch_cfg)
            else:
                log.info(f"No config file found at {_path}")
                cfg.batch.batch_num = None
        else:
            cfg = self.config

        log.info(f"Merging config with args: {args}")
        self._config = eKonf.merge(cfg, args)
        # reinit the batch config to update the config
        self._init_configs()

        return self.config

    def show_config(self, batch_name=None, batch_num=None):
        cfg = self.load_config(batch_name, batch_num)
        eKonf.print(cfg)

    def load_modules(self):
        """Load the modules"""
        if self.module.get("modules") is None:
            log.info("No modules to load")
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
            eKonf.ensure_import_module(name, libpath, liburi, specname, syspath)

    @property
    def verbose(self):
        return self.batch.verbose


class DynamicBaseModel(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        if __pydantic_self__.__custom_root_type__ and data.keys() != {ROOT_KEY}:
            data = {ROOT_KEY: data}
        super().__init__(**data)


class LabelStudioSecrets(BaseSettings):
    api_key: Optional[str] = SecretStr
    token: Optional[str] = SecretStr
    password: Optional[str] = SecretStr

    class Config:
        env_prefix = "LABELSTUDIO_"
        env_nested_delimiter = "_"
        case_sentive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                env_settings,
                init_settings,
                file_secret_settings,
            )
