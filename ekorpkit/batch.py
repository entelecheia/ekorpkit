import logging
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pydantic import BaseModel
from ekorpkit import eKonf
from .config import Secrets, BaseBatchConfig


log = logging.getLogger(__name__)


class BaseConfig:
    config: DictConfig = None
    _initial_config: DictConfig = None
    path: DictConfig = None
    batch: BaseBatchConfig = None
    secrets: Secrets = None

    def __init__(self, root_dir=None, **args):
        self.config = eKonf.to_config(args)
        self._initial_config = self.config.copy()
        self._init_path(root_dir=root_dir, **args)
        self._init_batch()
        self._init_secrets()

    class Config:
        arbitrary_types_allowed = True

    def update(self, key, value, merge=True, force_add=True):
        OmegaConf.update(self.config, key, value, merge=merge, force_add=force_add)

    @property
    def root_dir(self):
        return Path(self.path.root)

    def _init_path(self, path=None, root_dir=None, **kwargs):
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
            path.root = root_dir

        if path.verbose:
            eKonf.print(path)
        self.path = path

    def _init_batch(
        self,
    ):
        self.batch = BaseBatchConfig(output_dir=self.output_dir, **self.config.batch)

    def _init_secrets(self):
        if self.config.get("secret") is not None:
            for k, v in self.config.secret.items():
                if v:
                    eKonf.env_set(k, v)
        self.secrets = Secrets()

    @property
    def verbose(self):
        return self.batch.verbose

    @property
    def project_name(self):
        return self.project.project_name

    @property
    def project_dir(self):
        return Path(self.project.project_dir)

    @property
    def workspace_dir(self):
        return Path(self.project.path.workspace)

    @property
    def project(self):
        return self.config.project

    @property
    def auto(self):
        return self.config.get("auto")

    def autorun(self):
        return eKonf.methods(self.auto, self)

    @property
    def autoload(self):
        return self.config.get("autoload", False)

    @property
    def device(self):
        return self.config.get("device", "cpu")

    @property
    def num_devices(self):
        return self.config.get("num_devices")

    @num_devices.setter
    def num_devices(self, num_devices):
        self.config.num_devices = num_devices

    @property
    def version(self):
        return self.config.get("version", "0.0.0")

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
    def model(self):
        if "model" in self.config:
            return self.config.model
        return {}

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
    def name(self):
        return self.config.name

    @name.setter
    def name(self, value):
        self.batch_name = value

    @property
    def batch_name(self):
        return self.config.batch.batch_name

    @batch_name.setter
    def batch_name(self, value):
        self.config.batch.batch_name = value
        self.config.name = value

    @property
    def batch_num(self):
        return self.config.batch.batch_num

    @batch_num.setter
    def batch_num(self, value):
        self.config.batch.batch_num = value

    def save_config(
        self,
        config=None,
        exclude=["path", "module", "secret", "auto", "project"],
        include=None,
    ):
        """Save the batch config"""
        if config is not None:
            self.config = config
        log.info(f"Saving config to {self.batch.config_filepath}")
        cfg = eKonf.to_dict(self.config)
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
        self.save_settings()
        return self.batch.config_filename

    def save_settings(self):
        # interate self variables and combine basemodel objects into a dict
        # then save to json
        config = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BaseModel):
                config[key] = value.dict()
        log.info(f"Saving config to {self.batch.config_jsonpath}")
        eKonf.save_json(config, self.batch.config_jsonpath, default=str)

    def load_config(
        self,
        batch_name=None,
        batch_num=None,
        **args,
    ):
        """Load the config from the batch config file"""
        if batch_name is None:
            batch_name = self.batch_name
        else:
            self.batch_name = batch_name
        self.batch_num = batch_num

        cfg = self._initial_config
        self._init_path()
        self._init_batch()
        if batch_num is not None:
            _path = self.batch.config_filepath
            if _path.is_file():
                log.info(f"Loading config from {_path}")
                batch_cfg = eKonf.load(_path)
                log.info("Merging config with the loaded config")
                cfg = eKonf.merge(cfg, batch_cfg)

        log.info(f"Merging config with args: {args}")
        self.config = eKonf.merge(cfg, args)
        # reinit the batch config to update the config
        self._init_batch()
        if eKonf.osenv("WANDB_PROJECT") is None:
            eKonf.env_set("WANDB_PROJECT", self.project_name)

        return self.config

    def show_config(self, batch_name=None, batch_num=None):
        cfg = self.load_config(batch_name, batch_num)
        eKonf.print(cfg)

    @property
    def module(self):
        if "module" in self.config:
            return self.config.module
        return {}

    @property
    def library_dir(self):
        return Path(self.path.library_dir)

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
