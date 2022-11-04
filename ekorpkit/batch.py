import logging
import random
from omegaconf import OmegaConf
from pathlib import Path
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class BaseConfig:
    _config_file_ = "config.yaml"
    _config_dir_ = "configs"
    _batch_name_ = "demo"
    _batch_num_ = None
    _config_ = None
    _path_ = None

    def __init__(
        self, root_dir=None, config_file="config.yaml", config_dir="configs", **args
    ):
        self._config_file_ = config_file
        self._config_dir_ = config_dir
        self.config = args
        self.verbose = args.get("verbose", False)
        self._init_path(root_dir=root_dir, **args)
        self._init_batch(**args)

    @property
    def name(self):
        return self.config.name

    @property
    def path(self):
        return self._path_

    @path.setter
    def path(self, value):
        # self.update("path", value)
        self._path_ = value

    @property
    def config(self):
        return self._config_

    @config.setter
    def config(self, cfg):
        self._config_ = eKonf.to_config(cfg)

    def update(self, key, value, merge=True, force_add=True):
        OmegaConf.update(self._config_, key, value, merge=merge, force_add=force_add)

    @property
    def config_file(self):
        return self._config_file_

    @property
    def config_dir(self):
        return self._config_dir_

    @property
    def set_seed(self):
        return self.config.batch.get("set_seed") or "random_seed"

    @property
    def seed(self):
        return self.config.batch.get("seed")

    @seed.setter
    def seed(self, value):
        self.config.batch.seed = value

    @property
    def resume_run(self):
        return self.config.batch.get("resume_run") or False

    @property
    def run_to_resume(self):
        return self.config.batch.get("run_to_resume") or "latest"

    @property
    def root_dir(self):
        return Path(self.path.root)

    @property
    def ouput_dir(self):
        return Path(self.path.output_dir)

    @property
    def batch_dir(self):
        batch_dir = self.ouput_dir / self.batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    @property
    def batch_config_dir(self):
        batch_config_dir = self.batch_dir / self.config_dir
        batch_config_dir.mkdir(parents=True, exist_ok=True)
        return batch_config_dir

    @property
    def batch_name(self):
        return self.config.batch.batch_name

    @batch_name.setter
    def batch_name(self, value):
        self.config.batch.batch_name = value

    @property
    def batch_num(self):
        return self.config.batch.batch_num

    @batch_num.setter
    def batch_num(self, value):
        self.config.batch.batch_num = value

    @property
    def batch_file_prefix(self):
        return f"{self.batch_name}({self.batch_num})"

    @property
    def batch_config_file(self):
        return f"{self.batch_file_prefix}_{self.config_file}"

    @property
    def batch_config_pattern(self):
        return f"{self.batch_name}(*)_{self.config_file}"

    @property
    def batch_config_filepath(self):
        return self.batch_config_dir / self.batch_config_file

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
        # path.root = Path(path.root)
        # for _name, _path in path.items():
        #     if _name.endswith("_dir") or _name.endswith("_path"):
        #         _path = Path(_path)
        #         if _path.is_absolute():
        #             path[_name] = _path
        #         else:
        #             path[_name] = path.root / _path
        #     if _name.endswith("_dir") and not _path.is_dir():
        #         _path.mkdir(parents=True, exist_ok=True)
        # path.batch.base_dir = path.batch_dir

        if path.verbose:
            eKonf.print(path)
        self.path = path

    def _init_batch(self, batch_name=None, batch_num=None, **kwargs):
        if batch_name is None:
            batch_name = self.batch_name
        else:
            self.batch_name = batch_name

        if batch_num is not None:
            self.batch_num = batch_num
        else:
            num_files = len(list(self.batch_config_dir.glob(self.batch_config_pattern)))
            if self.resume_run:
                if (
                    isinstance(self.run_to_resume, str)
                    and self.run_to_resume.lower() == "latest"
                ):
                    self.batch_num = num_files - 1
                else:
                    self.batch_num = int(self.run_to_resume)
            else:
                self.batch_num = num_files
        log.info(f"Batch name: {self.batch_name}, Batch num: {self.batch_num}")

    def save_config(self, config=None, exclude=["path", "module"], selected=None):
        """Save the batch config"""
        if config is not None:
            self.config = config
        log.info(f"Saving config to {self.batch_config_filepath}")
        cfg = eKonf.to_dict(self.config)
        if selected:
            args = {}
            if isinstance(selected, str):
                selected = [selected]
            for key in selected:
                args[key] = cfg[key]
        else:
            args = cfg
            if exclude:
                if isinstance(exclude, str):
                    exclude = [exclude]
                for key in exclude:
                    args.pop(key, None)
        eKonf.save(args, self.batch_config_filepath)
        return self.batch_config_file

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

        cfg = self.config
        self._init_path()
        self._init_batch(batch_name=batch_name, batch_num=batch_num)

        if batch_num is not None:
            _path = self.batch_config_filepath
            if _path.is_file():
                log.info(f"Loading config from {_path}")
                batch_cfg = eKonf.load(_path)
                log.info("Merging config with the loaded config")
                cfg = eKonf.merge(cfg, batch_cfg)

        log.info(f"Merging config with args: {args}")
        self.config = eKonf.merge(cfg, args)

        seed = self.seed
        if isinstance(self.set_seed, str) and self.set_seed.lower() == "random_seed":
            random.seed()
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = int(self.set_seed)
        self.seed = seed
        log.info(f"Setting seed to {seed}")
        return self.config

    def show_config(self, batch_name=None, batch_num=None):
        cfg = self.load_config(batch_name, batch_num)
        eKonf.print(cfg)
