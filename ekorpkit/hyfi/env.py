import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf, SCMode
from pydantic import BaseModel, BaseSettings, SecretStr, root_validator, validator
from pydantic.env_settings import SettingsSourceCallable

from .. import _version
from .utils.batch import batcher
from .utils.env import load_dotenv
from .utils.logging import getLogger, setLogger
from .utils.notebook import _load_extentions, _set_matplotlib_formats, is_notebook

logger = getLogger(__name__)

__hydra_version_base__ = "1.2"


def __version__():
    """Returns the version of HyFI"""
    return _version.get_versions()["version"]


class AboutConfig(BaseModel):
    """About Configuration"""

    name: str = "HyFI"
    author: str = "entelecheia"
    description: str = (
        "Hydra Fast Interface (Hydra and Pydantic based interface framework)"
    )
    website: str = "https://entelecheia.cc"
    version: str = __version__()


class DistFramwork(BaseModel):
    """Distributed Framework Configuration"""

    backend: str = "joblib"
    initialize: bool = False
    num_workers: int = 1


class BatcherConfig(BaseModel):
    """Batcher Configuration"""

    procs: int = 1
    minibatch_size: int = 1_000
    backend: str = "joblib"
    task_num_cpus: int = 1
    task_num_gpus: int = 0
    verbose: int = 10


class JobLibConfig(BaseModel):
    """JobLib Configuration"""

    config_name: str = "__init__"
    distributed_framework: DistFramwork = DistFramwork()
    batcher: BatcherConfig = BatcherConfig()
    __initilized__: bool = False

    class Config:
        extra = "allow"
        underscore_attrs_are_private = True

    def __init__(
        self,
        config_name: str = "__init__",
        **data: Any,
    ):
        if not data:
            data = _compose(f"joblib={config_name}")
            logger.debug(
                "There are no arguments to initilize a config, using default config."
            )
        super().__init__(config_name=config_name, **data)

    def init_backend(
        self,
    ):
        """Initialize the backend for joblib"""
        backend = self.distributed_framework.backend

        if self.distributed_framework.initialize:
            backend_handle = None
            if backend == "ray":
                import ray

                ray_cfg = {"num_cpus": self.distributed_framework.num_workers}
                logger.info(f"initializing ray with {ray_cfg}")
                ray.init(**ray_cfg)
                backend_handle = ray

            elif backend == "dask":
                from dask.distributed import Client

                dask_cfg = {"n_workers": self.distributed_framework.num_workers}
                logger.info(f"initializing dask client with {dask_cfg}")
                client = Client(**dask_cfg)
                logger.debug(client)

            batcher.batcher_instance = batcher.Batcher(
                backend_handle=backend_handle, **self.batcher.dict()
            )
            logger.info(f"initialized batcher with {batcher.batcher_instance}")
        self.__initilized__ = True

    def stop_backend(self):
        """Stop the backend for joblib"""
        backend = self.distributed_framework.backend
        logger.info(f"stopping distributed framework")

        if self.distributed_framework.initialize:
            if backend == "ray":
                import ray

                if ray.is_initialized():
                    ray.shutdown()
                    logger.info("shutting down ray")

            # elif modin_engine == 'dask':
            #     from dask.distributed import Client

            #     if Client.initialized():
            #         client.close()
            #         log.info(f'shutting down dask client')


class PathConfig(BaseModel):
    config_name: str = "__init__"
    dotenv_path: str = None
    workspace: str = None
    project: str = "ekorpkit-default"
    data: str = None
    home: str = None
    ekorpkit: str = None
    resources: str = None
    runtime: str = None
    archive: str = None
    corpus: str = None
    datasets: str = None
    logs: str = None
    models: str = None
    outputs: str = None
    cache: str = None
    tmp: str = None
    library: str = None
    verbose: bool = False

    class Config:
        extra = "allow"

    def __init__(
        self,
        config_name: str = "__init__",
        **data: Any,
    ):
        if not data:
            data = _compose(f"path={config_name}")
            logger.debug(
                "There are no arguments to initilize a config, using default config."
            )
        super().__init__(config_name=config_name, **data)

    @property
    def log_dir(self):
        Path(self.logs).mkdir(parents=True, exist_ok=True)
        return Path(self.logs).absolute()

    @property
    def cache_dir(self):
        Path(self.cache).mkdir(parents=True, exist_ok=True)
        return Path(self.cache).absolute()


class DotEnvConfig(BaseSettings):
    """Environment variables for HyFI"""

    # Internal
    HYFI_WORKSPACE_ROOT: Optional[str]
    HYFI_PROJECT_NAME: Optional[str]
    HYFI_TASK_NAME: Optional[str]
    HYFI_PROJECT_ROOT: Optional[str]
    HYFI_DATA_ROOT: Optional[str]
    HYFI_LOG_LEVEL: Optional[str]
    HYFI_VERBOSE: Optional[Union[bool, str, int]]
    NUM_WORKERS: Optional[int]
    CACHED_PATH_CACHE_ROOT: Optional[str]
    # For other packages
    CUDA_DEVICE_ORDER: Optional[str]
    CUDA_VISIBLE_DEVICES: Optional[str]
    WANDB_PROJECT: Optional[str]
    WANDB_DISABLED: Optional[str]
    WANDB_DIR: Optional[str]
    WANDB_NOTEBOOK_NAME: Optional[str]
    WANDB_SILENT: Optional[Union[bool, str]]
    LABEL_STUDIO_SERVER: Optional[str]
    KMP_DUPLICATE_LIB_OK: Optional[str] = "True"
    TOKENIZERS_PARALLELISM: Optional[bool] = False
    # API Keys and Tokens
    WANDB_API_KEY: Optional[SecretStr]
    HUGGING_FACE_HUB_TOKEN: Optional[SecretStr]
    ECOS_API_KEY: Optional[SecretStr]
    FRED_API_KEY: Optional[SecretStr]
    NASDAQ_API_KEY: Optional[SecretStr]
    HF_USER_ACCESS_TOKEN: Optional[SecretStr]
    LABEL_STUDIO_USER_TOKEN: Optional[SecretStr]

    class Config:
        env_prefix = ""
        env_nested_delimiter = "__"
        case_sentive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        validate_assignment = True
        extra = "allow"

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            load_dotenv()
            return env_settings, file_secret_settings, init_settings

    @root_validator()
    def _check_and_set_values(cls, values):
        workspace = values.get("HYFI_WORKSPACE_ROOT")
        project = values.get("HYFI_PROJECT_NAME")
        if workspace is not None and project is not None:
            project_dir = os.path.join(workspace, "projects", project)
            values["HYFI_PROJECT_ROOT"] = project_dir
        for k, v in values.items():
            if v is not None:
                old_value = os.getenv(k.upper())
                if old_value is None or old_value != str(v):
                    os.environ[k.upper()] = str(v)
                    logger.debug(f"Set environment variable {k.upper()}={v}")
        return values


class ProjectConfig(BaseModel):
    """Project Config"""

    config_name: str = "__init__"
    project_name: str = "hyfi-project"
    task_name: str = None
    workspace_root: str = None
    project_root: str = None
    description: str = None
    use_huggingface_hub: bool = False
    use_wandb: bool = False
    version: str = __version__()
    verbose: bool = False
    # Config Classes
    dotenv: DotEnvConfig = None
    joblib: JobLibConfig = None
    path: PathConfig = None

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(
        self,
        config_name: str = "__init__",
        **data: Any,
    ):
        if not data:
            data = _compose(f"project={config_name}")
            logger.debug(
                "There are no arguments to initilize a config, using default config."
            )
        super().__init__(config_name=config_name, **data)

    @validator("project_name", allow_reuse=True)
    def _validate_project_name(cls, v):
        if v is None:
            raise ValueError("Project name must be specified.")
        return v

    @property
    def workspace_dir(self):
        return Path(self.path.workspace)

    @property
    def project_dir(self):
        return Path(self.path.project)

    def init_project(self):
        self.dotenv = DotEnvConfig()
        if self.path is None:
            self.path = PathConfig()
        if self.joblib is None:
            self.joblib = JobLibConfig()

        if self.dotenv.HYFI_VERBOSE is not None:
            self.verbose = self.dotenv.HYFI_VERBOSE
        self.dotenv.HYFI_DATA_ROOT = str(self.path.data)
        self.dotenv.CACHED_PATH_CACHE_ROOT = str(self.path.cache_dir / "cached_path")
        wandb_dir = str(self.path.log_dir)
        self.dotenv.WANDB_DIR = wandb_dir
        project_name = self.project_name.replace("/", "-").replace("\\", "-")
        self.dotenv.WANDB_PROJECT = project_name
        task_name = self.task_name.replace("/", "-").replace("\\", "-")
        notebook_name = self.path.log_dir / f"{task_name}-nb"
        notebook_name.mkdir(parents=True, exist_ok=True)
        self.dotenv.WANDB_NOTEBOOK_NAME = str(notebook_name)
        self.dotenv.WANDB_SILENT = str(not self.verbose)
        if self.use_wandb:
            try:
                import wandb

                wandb.init(project=self.project_name)
            except ImportError:
                logger.warning(
                    "wandb is not installed, please install it to use wandb."
                )
        if self.use_huggingface_hub:
            self.init_huggingface_hub()

    def init_huggingface_hub(self):
        """Initialize huggingface_hub"""
        from huggingface_hub import notebook_login
        from huggingface_hub.hf_api import HfFolder

        self.dotenv = DotEnvConfig()
        if (
            self.dotenv.HUGGING_FACE_HUB_TOKEN is None
            and self.dotenv.HF_USER_ACCESS_TOKEN is not None
        ):
            self.dotenv.HUGGING_FACE_HUB_TOKEN = self.dotenv.HF_USER_ACCESS_TOKEN

        local_token = HfFolder.get_token()
        if local_token is None:
            if is_notebook():
                notebook_login()
            else:
                logger.info(
                    "huggingface_hub.notebook_login() is only available in notebook,"
                    "set HUGGING_FACE_HUB_TOKEN manually"
                )


def _check_and_set_value(key, value):
    """Check and set value to environment variable"""
    env_key = key.upper()
    if value is not None:
        old_value = os.getenv(env_key, "")
        if str(old_value).lower() != str(value).lower():
            os.environ[env_key] = str(value)
            logger.debug("Set environment variable %s=%s", env_key, str(value))
    return value


class HyfiConfig(BaseModel):
    """HyFI config primary class"""

    hyfi_package_config_path: str = "pkg://ekorpkit.hyfi.conf"
    hyfi_config_module: str = "ekorpkit.hyfi.conf"
    hyfi_user_config_path: str = None

    debug_mode: bool = False
    print_config: bool = False
    print_resolved_config: bool = False
    verbose: bool = False
    logging_level: str = "WARNING"

    hydra: DictConfig = None

    about: AboutConfig = AboutConfig()
    project: ProjectConfig = None

    __version__: str = __version__()
    __initilized__: bool = False

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        validate_assignment = True
        extra = "allow"

    @validator("hyfi_user_config_path")
    def _validate_hyfi_user_config_path(cls, v):
        return _check_and_set_value("hyfi_user_config_path", v)

    @validator("logging_level")
    def _validate_logging_level(cls, v, values):
        verbose = values.get("verbose", False)
        if verbose and v == "WARNING":
            v = "INFO"
        logger.setLevel(v)
        return v

    def __init__(self, **data: Any):
        super().__init__(**data)

    def init_notebook(
        self,
        workspace=None,
        project=None,
        task=None,
        log_level=None,
        autotime=True,
        retina=True,
        verbose=None,
        **kwargs,
    ):
        """Initialize project in notebook"""
        envs = DotEnvConfig(HYFI_VERBOSE=verbose)
        if isinstance(workspace, str):
            envs.HYFI_WORKSPACE_ROOT = workspace
        if isinstance(project, str):
            envs.HYFI_PROJECT_NAME = project
        if isinstance(task, str):
            envs.HYFI_TASK_NAME = task
        if isinstance(log_level, str):
            envs.HYFI_LOG_LEVEL = log_level
            setLogger(log_level)
        if autotime:
            _load_extentions(exts=["autotime"])
        if retina:
            _set_matplotlib_formats("retina")
        self.initialize()

    def initialize(self, config: Union[DictConfig, Dict] = None):
        """Initialize hyfi config"""
        if self.__initilized__:
            return
        if config is None:
            config = _compose(
                overrides=["+project=__init__"], config_module=self.hyfi_config_module
            )
            logger.debug("Using default config.")
        if "project" not in config:
            logger.warning(
                "No project config found, skip project config initialization."
            )
            return

        if "about" in config:
            self.about = AboutConfig(**config["about"])
        self.project = ProjectConfig(**config["project"])
        self.project.init_project()
        self.project.joblib.init_backend()
        self.__initilized__ = True

    def terminate(self):
        """Terminate hyfi config"""
        if not self.__initilized__:
            return
        if self.project is not None:
            self.project.joblib.stop_backend()
        self.__initilized__ = False

    def __repr__(self):
        return f"HyFIConfig(project={self.project})"

    def __str__(self):
        return self.__repr__()


__global_config__ = HyfiConfig()


class Dummy:
    def __call__(self, *args, **kwargs):
        return Dummy()


def _compose(
    config_group: str = None,
    overrides: List[str] = [],
    *,
    return_as_dict: bool = False,
    throw_on_resolution_failure: bool = True,
    throw_on_missing: bool = False,
    config_name: str = None,
    config_module: str = None,
    verbose: bool = False,
) -> Union[DictConfig, Dict]:
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
    config_module = config_module or __global_config__.hyfi_config_module
    if verbose:
        logger.info("config_module: %s", config_module)
    is_initialized = hydra.core.global_hydra.GlobalHydra.instance().is_initialized()
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
        if is_initialized:
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
        else:
            with hydra.initialize_config_module(
                config_module=config_module, version_base=__hydra_version_base__
            ):
                cfg = hydra.compose(config_name=config_name, overrides=overrides)
        cfg = _select(
            cfg,
            key=key,
            default=None,
            throw_on_missing=False,
            throw_on_resolution_failure=False,
        )
        if cfg is not None:
            overide = config_group
        else:
            overide = f"+{config_group}"
        if overrides:
            overrides.append(overide)
        else:
            overrides = [overide]
    if verbose:
        logger.info(f"compose config with overrides: {overrides}")
    if is_initialized:
        if verbose:
            logger.info("Hydra is already initialized")
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        with hydra.initialize_config_module(
            config_module=config_module, version_base=__hydra_version_base__
        ):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)

    if key and key != "task":
        cfg = _select(
            cfg,
            key=key,
            default=None,
            throw_on_missing=throw_on_missing,
            throw_on_resolution_failure=throw_on_resolution_failure,
        )
    logger.debug("Composed config: %s", OmegaConf.to_yaml(cfg))
    if return_as_dict and isinstance(cfg, DictConfig):
        return _to_dict(cfg)
    return cfg


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
