import os
import hydra
from omegaconf import OmegaConf, SCMode, DictConfig, ListConfig
from pydantic.env_settings import SettingsSourceCallable
from pydantic import BaseModel, BaseSettings, SecretStr, root_validator, validator
from pydantic.utils import ROOT_KEY
from omegaconf import DictConfig
from typing import Any, Union, Tuple, Optional, List, Dict
from pathlib import Path
from .utils.env import load_dotenv
from .utils.logging import getLogger
from .utils.notebook import is_notebook
from .utils.env import load_dotenv
from .. import _version


logger = getLogger(__name__)

__hydra_version_base__ = "1.2"


def __version__():
    return _version.get_versions()["version"]


class HyfiConfig(BaseSettings):
    hyfi_package_config_path: str = "pkg://ekorpkit.hyfi.conf"
    hyfi_config_module: str = "ekorpkit.hyfi.conf"
    hyfi_user_config_path: str = None
    __hyfi_env_initilized__: bool = False

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        validate_assignment = True
        extra = "allow"

    @root_validator()
    def _check_and_set_values(cls, values):
        for k, v in values.items():
            if k == "hyfi_package_config_path":
                values["hyfi_config_module"] = v.replace("pkg://", "")
            if v is not None:
                old_value = os.getenv(k.upper(), "")
                if str(old_value).lower() != str(v).lower():
                    os.environ[k.upper()] = str(v)
                    logger.debug("Set environment variable %s=%s", k.upper(), str(v))
        return values


__global_config__ = HyfiConfig()


class Environments(BaseSettings):
    """Environment variables for ekorpkit"""

    EKORPKIT_CONFIG_DIR: Optional[str]
    EKORPKIT_WORKSPACE_ROOT: Optional[str]
    EKORPKIT_PROJECT_NAME: Optional[str]
    EKORPKIT_TASK_NAME: Optional[str]
    EKORPKIT_PROJECT_ROOT: Optional[str]
    EKORPKIT_DATA_ROOT: Optional[str]
    EKORPKIT_LOG_LEVEL: Optional[str]
    EKORPKIT_VERBOSE: Optional[Union[bool, str, int]]
    NUM_WORKERS: Optional[int]
    KMP_DUPLICATE_LIB_OK: Optional[str]
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
    CACHED_PATH_CACHE_ROOT: Optional[str]

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
        verbose = values.get("EKORPKIT_VERBOSE")
        workspace = values.get("EKORPKIT_WORKSPACE_ROOT")
        project = values.get("EKORPKIT_PROJECT_NAME")
        if workspace is not None and project is not None:
            project_dir = os.path.join(workspace, "projects", project)
            values["EKORPKIT_PROJECT_ROOT"] = project_dir
        for k, v in values.items():
            if v is not None:
                old_value = os.getenv(k.upper())
                if old_value is None or old_value != str(v):
                    os.environ[k.upper()] = str(v)
                    if verbose:
                        logger.info(f"Set environment variable {k.upper()}={v}")
        return values


class Secrets(BaseSettings):
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
        for k, v in values.items():
            if v is not None:
                old_value = os.getenv(k.upper())
                if old_value is None or old_value != v.get_secret_value():
                    os.environ[k.upper()] = v.get_secret_value()
                    logger.info(f"Set environment variable {k.upper()}={v}")
        return values

    def init_huggingface_hub(self):
        from huggingface_hub import notebook_login
        from huggingface_hub.hf_api import HfFolder

        if (
            self.HUGGING_FACE_HUB_TOKEN is None
            and self.HF_USER_ACCESS_TOKEN is not None
        ):
            self.HUGGING_FACE_HUB_TOKEN = self.HF_USER_ACCESS_TOKEN

        local_token = HfFolder.get_token()
        if local_token is None:
            if is_notebook():
                notebook_login()
            else:
                logger.info(
                    "huggingface_hub.notebook_login() is only available in notebook,"
                    "set HUGGING_FACE_HUB_TOKEN manually"
                )


class ProjectPathConfig(BaseModel):
    config_name: str = None
    config_module: str = "ekorpkit.hyfi.conf"
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
        config_module: str = "ekorpkit.hyfi.conf",
        **data: Any,
    ):
        if not data:
            data = _compose(f"path={config_name}", config_module=config_module)
            logger.debug(
                "There are no arguments to initilize a config, using default config."
            )
        super().__init__(config_name=config_name, config_module=config_module, **data)

    @property
    def log_dir(self):
        Path(self.logs).mkdir(parents=True, exist_ok=True)
        return Path(self.logs).absolute()

    @property
    def cache_dir(self):
        Path(self.cache).mkdir(parents=True, exist_ok=True)
        return Path(self.cache).absolute()


class ProjectConfig(BaseModel):
    config_name: str = None
    config_module: str = "ekorpkit.hyfi.conf"
    project_name: str = "ekorpkit-project"
    task_name: str = None
    workspace_root: str = None
    project_root: str = None
    description: str = None
    use_huggingface_hub: bool = False
    use_wandb: bool = False
    version: str = __version__()
    path: ProjectPathConfig = None
    env: DictConfig = None
    verbose: bool = False

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(
        self,
        config_name: str = "__init__",
        config_module: str = "ekorpkit.hyfi.conf",
        **data: Any,
    ):
        if not data:
            data = _compose(f"project={config_name}", config_module=config_module)
            logger.debug(
                "There are no arguments to initilize a config, using default config."
            )
        super().__init__(config_name=config_name, config_module=config_module, **data)

        if self.envs.EKORPKIT_VERBOSE is not None:
            self.verbose = self.envs.EKORPKIT_VERBOSE
        self.envs.EKORPKIT_DATA_ROOT = str(self.path.data)
        self.envs.CACHED_PATH_CACHE_ROOT = str(self.path.cache_dir / "cached_path")
        wandb_dir = str(self.path.log_dir)
        self.envs.WANDB_DIR = wandb_dir
        project_name = self.project_name.replace("/", "-").replace("\\", "-")
        self.envs.WANDB_PROJECT = project_name
        task_name = self.task_name.replace("/", "-").replace("\\", "-")
        notebook_name = self.path.log_dir / f"{task_name}-nb"
        notebook_name.mkdir(parents=True, exist_ok=True)
        self.envs.WANDB_NOTEBOOK_NAME = str(notebook_name)
        self.envs.WANDB_SILENT = str(not self.verbose)
        if self.use_wandb:
            try:
                import wandb

                wandb.init(project=self.project_name)
            except ImportError:
                logger.warning(
                    "wandb is not installed, please install it to use wandb."
                )
        if self.use_huggingface_hub:
            self.secrets.init_huggingface_hub()

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

    @property
    def envs(self):
        return Environments()

    @property
    def secrets(self):
        return Secrets()


class DynamicBaseModel(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        if __pydantic_self__.__custom_root_type__ and data.keys() != {ROOT_KEY}:
            data = {ROOT_KEY: data}
        super().__init__(**data)


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
