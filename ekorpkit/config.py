import logging
from pydantic import (
    BaseModel,
    BaseSettings,
    SecretStr,
)
from pydantic.env_settings import SettingsSourceCallable
from pydantic.utils import ROOT_KEY
from typing import (
    Any,
    Optional,
    Tuple,
)


log = logging.getLogger(__name__)


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


class Secrets(BaseSettings):
    wandb_api_key: Optional[SecretStr]
    hf_user_access_token: Optional[SecretStr]
    ecos_api_key: Optional[SecretStr]
    fred_api_key: Optional[SecretStr]
    nasdaq_api_key: Optional[SecretStr]

    class Config:
        env_prefix = ""
        env_nested_delimiter = "__"
        case_sentive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        # extra = Extra.allow

        # @classmethod
        # def customise_sources(
        #     cls,
        #     init_settings: SettingsSourceCallable,
        #     env_settings: SettingsSourceCallable,
        #     file_secret_settings: SettingsSourceCallable,
        # ) -> Tuple[SettingsSourceCallable, ...]:
        #     return (
        #         env_settings,
        #         init_settings,
        #         file_secret_settings,
        #     )


class Environments(BaseSettings):
    EKORPKIT_CONFIG_DIR: Optional[str]
    EKORPKIT_DATA_DIR: Optional[str]
    EKORPKIT_PROJECT: Optional[str]
    EKORPKIT_WORKSPACE_ROOT: Optional[str]
    EKORPKIT_LOG_LEVEL: Optional[str]
    FRED_API_KEY: Optional[SecretStr]
    NASDAQ_API_KEY: Optional[SecretStr]
    WANDB_API_KEY: Optional[SecretStr]
    NUM_WORKERS: Optional[int]
    KMP_DUPLICATE_LIB_OK: Optional[str]
    CUDA_DEVICE_ORDER: Optional[str]
    CUDA_VISIBLE_DEVICES: Optional[str]

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> Tuple[SettingsSourceCallable, ...]:
            return env_settings, file_secret_settings
