import os
import dotenv
import hydra
from typing import Any
from pathlib import Path
from .logging import getLogger
from ..io.file import is_dir


logger = getLogger(__name__)


def getcwd():
    """Get the original working directory before hydra changed it"""
    try:
        return hydra.utils.get_original_cwd()
    except ValueError:
        return os.getcwd()


def dotenv_values(dotenv_path=None, **kwargs):
    config = dotenv.dotenv_values(dotenv_path=dotenv_path, **kwargs)
    return dict(config)


def load_dotenv(
    verbose: bool = False,
    override: bool = False,
):
    original_cwd = getcwd()
    config_dir = os.environ.get("EKORPKIT_CONFIG_DIR")
    dotenv_dir = config_dir or original_cwd
    dotenv_path = Path(dotenv_dir, ".env")
    if dotenv_path.is_file():
        dotenv.load_dotenv(dotenv_path=dotenv_path, verbose=verbose, override=override)
        if verbose:
            logger.info(f"Loaded .env from {dotenv_path}")
    else:
        if verbose:
            logger.info(
                f"No .env file found in {dotenv_dir}, finding .env in parent dirs"
            )
        dotenv_path = dotenv.find_dotenv()
        if dotenv_path:
            dotenv.load_dotenv(
                dotenv_path=dotenv_path, verbose=verbose, override=override
            )
            if verbose:
                logger.info(f"Loaded .env from {dotenv_path}")
        else:
            if verbose:
                logger.info(f"No .env file found in {dotenv_path}")


def get_osenv(key: str = None, default: str = None) -> Any:
    load_dotenv()
    if key:
        return os.environ.get(key, default)
    return os.environ


def set_osenv(key: str, value: Any) -> None:
    if value and is_dir(value):
        value = os.path.abspath(value)
    pre_val = os.environ.get(key)
    if pre_val:
        logger.info(f"Overwriting {key}={pre_val} with {value}")
    else:
        logger.info(f"Setting {key}={value}")
    os.environ[key] = value
