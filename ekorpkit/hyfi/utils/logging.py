"""Logging utilities"""
import logging
import os
import sys
import warnings


def setLogger(level=None, force=True, filterwarnings_action="ignore", **kwargs):
    """Set the logging level and format"""
    level = level or os.environ.get("HYFI_LOG_LEVEL", "INFO")
    level = level.upper()
    os.environ["HYFI_LOG_LEVEL"] = level
    if filterwarnings_action is not None:
        warnings.filterwarnings(filterwarnings_action)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if sys.version_info >= (3, 8):
        logging.basicConfig(level=level, force=force, **kwargs)
    else:
        logging.basicConfig(level=level, **kwargs)


def getLogger(
    _name=None,
    _log_level=None,
    _fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    """Get a logger with a given name and log level"""
    _name = _name or __name__
    logger = logging.getLogger(_name)
    _log_level = _log_level or os.environ.get("HYFI_LOG_LEVEL", "INFO")
    logger.setLevel(_log_level)
    return logger
