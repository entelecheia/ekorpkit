import logging
import warnings
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from enum import Enum


def _setLogger(level=None, force=True, filterwarnings_action="ignore", **kwargs):
    level = level or os.environ.get("EKORPKIT_LOG_LEVEL", "INFO")
    level = level.upper()
    os.environ["EKORPKIT_LOG_LEVEL"] = level
    if filterwarnings_action is not None:
        warnings.filterwarnings(filterwarnings_action)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if sys.version_info >= (3, 8):
        logging.basicConfig(level=level, force=force, **kwargs)
    else:
        logging.basicConfig(level=level, **kwargs)


def _getLogger(
    _name=None,
    _log_level=None,
    _fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    _name = _name or __name__
    logger = logging.getLogger(_name)
    _log_level = _log_level or os.environ.get("EKORPKIT_LOG_LEVEL", "INFO")
    logger.setLevel(_log_level)
    return logger


logger = _getLogger()


def __ekorpkit_path__():
    return Path(__file__).parent.as_posix()


OmegaConf.register_new_resolver("__ekorpkit_path__", __ekorpkit_path__)


class _SPLITS(str, Enum):
    """Split keys in configs used by Dataset."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class _Defaults(str, Enum):
    ID_SEP = "_"
    SENT_SEP = "\n"
    SEG_SEP = "\n\n"
    POS_DELIM = "\\"
    NGRAM_DELIM = ";"


class _Keys(str, Enum):
    """Special keys in configs used by ekorpkit."""

    TARGET = "_target_"
    CONVERT = "_convert_"
    RECURSIVE = "_recursive_"
    ARGS = "_args_"
    PARTIAL = "_partial_"
    CONFIG = "_config_"
    CONFIG_GROUP = "_config_group_"
    PIPELINE = "_pipeline_"
    TASK = "_task_"
    CALL = "_call_"
    EXEC = "_exec_"
    rcPARAMS = "rcParams"
    METHOD = "_method_"
    METHOD_NAME = "_name_"
    FUNC = "_func_"
    NAME = "name"
    SPLIT = "split"
    CORPUS = "corpus"
    DATASET = "dataset"
    PATH = "path"
    OUTPUT = "output"
    ID = "id"
    _ID = "_id_"
    META_MERGE_ON = "meta_merge_on"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    DATETIME = "datetime"
    X = "x"
    Y = "y"
    INDEX = "index"
    COLUMNS = "columns"
    KEY = "key"
    KEYS = "_keys_"
    DATA = "data"
    META = "meta"
    FORMAT = "format"
    VERBOSE = "verbose"
    FILE = "file"
    FILENAME = "filename"
    SUFFIX = "suffix"
    MODEL = "model"
    LOG = "log"
    PRED = "pred"
    DEFAULT = "_default_"
    EVAL = "_eval_"
    TRAIN = "_train_"
    PREDICT = "_predict_"
    PREDICTED = "predicted"
    PRED_PROBS = "pred_probs"
    ACTUAL = "actual"
    INPUT = "input"
    TARGET_TEXT = "target_text"
    MODEL_OUTPUTS = "model_outputs"
    LABELS = "labels"
    PREFIX = "prefix"
    FEATURES = "features"
    COUNT = "count"
    CLASSIFICATION = "classification"
