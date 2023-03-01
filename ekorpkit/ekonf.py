from enum import Enum
from pathlib import Path

from hyfi import HyFI
from omegaconf import OmegaConf


def __ekorpkit_path__():
    return Path(__file__).parent.as_posix()


OmegaConf.register_new_resolver("__ekorpkit_path__", __ekorpkit_path__)


class SPLITS(str, Enum):
    """Split keys in configs used by Dataset."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Defaults(str, Enum):
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


class eKonf(HyFI):
    """ekorpkit config primary class"""

    __ekorpkit_path__ = __ekorpkit_path__()
    book_repo = "https://github.com/entelecheia/ekorpkit-book/raw/main/"
    book_repo_assets = book_repo + "assets/"
    book_url = "https://entelecheia.github.io/ekorpkit-book/"
    book_assets_url = book_url + "assets/"

    Defaults = Defaults
    Keys = _Keys
    SPLITS = SPLITS
