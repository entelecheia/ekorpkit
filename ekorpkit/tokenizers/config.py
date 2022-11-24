from enum import Enum


class ModelType(str, Enum):
    UNIGRAM = "unigram"
    BPE = "bpe"
    WORD = "word"
    CHAR = "char"
    WORDPIECE = "wordpiece"


class TrainerType(str, Enum):
    SPM = "spm"
    HF = "huggingface"


class DatasetType(str, Enum):
    DATASET = "dataset"
    TEXT = "text"
