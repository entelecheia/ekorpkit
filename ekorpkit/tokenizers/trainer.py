import os
import logging
from .trainers.spm import train_spm
from .trainers.hf import train_hf_tokenizer
from .config import ModelType, TrainerType
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def train_tokenizer(
    model_prefix,
    input_files,
    input_dir=None,
    output_dir="tokenizers",
    vocab_size=30000,
    model_type: ModelType = ModelType.UNIGRAM,
    trainer_type: TrainerType = TrainerType.SPM,
    character_coverage=1.0,
    num_workers=1,
    train_extremely_large_corpus=False,
    project_dir=None,
    verbose=False,
    **kwargs,
):
    if model_prefix is None:
        raise ValueError("model_prefix must be specified")
    if kwargs:
        kwargs = eKonf.to_dict(kwargs)
        log.info(f"Additional kwargs: {kwargs}")

    if project_dir is not None:
        log.info(f"Using project_dir {project_dir}")
        output_dir = os.path.join(project_dir, output_dir)
        if input_dir is not None:
            input_dir = os.path.join(project_dir, input_dir)
        else:
            input_dir = project_dir

    input_files = eKonf.get_filepaths(input_files, input_dir)

    if trainer_type == TrainerType.SPM:
        model_path = train_spm(
            model_prefix=model_prefix,
            input=input_files,
            output_dir=output_dir,
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            num_threads=num_workers,
            train_extremely_large_corpus=train_extremely_large_corpus,
            **kwargs,
        )
    elif trainer_type == TrainerType.HF:
        model_path = train_hf_tokenizer(
            model_prefix=model_prefix,
            input_files=input_files,
            output_dir=output_dir,
            vocab_size=vocab_size,
            model_type=model_type,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid trainer type: {trainer_type}")

    if verbose:
        print(f"saved model to {model_path}")
