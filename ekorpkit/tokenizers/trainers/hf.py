import os
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import StripAccents, NFKC
from ..config import ModelType, TrainerType


log = logging.getLogger(__name__)

UNK_TOKEN = "<UNK>"  # token for unknown words
SPECIAL_TOKENS = ["<UNK>", "<SEP>", "<MASK>", "<CLS>", "[MASK]"]  # special tokens


def prepare_tokenizer_trainer(
    model_type: ModelType,
    vocab_size=30000,
    unk_token=UNK_TOKEN,
    special_tokens=SPECIAL_TOKENS,
    **kwargs,
):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if model_type == ModelType.BPE:
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens, **kwargs
        )
    elif model_type == ModelType.UNIGRAM:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            unk_token=unk_token,
            special_tokens=special_tokens,
            **kwargs,
        )
    else:
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens, **kwargs
        )

    normalizer = normalizers.Sequence([NFKC(), StripAccents()])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = Whitespace()

    return tokenizer, trainer


def train_hf_tokenizer(
    model_prefix,
    input_files,
    output_dir,
    vocab_size=30000,
    model_type: ModelType = ModelType.UNIGRAM,
    unk_token=UNK_TOKEN,
    special_tokens=SPECIAL_TOKENS,
    **kwargs,
):
    """
    Takes the files and trains the tokenizer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_name = f"{model_prefix}_{model_type}_{TrainerType.HF}_vocab_{vocab_size}.json"
    model_path = os.path.join(output_dir, model_name)

    tokenizer, trainer = prepare_tokenizer_trainer(
        model_type, vocab_size, unk_token, special_tokens, **kwargs
    )

    tokenizer.train(
        input_files,
        trainer=trainer,
    )
    tokenizer.save(model_path)
    log.info(f"saved model to {model_path}")
    return model_path
