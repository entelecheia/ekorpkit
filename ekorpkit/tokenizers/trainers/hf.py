import os
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer
from tokenizers import (
    Regex,
    normalizers,
    pre_tokenizers,
)
from ..config import ModelType, TrainerType


log = logging.getLogger(__name__)

UNK_TOKEN = "<unk>"  # token for unknown words
SPECIAL_TOKENS = [
    "<s>",
    "</s>",
    "<mask>",
    "<pad>",
    "<cls>",
    "<sep>",
    "<unk>",
]  # special tokens


def prepare_tokenizer_trainer(
    model_type: ModelType,
    vocab_size=30000,
    unk_token=UNK_TOKEN,
    special_tokens=SPECIAL_TOKENS,
    lowercase=True,
    whitespace_token="▁",
    add_prefix_space=True,
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
        pre_tokenizers_ = [
            pre_tokenizers.Whitespace(),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.UnicodeScripts(),
        ]
    elif model_type == ModelType.UNIGRAM:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            unk_token=unk_token,
            special_tokens=special_tokens,
            **kwargs,
        )
        pre_tokenizers_ = [
            pre_tokenizers.Metaspace(
                replacement=whitespace_token, add_prefix_space=add_prefix_space
            ),
            pre_tokenizers.Punctuation(),
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.Digits(individual_digits=True),
        ]
    else:
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(
            vocab_size=vocab_size, special_tokens=special_tokens, **kwargs
        )
        pre_tokenizers_ = [pre_tokenizers.Whitespace()]
    normalizers_ = [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.StripAccents(),
    ]
    if lowercase:
        normalizers_ += [normalizers.Lowercase()]

    tokenizer.normalizer = normalizers.Sequence(normalizers_)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizers_)

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
