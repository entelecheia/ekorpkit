import os
import logging
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import (
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BigBirdConfig,
    BigBirdForMaskedLM,
    BigBirdTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    ElectraTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LongformerConfig,
    LongformerForMaskedLM,
    LongformerTokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    XLMRobertaTokenizer,
)

log = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModelWithLMHead, AutoTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "electra": (ElectraConfig, ElectraForMaskedLM, ElectraTokenizer),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


def train_tokenizer(
    train_files,
    args,
    tokenizer_name=None,
    output_dir=None,
):
    """
    Train a new tokenizer on `train_files`.

    Args:

    - train_files: List of files to be used when training the tokenizer.

    - tokenizer_name: Name of a pretrained tokenizer or a path to a directory containing a tokenizer.

    - output_dir (optional): The directory where model files will be saved. If not given, args.output_dir
    will be used.

    Returns: None
    """

    if not args.vocab_size:
        raise AttributeError(
            "Cannot train a new tokenizer as vocab_size is not specified in args dict. "
            "Either provide a tokenizer or specify vocab_size."
        )

    if not isinstance(train_files, list):
        train_files = [train_files]

    if not output_dir:
        output_dir = args.output_dir

    if args.model_type in ["bert", "electra"]:
        tokenizer = BertWordPieceTokenizer(
            clean_text=args.clean_text,
            handle_chinese_chars=args.handle_chinese_chars,
            strip_accents=args.strip_accents,  # Must be False if cased model
            lowercase=args.do_lower_case,
        )
        # args.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        # args.wordpieces_prefix = "##"
        unused_tokens = [f"[unused{n}]" for n in range(args.unused_token_num)]
        special_tokens = list(args.special_tokens) + unused_tokens

        tokenizer.train(
            files=train_files,
            vocab_size=args.vocab_size,
            limit_alphabet=args.limit_alphabet,
            min_frequency=args.min_frequency,
            special_tokens=special_tokens,
            wordpieces_prefix=args.wordpieces_prefix,
        )
    elif args.model_type in ["bigbird", "xlmroberta"]:
        # The google BigBird way
        # Tokenizers sentencepiece does not build a BigBird compatible vocabulary model
        import sentencepiece as spm
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        files = ",".join(train_files)

        if args.model_type in ["xlmroberta"]:
            # </s>,<s>,<unk>,<pad> are built in -- leave as default
            # XLMRoberta uses sentencepiece.bpe as a vocab model prefix
            prefix = "sentencepiece.bpe"
            spm.SentencePieceTrainer.Train(
                f"--input={files} --user_defined_symbols='<mask>,<s>NOTUSED,</s>NOTUSED' --model_prefix={prefix} --vocab_size={args.vocab_size - 2}"
            )
        else:
            # </s>,<s>,<unk>,<pad> are built in -- leave as default
            # BigBird uses spiece as a vocab model prefix
            prefix = "spiece"
            spm.SentencePieceTrainer.Train(
                f"--input={files} --user_defined_symbols='[SEP],[CLS],[MASK]' --model_prefix=spiece --vocab_size={args.vocab_size - 3}"
            )

        # SentencePiece There is no option for output path https://github.com/google/sentencepiece/blob/master/doc/options.md
        if os.path.exists(output_dir + "/" + f"{prefix}.model"):
            os.remove(output_dir + "/" + f"{prefix}.model")
        shutil.move(src=f"{prefix}.model", dst=output_dir)

        if os.path.exists(output_dir + "/" + f"{prefix}.vocab"):
            os.remove(output_dir + "/" + f"{prefix}.vocab")
        shutil.move(src=f"{prefix}.vocab", dst=output_dir)
    else:
        tokenizer = ByteLevelBPETokenizer(lowercase=args.do_lower_case)

        tokenizer.train(
            files=train_files,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=list(args.special_tokens),
        )

    if args.model_type not in ["bigbird", "xlmroberta"]:
        os.makedirs(output_dir, exist_ok=True)

        tokenizer.save_model(output_dir)
        log.info(
            " Training of {} tokenizer complete. Saved to {}.".format(
                tokenizer_name, output_dir
            )
        )

    # _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    # tokenizer = tokenizer_class.from_pretrained(output_dir)


def train_spm(
    train_files,
    output_dir=None,
    train_args=None,
    **kwargs,
):
    import sentencepiece as spm
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    files = ",".join(train_files)

    if args.model_type in ["xlmroberta"]:
        # </s>,<s>,<unk>,<pad> are built in -- leave as default
        # XLMRoberta uses sentencepiece.bpe as a vocab model prefix
        prefix = "sentencepiece.bpe"
        spm.SentencePieceTrainer.Train(
            f"--input={files} --user_defined_symbols='<mask>,<s>NOTUSED,</s>NOTUSED' --model_prefix={prefix} --vocab_size={args.vocab_size - 2}"
        )
    else:
        # </s>,<s>,<unk>,<pad> are built in -- leave as default
        # BigBird uses spiece as a vocab model prefix
        prefix = "spiece"
        spm.SentencePieceTrainer.Train(
            f"--input={files} --user_defined_symbols='[SEP],[CLS],[MASK]' --model_prefix=spiece --vocab_size={args.vocab_size - 3}"
        )

    # SentencePiece There is no option for output path https://github.com/google/sentencepiece/blob/master/doc/options.md
    if os.path.exists(output_dir + "/" + f"{prefix}.model"):
        os.remove(output_dir + "/" + f"{prefix}.model")
    shutil.move(src=f"{prefix}.model", dst=output_dir)

    if os.path.exists(output_dir + "/" + f"{prefix}.vocab"):
        os.remove(output_dir + "/" + f"{prefix}.vocab")
    shutil.move(src=f"{prefix}.vocab", dst=output_dir)
