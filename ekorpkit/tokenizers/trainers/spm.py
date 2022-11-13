import unicodedata
import os
import logging
import nltk
import glob
import sentencepiece as spm
from tqdm.auto import tqdm
from random import sample
from ..config import ModelType, TrainerType
from ekorpkit.utils.func import change_directory


log = logging.getLogger(__name__)


def train_spm(
    model_prefix,
    input,
    output_dir,
    vocab_size=30000,
    model_type: ModelType = ModelType.UNIGRAM,
    character_coverage=1.0,
    num_threads=1,
    train_extremely_large_corpus=False,
    **kwargs,
):
    model_prefix = f"{model_prefix}_{model_type}_{TrainerType.SPM}_vocab_{vocab_size}"
    model_name = f"{model_prefix}.model"
    log.info(f"Training SentencePiece model {model_name}")
    # change context work dir to output_dir
    # so that the model is saved to the correct location
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with change_directory(output_dir):
        spm.SentencePieceTrainer.train(
            input=input,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            num_threads=num_threads,
            train_extremely_large_corpus=train_extremely_large_corpus,
            model_type=model_type,
            **kwargs,
        )
    output_filepath = os.path.join(output_dir, model_name)
    log.info(f"saved model to {output_filepath}")
    return output_filepath


def sample_and_combine(
    input_dir, output_dir, sample_size, filename="sampled_sentences.txt"
):
    """
    Use the set of files containing a sentence per line,
    sample num_files out of those and save as one text file
    """

    sentence_files = glob.glob(input_dir + "/*.txt")

    # sample num_files
    if sample_size <= len(sentence_files):
        sentence_files = sample(sentence_files, sample_size)
    else:
        log.info(
            f"Sample size {sample_size} is larger than number of files {len(sentence_files)}"
        )

    filenames = [os.path.basename(f) for f in sentence_files]
    log.info(f"sampled files: {filenames}")

    # read all the lines from sampled files and save to a list
    all_lines = []
    for fp in sentence_files:
        with open(fp) as f:
            lines = f.read().splitlines()

        all_lines.extend(lines)

    log.info(f"number of lines sampled: {len(all_lines):,}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filepath = os.path.join(output_dir, filename)
    with open(output_filepath, "w") as f:

        for sentence in tqdm(all_lines):

            # remove newlines
            sentence = sentence.strip()

            # do not save empty items such as
            if sentence != []:

                f.writelines(sentence + "\n")

    log.info(f"saved sampled sentences to {output_filepath}")
    return output_filepath


def batch_chunks(dataset, batch_size, text_column="text"):
    """Yield successive batch-sized chunks from dataset."""
    for i in tqdm(range(0, len(dataset), batch_size)):
        end_i = min(len(dataset), i + batch_size)
        yield dataset[i:end_i][text_column]


def export_sentence_chunk_files(
    dataset,
    output_dir,
    chunk_size=10_000,
    text_column="text",
    normalize=True,
    normmalize_form="NFKC",
    sent_tokenize=None,
    filename_fmt="sent_chunk_{:04d}.txt",
):
    """
    Make a sentence per line files, chuncsize sentences per file
    """

    # make sure data dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log.info(f"Writing sentence chunks to {output_dir}")

    if sent_tokenize is None:
        # use simple regex for sentence tokenizing
        sent_tokenize = nltk.RegexpTokenizer("[^　！？。]*[！？。.\n]").tokenize

    # loop over the chunks
    for chunk_id, data_chunk in enumerate(
        batch_chunks(dataset, chunk_size, text_column)
    ):
        # new file for each chunk
        filename = filename_fmt.format(chunk_id)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:

            for line in tqdm(data_chunk, desc=filename):

                line = line.strip()

                # unicode normalize text
                if normalize:
                    line = unicodedata.normalize(normmalize_form, line)

                # tokenize into sentences
                sentences = sent_tokenize(line)

                # do not save empty items such as
                if sentences != []:

                    f.writelines(s + "\n" for s in sentences)
