import pandas as pd
import re
import os
import pynori
from pathlib import Path
from soynlp.noun import LRNounExtractor_v2
from omegaconf import OmegaConf

from ..corpora.loader import eKorpkit, load_corpus_paths
from ..utils import print_status


def extract_nouns(**args):
    cfg = OmegaConf.create(args)
    # print(cfg)
    corpus_args = cfg.corpus

    corpus_paths = load_corpus_paths(**corpus_args)
    args = cfg.task.vocab.extract_nouns

    output_dir = args.output_dir
    extacted_nouns_file = args.extacted_nouns_file
    filterd_nouns_file = args.filterd_nouns_file
    min_noun_frequency = args.min_noun_frequency
    min_eojeol_frequency = args.min_eojeol_frequency
    max_noun_length = args.max_noun_length
    min_noun_length = args.min_noun_length
    min_noun_score = args.min_noun_score
    min_keep_score = args.min_keep_score
    stopwords_file = args.stopwords_file
    addwords_file = args.addwords_file
    prebuilt_dic_dir = args.prebuilt_dic_dir
    prebuilt_dic_files = args.prebuilt_dic_files
    prebuilt_dic_file_pattern = args.prebuilt_dic_file_pattern
    filter_only = args.filter_only
    save_as_pynori_userdict = args.save_as_pynori_userdict
    keep_noun_patten = args.keep_noun_patten

    prebuilt_dic_words = []
    for file in Path(prebuilt_dic_dir).rglob(prebuilt_dic_file_pattern):
        if prebuilt_dic_files:
            if file.name not in prebuilt_dic_files:
                continue
        df = pd.read_csv(file, header=None)
        prebuilt_dic_words += df[0].to_list()
    if prebuilt_dic_words:
        prebuilt_dic_words = set(prebuilt_dic_words)
        print(
            f"{len(prebuilt_dic_words)} words from the pre-built dictionary in {prebuilt_dic_dir} are loaded"
        )

    status = [
        [f" {i} ", name, " - ", " - ", " - ", Path(corpus_file).name]
        for i, (name, corpus_file) in enumerate(corpus_paths)
    ]
    if not filter_only:
        print_status(status)

    def iter_corpus_files():
        for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
            n_sampled = 0

            corpus = eKorpkit.load(
                corpus_name,
                corpus_dir=corpus_file,
                corpus_type=corpus_args.corpus_type,
                load_light=False,
                load_dataframe=True,
                id_keys=corpus_args.id_keys,
            )

            for rn, doc in corpus.docs.iterrows():
                for sent in str(doc[corpus.text_key]).split("\n"):
                    sent = sent.strip()
                    if len(sent) > 0:
                        n_sampled += 1
                        yield sent
            status[i_corpus][0] = " x "
            status[i_corpus][3] = n_sampled
            print_status(status)

    extacted_nouns_file = f"{output_dir}/{extacted_nouns_file}"
    os.makedirs(output_dir, exist_ok=True)
    if filter_only:
        # load from file
        if os.path.exists(extacted_nouns_file):
            df = pd.read_csv(extacted_nouns_file, index_col=0)
            nouns = df.to_dict("records")
        else:
            raise FileNotFoundError(f"{extacted_nouns_file} does not exist")
    else:
        noun_extractor = LRNounExtractor_v2(
            max_left_length=max_noun_length,
            max_right_length=max_noun_length,
            verbose=True,
            min_num_of_features=min(min_noun_frequency, min_eojeol_frequency),
            max_frequency_when_noun_is_eojeol=max(
                min_noun_frequency, min_eojeol_frequency
            ),
            ensure_normalized=False,
            extract_compound=False,
            extract_determiner=False,
        )
        nouns = noun_extractor.train_extract(
            iter_corpus_files(),
            min_noun_score=min_noun_score,
            min_noun_frequency=min_noun_frequency,  # 추출되는 명사의 최소 빈도수
            min_eojeol_frequency=min_eojeol_frequency,
            reset_lrgraph=False,  # predicator extraction 을 위해서
        )
        nouns = [
            {"word": word, "frequency": n_score.frequency, "score": n_score.score}
            for word, n_score in nouns.items()
        ]
        df = pd.DataFrame(nouns)
        df.to_csv(extacted_nouns_file, header=True)
        print(df.tail())
        print(f"Extracted nouns is saved at {extacted_nouns_file}")

    if stopwords_file:
        stopwords = set(
            [
                w.strip().lower()
                for w in open(stopwords_file).read().split("\n")
                if len(w.strip()) > 1
            ]
        )
        print("Stopwords:")
        print(stopwords)
        print("-" * 80)
        if stopwords:
            open(stopwords_file, "w").write("\n".join(sorted(stopwords)))
            # '\n'.join(sorted(stopwords, reverse=True, key=lambda x: len(x)))
    else:
        stopwords = []

    noun_pattern = re.compile(keep_noun_patten)
    saving_nouns = []
    for noun in nouns:
        word = noun["word"]
        if len(word) >= min_noun_length and noun["score"] >= min_keep_score:
            if (
                re.search(noun_pattern, word)
                and word[-1] != "들"
                and word not in stopwords
                and word not in prebuilt_dic_words
            ):
                saving_nouns.append(word)

    if addwords_file:
        addwords = set(
            [
                w.strip().lower()
                for w in open(addwords_file).read().split("\n")
                if len(w.strip()) > 1
            ]
        )
        print("Add words:")
        print(addwords)
        print("-" * 80)
        if addwords:
            open(addwords_file, "w").write("\n".join(sorted(addwords)))
        saving_nouns += addwords
    saving_nouns = sorted(set(saving_nouns))

    if save_as_pynori_userdict:
        dic_path = os.path.dirname(pynori.__file__) + "/resources/" + filterd_nouns_file
        with open(dic_path, "w") as f:
            f.write("\n".join(saving_nouns))
        print(f"User dictionary for pynori is saved at {dic_path}")

    filtered_nouns_path = f"{output_dir}/{filterd_nouns_file}"
    with open(filtered_nouns_path, "w") as f:
        f.write("\n".join(saving_nouns))
    print(f"Filtered nouns is saved at {filtered_nouns_path}")

    if not filter_only:
        print_status(status)
