import os
from re import I
import pandas as pd
import multiprocessing
import logging

# from pandarallel import pandarallel
# nori multiprocessor

from ..preprocessors.normalizer import strict_normalize, only_text


logger = logging.getLogger(__name__)


class MPTokenizer:
    """Tokenizer 를 multiprocessing 으로 수행하기 위한 클래스."""

    def __init__(self, backend="mecab", num_workers=4, **kwargs):
        self.kwargs = kwargs
        self.num_workers = num_workers
        self.backend = backend.lower()

        assert self.backend in [
            "pynori",
            "mecab",
            "bwp",
        ], "Wrong backend! Currently, we support [`pynori`, `mecab`, `bwp`] backend."

    def tokenize_texts(self, input_path=None, output_path=None, **args):
        if input_path is None or not os.path.exists(input_path):
            logger.error(
                f'\n\t[Error] Please check your read file path - "{input_path}"\n'
            )
            exit()
        num_workers = self.num_workers
        count = _get_file_counts(input_path)
        offsets = _get_offset_ranges(count, num_workers)
        logger.info(f"file offsets based on number of workers: {offsets}")

        pool = multiprocessing.Pool(processes=num_workers)
        logger.info(f"[Start] Multiprocessing. number of workers: {num_workers}")
        for worker_id in range(num_workers):
            pool.apply_async(
                worker_function,
                (
                    input_path,
                    output_path,
                    worker_id,
                    self.backend,
                    self.kwargs,
                    offsets[worker_id],
                    offsets[worker_id + 1],
                ),
            )
        pool.close()
        logger.info("[Complete] join all workers")
        pool.join()

        merge_worker_files(num_workers, output_path)
        logger.info('[Complete] merge all worker files to path: "{write_path}"')
        for worker_id in range(num_workers):
            if os.path.exists(f"{output_path}_{worker_id}"):
                os.remove(f"{output_path}_{worker_id}")
        logger.info(f"[End] Multiprocessing.")
        return count

    def tokenize_dataframe(self, input_dataframe=None, text_key="text", **args):
        num_workers = self.num_workers
        count = len(input_dataframe.index)
        offsets = _get_offset_ranges(count, num_workers)
        logger.info(f"file offsets based on number of workers: {offsets}")

        split_dfs = [
            input_dataframe.iloc[offsets[worker_id] : offsets[worker_id + 1], :]
            for worker_id in range(num_workers)
        ]

        pool = multiprocessing.Pool(processes=num_workers)
        logger.info(f"[Start] Multiprocessing. number of workers: {num_workers}")
        results = []
        for worker_id in range(num_workers):
            results.append(
                pool.apply_async(
                    worker_function_for_df,
                    (
                        split_dfs[worker_id],
                        text_key,
                        worker_id,
                        self.backend,
                        self.kwargs,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                )
            )
        pool.close()
        logger.info("[Complete] join all workers")
        pool.join()

        new_dfs = [result.get() for result in results]
        logger.info("[Complete] join all workers")
        df = pd.concat(new_dfs, ignore_index=True, axis=0)
        logger.info("[Complete] merge all worker files to the return dataframe")
        logger.info(f"[End] Multiprocessing.")
        # checking if the dfs were merged correctly
        # pdt.assert_series_equal(in_df['id'], df['id'])
        return df


class Tokenizer:
    def __init__(
        self,
        backend="pynori",
        wordpieces_prefix=None,
        tokenize_each_word=False,
        normalize=False,
        **kwargs,
    ):

        self.backend = backend
        self.wordpieces_prefix = wordpieces_prefix if wordpieces_prefix else ""
        self.tokenize_each_word = tokenize_each_word
        self.normalize = normalize

        if backend == "pynori":
            logging.warning("Initializing Pynori...")
            try:
                from pynori.korean_analyzer import KoreanAnalyzer

                self._tokenizer = KoreanAnalyzer(**kwargs)
            except ImportError:
                raise ImportError(
                    "\n"
                    "You must install `pynori` if you want to use `pynori` backend.\n"
                    "Please install using `pip install pynori`.\n"
                )
        elif backend == "mecab":
            logging.warning("Initializing mecab...)")
            try:
                # import mecab
                # logging.warning(f"MeCab Version: {mecab.__version__})")
                from .mecab import MeCab

                self._tokenizer = MeCab(**kwargs)
            except ImportError:
                raise ImportError(
                    "\n"
                    "You must install `fugashi` and `mecab_ko_dic` if you want to use `mecab` backend.\n"
                    "Please install using `pip install python-mecab-ko`.\n"
                )
        elif backend == "bwp":
            logging.warning("Initializing BertWordPieceTokenizer...")
            try:
                from transformers import BertTokenizerFast

                self._tokenizer = BertTokenizerFast.from_pretrained(**kwargs)

            except ImportError:
                raise ImportError(
                    "\n"
                    "You must install `BertWordPieceTokenizer` if you want to use `bwp` backend.\n"
                    "Please install using `pip install transformers`.\n"
                )

        else:
            self._tokenizer = None

    def tokenize(self, text):

        if self.normalize:
            text = only_text(strict_normalize(text))

        if self.backend == "pynori":
            tokens = self._tokenizer.do_analysis(text)
            term_pos = [
                f"{term}/{pos}"
                for term, pos in zip(tokens["termAtt"], tokens["posTagAtt"])
            ]

        elif self.backend == "bwp":
            term_pos = self._tokenizer.tokenize(text)

        elif self.backend == "mecab":

            def _tokenize(word):
                tokens = self._tokenizer.pos(word)
                term_pos = []
                for i, (term, pos) in enumerate(tokens):
                    if i == 0:
                        term_pos.append(f"{term}/{pos}")
                    else:
                        term_pos.append(f"{self.wordpieces_prefix}{term}/{pos}")
                return term_pos

            if self.tokenize_each_word:
                term_pos = []
                for word in text.split():
                    term_pos += _tokenize(word)
            else:
                text = " ".join(text.split())
                tokens = self._tokenizer.pos(text)
                term_pos = []
                term_start = 0
                add_prefix = False
                for term, pos in tokens:
                    if text[term_start] == " ":
                        term_pos.append(f"/SP")
                        term_start += 1
                        add_prefix = False
                    if add_prefix:
                        term_pos.append(f"{self.wordpieces_prefix}{term}/{pos}")
                    else:
                        term_pos.append(f"{term}/{pos}")
                    add_prefix = True
                    term_start += len(term)
        else:
            raise AttributeError(
                "Wrong backend ! currently, it only supports `pynori`, `mecab` backend."
            )
        return term_pos


def worker_function(
    input_path, output_path, worker_id, backend, kwargs, start_offset=0, end_offset=-1
):
    """단일 프로세스에 사용되는 worker 함수"""
    logger.info(
        f"[Start] worker_id: {worker_id} [start_offset: {start_offset} & end_offset: {end_offset}]"
    )
    _tokenizer = Tokenizer(backend=backend, **kwargs)
    rf = open(input_path, "r", encoding="utf-8")
    with open(f"{output_path}_{worker_id}", "w", encoding="utf-8") as wf:
        cnt = 0
        for line in rf:
            cnt += 1
            if cnt <= start_offset or cnt > end_offset:
                continue

            line = line.strip()
            if line is None or len(line) == 0:
                tokenized_line = "\n"
            else:
                tokens = _tokenizer.tokenize(line)
                tokenized_line = " ".join(tokens) + "\n"
            wf.write(tokenized_line)
    rf.close()


def _get_file_counts(path):
    with open(path, "r", encoding="utf-8") as rf:
        for i, _ in enumerate(rf):
            pass
    return i + 1


def _get_offset_ranges(count, num_workers):
    assert count > num_workers
    step_sz = int(count / num_workers)
    offset_ranges = [0]
    pv_cnt = 1
    for i in range(num_workers):
        if i == num_workers - 1:
            pv_cnt = count
        else:
            pv_cnt = pv_cnt + step_sz
        offset_ranges.append(pv_cnt)
    return offset_ranges


def merge_worker_files(num_workers, write_path):
    total_cnt = 0
    worker_files = [f"{write_path}_{x}" for x in range(num_workers)]
    wf = open(write_path, "w", encoding="utf-8")
    for file in worker_files:
        with open(file, "r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                wf.write(line + "\n")
                total_cnt += 1
    logger.info(f"[Complete] merge worker files. total cnt: {total_cnt}")
    wf.close()


def worker_function_for_df(
    df, text_key, worker_id, backend, kwargs, start_offset=0, end_offset=-1
):
    """단일 프로세스에 사용되는 worker 함수"""
    logger.info(
        f"[Start] worker_id: {worker_id} [start_offset: {start_offset} & end_offset: {end_offset}]"
    )
    _tokenizer = Tokenizer(backend=backend, **kwargs)

    def tokenized_row(row):
        text = row[text_key]
        if not isinstance(text, str):
            return None

        sents = []
        for sent in text.split("\n"):
            if len(sent) > 0:
                tokens = _tokenizer.tokenize(sent)
                tokenized_sent = " ".join(tokens)
                sents.append(tokenized_sent)
            else:
                sents.append("")
        return "\n".join(sents)

    df[text_key] = df.apply(tokenized_row, axis=1)
    # print(df.tail())
    return df


def extract_tokens(
    text,
    nouns_only=False,
    noun_pos_filter=["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"],
    token_xpos_filter=["SP"],
    no_space_for_non_nouns=False,
    **kwargs,
):

    _tokens = [token.split("/") for token in text.split() if len(token.split("/")) == 2]

    if nouns_only:
        tokens = [token[0].strip() for token in _tokens if token[1] in noun_pos_filter]
    else:
        exist_sp_tag = False
        for i, token in enumerate(_tokens):
            if token[1] == "SP":
                exist_sp_tag = True
                break

        tokens = []
        if exist_sp_tag and no_space_for_non_nouns:
            prev_nonnoun_check = False
            cont_morphs = []
            i = 0
            while i < len(_tokens):
                token = _tokens[i]
                if not prev_nonnoun_check and token[1] in noun_pos_filter:
                    tokens.append(token[0])
                elif (
                    not prev_nonnoun_check
                    and token[1] not in noun_pos_filter
                    and token[1][0] != "S"
                ):
                    prev_nonnoun_check = True
                    cont_morphs.append(token[0])
                elif (
                    prev_nonnoun_check
                    and token[1] not in noun_pos_filter
                    and token[1][0] != "S"
                ):
                    cont_morphs.append(token[0])
                else:
                    if len(cont_morphs) > 0:
                        tokens.append("".join(cont_morphs))
                        cont_morphs = []
                        prev_nonnoun_check = False
                    if token[1] != "SP":
                        tokens.append(token[0])
                i += 1
            if len(cont_morphs) > 0:
                tokens.append("".join(cont_morphs))
        else:
            tokens = [
                token[0].strip()
                for token in _tokens
                if token[1] not in token_xpos_filter
            ]
    return tokens


def extract_tokens_dataframe(df, **args):
    x_args = args["extract_func"]
    text_key = x_args["text_key"]
    num_workers = args["num_workers"]

    # if num_workers > 1:
    #     pandarallel.initialize(nb_workers=num_workers)
    # elif num_workers < 1:
    #     pandarallel.initialize()

    def extact_tokens_row(row):
        text = row[text_key]
        if not isinstance(text, str):
            return None

        sents = []
        for sent in text.split("\n"):
            if len(sent) > 0:
                tokens = extract_tokens(sent, **x_args)
                token_sent = " ".join(tokens)
                sents.append(token_sent)
            else:
                sents.append("")
        return "\n".join(sents)

    # if num_workers > 1:
    #     df[text_key] = df.parallel_apply(extact_tokens_row, axis=1)
    # else:
    df[text_key] = df.apply(extact_tokens_row, axis=1)

    df = df.dropna(subset=[text_key])
    return df


def tokenize_dataframe(df, **args):
    from hydra.utils import instantiate

    mpt_args = args["mp_tokenizer"]
    tokenize_func = args["tokenize_func"]

    mp_tokenizer = instantiate(mpt_args)
    # df = mp_tokenizer.tokenize_dataframe(input_dataframe=df, text_key=text_key)
    df = getattr(mp_tokenizer, tokenize_func["_target_"])(df, **tokenize_func)
    return df
