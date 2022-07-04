import logging
import codecs
import re
import math
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Callable
from ekorpkit import eKonf


log = logging.getLogger(__name__)


Offset = namedtuple("Offset", ["this", "next", "prev_cum_len", "this_cum_len"])


class BaseSegmenter:
    __metaclass__ = ABCMeta
    """Abstract segmenter class from which all segmenters inherit.
    Subclasses must implement a ``segment()`` method.
    """

    @abstractmethod
    def segment(self, text):
        raise NotImplementedError("Must override segment")

    def __call__(self, text):
        """Calling a segmenter instance like a function just calls the segment method."""
        return self.segment(text)


class Segmenter(BaseSegmenter):
    def __init__(
        self,
        separators={},
        merge={},
        split={},
        filter_language={},
        filter_programming_language=False,
        filter_sentence_length={},
        chunk={},
        return_as_list=True,
        verbose=False,
        print_args=False,
        **kwargs,
    ):
        _separators = separators or {}
        _merge = merge or {}
        _split = split or {}
        _filter_language = filter_language or {}
        _filter_sentence_length = filter_sentence_length or {}
        _chunk = chunk or {}

        in_segment_separator = _separators.get("in_segment", "\n\n")
        in_sentence_separator = _separators.get("in_sentence", "\n")
        out_segment_separator = _separators.get("out_segment", "\n\n")
        out_sentence_separator = _separators.get("out_sentence", "\n")

        keep_segment = _split.get("keep_segment", True)
        max_split_length = _split.get("max_split_length", 30_000)
        max_split_iterations = _split.get("max_split_iterations", 100)

        merge_lines = _merge.get("merge_lines", False)
        merge_level = _merge.get("merge_level", "segmenmt")
        empty_lines_threshold = _merge.get("empty_lines_threshold", 0.6)
        broken_lines_threshold = _merge.get("broken_lines_threshold", 0.4)

        filter_language = _filter_language.get("filter", False)
        detection_level = _filter_language.get("detection_level", "segment")
        languages_to_keep = _filter_language.get("languages_to_keep", ["en", "ko"])
        min_language_probability = _filter_language.get("min_language_probability", 0.8)

        filter_sentence_length = _filter_sentence_length.get("filter", False)
        min_num_words = _filter_sentence_length.get("min_num_words", 3)
        min_length = _filter_sentence_length.get("min_length", 10)

        chunk_size = _chunk.get("chunk_size", 300)
        chunk_overlap = _chunk.get("chunk_overlap", False)
        self.len_func_name = _chunk.get("len_func", "len_bytes")
        len_func = _chunk.get(eKonf.Keys.FUNC, {}).get(self.len_func_name, None)
        self.len_func = eKonf.partial(len_func)

        self._in_segment_separator = codecs.decode(
            in_segment_separator, "unicode_escape"
        )
        self._in_sentence_separator = codecs.decode(
            in_sentence_separator, "unicode_escape"
        )
        self._out_segment_separator = codecs.decode(
            out_segment_separator, "unicode_escape"
        )
        self._out_sentence_separator = codecs.decode(
            out_sentence_separator, "unicode_escape"
        )
        self._return_as_list = return_as_list
        self._max_split_length = max_split_length
        self._max_split_iterations = max_split_iterations
        self._keep_segment = keep_segment
        self._merge_lines = (
            merge_lines
            if isinstance(merge_lines, bool)
            else str(merge_lines).lower() in ["true", "t", "1"]
        )
        self._merge_level = merge_level
        self._empty_lines_threshold = empty_lines_threshold
        self._broken_lines_threshold = broken_lines_threshold
        self._filter_language = filter_language
        self._detection_level = detection_level
        self._languages_to_keep = list(languages_to_keep)
        self._min_language_probability = min_language_probability
        self._filter_programming_language = filter_programming_language
        self._filter_sentence_length = filter_sentence_length
        self._min_num_words = min_num_words
        self._min_length = min_length
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.verbose = verbose
        self._kwargs = kwargs
        if print_args:
            log.info(f"in_segment_separator = {self._in_segment_separator}")
            log.info(f"in_sentence_separator = {self._in_sentence_separator}")
            log.info(f"out_segment_separator = {self._out_segment_separator}")
            log.info(f"out_sentence_separator = {self._out_sentence_separator}")
            log.info(f"return_as_list = {self._return_as_list}")
            log.info(f"max_split_length = {self._max_split_length}")
            log.info(f"max_split_iterations = {self._max_split_iterations}")
            log.info(f"keep_segment = {self._keep_segment}")
            log.info(f"merge_lines = {self._merge_lines}")
            log.info(f"merge_level = {self._merge_level}")
            log.info(f"empty_lines_threshold = {self._empty_lines_threshold}")
            log.info(f"broken_lines_threshold = {self._broken_lines_threshold}")
            log.info(f"filter_language = {self._filter_language}")
            log.info(f"detection_level = {self._detection_level}")
            log.info(f"languages_to_keep = {self._languages_to_keep}")
            log.info(f"min_language_probability = {self._min_language_probability}")
            log.info(
                f"filter_programming_language = {self._filter_programming_language}"
            )
            log.info(f"filter_sentence_length = {self._filter_sentence_length}")
            log.info(f"min_num_words = {self._min_num_words}")
            log.info(f"min_length = {self._min_length}")
            log.info(f"chunk_size = {self._chunk_size}")
            log.info(f"chunk_overlap = {self._chunk_overlap}")
            log.info(f"verbose = {self.verbose}")
            log.info(f"kwargs = {self._kwargs}")

        self._guess = None

    def _check_if_merge_lines(self, text):
        num_empty_lines = 0
        num_lines_endswith_period = 0
        num_lines = len(text.split("\n"))
        for line in text.split("\n"):
            if len(line.strip()) == 0:
                num_empty_lines += 1
            if line.strip().endswith("."):
                num_lines_endswith_period += 1
        empty_lines_ratio = num_empty_lines / num_lines if num_lines > 0 else 0
        broken_lines_ratio = (
            1.0 - num_lines_endswith_period / (num_lines - num_empty_lines)
            if (num_lines - num_empty_lines) > 0
            else 0
        )
        if self.verbose:
            log.info(f"num_empty_lines={num_empty_lines}")
            log.info(f"num_lines_endswith_period={num_lines_endswith_period}")
            log.info(f"num_lines={num_lines}")
            log.info(f"empty_lines_ratio={empty_lines_ratio}")
            log.info(f"broken_lines_ratio={broken_lines_ratio}")
        if empty_lines_ratio > self._empty_lines_threshold:
            return True
        if broken_lines_ratio > self._broken_lines_threshold:
            return True

    def segment_article(self, article):

        if article is None:
            return None

        article = re.sub(
            f"{self._in_segment_separator}+", self._in_segment_separator, article
        )
        if self._merge_lines == "auto":
            merge_lines = self._check_if_merge_lines(article)
        else:
            merge_lines = self._merge_lines
        if merge_lines and self._merge_level == "article":
            if self.verbose > 10:
                log.info("=> merging lines of the article")
            article = re.sub(r"\s+", " ", article)

        if self._keep_segment:
            in_segments = article.split(self._in_segment_separator)
        else:
            in_segments = [article]

        segments = []
        for i, segment in enumerate(in_segments):
            segment = segment.strip()
            if len(segment) > 0:
                if self._filter_programming_language:
                    if self._skip_programming_language(segment):
                        continue
                if merge_lines and self._merge_level == "segment":
                    if self.verbose > 10:
                        log.info("=> merging lines of the segment")
                    segment = re.sub(r"\s+", " ", segment)

                if self._filter_language and self._detection_level == "segment":
                    if self._skip_language(segment):
                        continue
                sentences = []
                for sent in segment.split(self._in_sentence_separator):
                    sent = sent.strip()
                    if len(sent) > 0:

                        if (
                            self._filter_language
                            and self._detection_level == "sentence"
                        ):
                            if self._skip_language(sent):
                                continue

                        try:
                            sentences.extend(self._split_sentences(sent))
                        except Exception as e:
                            log.info(f"===> skipping segment {i}: {e}")
                            continue

                sentences = self.filter_sentence_length(sentences)
                if self._return_as_list:
                    segments.append(sentences)
                else:
                    segments.append(self._out_sentence_separator.join(sentences))

        return (
            segments
            if self._return_as_list
            else self._out_segment_separator.join(segments)
        )

    def filter_sentence_length(self, sentences):
        if self._filter_sentence_length:
            sentences = [
                sent
                for sent in sentences
                if len(sent.split()) >= self._min_num_words
                and len(sent) >= self._min_length
            ]
        return sentences

    def _skip_language(self, text):
        from ftlangdetect import detect

        lang = detect(text.replace("\n", " "), low_memory=False)
        if (
            lang is not None
            and lang["lang"] not in self._languages_to_keep
            and lang["score"] < self._min_language_probability
        ):
            if self.verbose > 5:
                log.info(f"===> lang={lang} not in languages_to_keep")
            return True
        return False

    # TODO: implement this
    # refer to: https://huggingface.co/huggingface/CodeBERTa-language-id
    def _skip_programming_language(self, text):
        return False
        # if self._guess is None:
        #     self._guess = Guess()

        # # Guess the language from code
        # lang = self._guess.language_name(text)
        # if lang is not None:
        #     if self.verbose > 5:
        #         log.info(f"===> Programming language: {lang}")
        #     if self.verbose > 10:
        #         log.info(text)
        #     return True
        # return False

    def _split_sentences(self, text):
        text = text.strip()
        if len(text) <= self._max_split_length:
            return self.segment(text)
        else:
            # if self.verbose:
            #     log.info(f"==> too long text: {len(text)}")
            text_chunk = text[: self._max_split_length]
            next_chunk = text[self._max_split_length :]
            sentences = []
            split_iter = 0
            while len(text_chunk) > 0 and split_iter < self._max_split_iterations:
                sents = self.segment(text_chunk)
                if len(sents) > 1 and len(next_chunk) > 0:
                    sentences.extend(sents[:-1])
                    last_sent = sents[-1]
                else:
                    sentences.extend(sents)
                    last_sent = ""
                end_pos = self._max_split_length - len(last_sent)
                if end_pos >= len(next_chunk):
                    text_chunk = last_sent + next_chunk
                    next_chunk = ""
                else:
                    text_chunk = last_sent + next_chunk[:end_pos]
                    next_chunk = next_chunk[end_pos:]
                split_iter += 1
            if split_iter >= self._max_split_iterations:
                if self.verbose:
                    log.critical(f"==> too many split iterations: {split_iter}")
                raise Exception(f"Maximum number of iterations reached: {split_iter}")
            return sentences

    def segment_articles(self, articles):
        return [self.segment_article(article) for article in articles]

    def chunk_article(self, article):

        if article is None:
            return None

        article = re.sub(
            f"{self._in_segment_separator}+", self._in_segment_separator, article
        )
        if self._keep_segment:
            in_segments = article.split(self._in_segment_separator)
        else:
            in_segments = [article]

        segments = []
        for i, segment in enumerate(in_segments):
            segment = segment.strip()
            segment = re.sub(
                f"{self._in_sentence_separator}+", self._in_sentence_separator, segment
            )
            if len(segment) > 0:
                sentences = segment.split(self._in_sentence_separator)

                chunks = self.chunk(
                    sentences,
                    max_length=self._chunk_size,
                    overlap=self._chunk_overlap,
                )
                if self._return_as_list != "list":
                    chunks = [
                        self._out_sentence_separator.join(chunk) for chunk in chunks
                    ]
                segments += chunks

        return (
            segments
            if self._return_as_list
            else self._out_segment_separator.join(segments)
        )

    def chunk(
        self,
        sentences: list,
        max_length: int,
        overlap: bool = False,
        len_func: Callable = None,
    ):
        """
        Split chunks from input texts by max_length.

        Args:
            text (Union[str, List[str], tuple]): input texts
            max_length (int): max length of ecah chunk
            overlap (bool): whether allow duplicated sentence

        Returns:
            chunks of segmented sentences
        """

        assert isinstance(sentences, list), "param `sentences` must be str."
        assert isinstance(max_length, int), "param `max_length` must be `int` type."
        assert isinstance(overlap, bool), "param `overlap` must be `bool` type."
        if len_func is None:
            len_func = self.len_func
        if not callable(len_func):
            len_func = len

        span, chunks = [], []

        for offset in _get_sentence_offsets(sentences, len_func):
            if len(span) > 0:
                if offset.this_cum_len - span[0].prev_cum_len > max_length:
                    chunks.append(_get_chunk_with_offset(sentences, span))
                    if overlap:
                        span = span[math.trunc(len(span) / 2) :]
                    else:
                        span = []

            span.append(offset)
        chunks.append(_get_chunk_with_offset(sentences, span))
        return chunks


def _get_sentence_offsets(sentences, len_func: Callable = None):
    sentence_offsets = []
    offset = None
    for i, sentence in enumerate(sentences):
        prev_cum_len = offset.this_cum_len if offset is not None else 0
        offset = Offset(i, i + 1, prev_cum_len, prev_cum_len + len_func(sentence))
        sentence_offsets.append(offset)
    return sentence_offsets


def _remove_useless_space(text):
    return re.sub(r"\s+", " ", text)


def _get_chunk_with_offset(sentences, span):
    start = span[0].this
    end = span[-1].next
    return sentences[start:end]


class NLTKSegmenter(Segmenter):
    def __init__(self, **kwargs):
        import nltk

        nltk.download("punkt", quiet=True)
        super().__init__(**kwargs)

    def segment(self, text):
        import nltk

        return nltk.tokenize.sent_tokenize(text)


class SimpleSegmenter(Segmenter):
    def __init__(self, simple={}, **kwargs):

        simple = simple or {}
        if not simple:
            simple["separator"] = "\n"
            simple["remove_newlines"] = True

        self.separator = codecs.decode(simple["separator"], "unicode_escape")
        self.remove_newlines = simple["remove_newlines"]
        log.info(f"segment_separator: -->{self.separator}<--")
        if self.remove_newlines:
            log.info("Remove newlines inside segments")
        super().__init__(**kwargs)

    def segment(self, text):
        segments = text.split(self.separator)
        if self.remove_newlines:
            segments = [segment.replace("\n", " ") for segment in segments]
        return segments


class PySBDSegmenter(Segmenter):
    def __init__(self, pysbd={}, **kwargs):
        import pysbd as pySBD

        pysbd = pysbd or {}
        if not pysbd:
            pysbd["language"] = "en"
            pysbd["clean"] = False

        if pysbd.get("doc_type", None) == "pdf":
            pysbd["clean"] = True
        self.seg = pySBD.Segmenter(**pysbd)
        super().__init__(**kwargs)

    def segment(self, text):
        return self.seg.segment(text)


class KSSSegmenter(Segmenter):
    def __init__(self, kss={}, **kwargs):
        from ekorpkit.preprocessors.kss import KSS

        kss = kss or {}
        if not kss:
            kss["use_heuristic"] = False
            kss["backend"] = "fugashi"

        self.seg = KSS(**kss)
        super().__init__(**kwargs)

    def segment(self, text):
        return self.seg.segment(text)
