import codecs
import re
import math
from abc import ABCMeta, abstractmethod
from collections import namedtuple


Offset = namedtuple("Offset", ["this", "next", "prev_cum_len", "this_cum_len"])


class Segmenter:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        in_segment_separator="\n\n",
        in_sentence_separator="\n",
        out_segment_separator="\n\n",
        out_sentence_separator="\n",
        return_type="list",
        max_split_length=30_000,
        max_split_iterations=100,
        keep_segment=True,
        merge_lines=False,
        merge_level="segment",
        empty_lines_threshold=0.6,
        broken_lines_threshold=0.4,
        filter_language=False,
        detection_level="segment",
        languages_to_keep=["en", "ko"],
        min_language_probability=0.8,
        filter_programming_language=False,
        filter_sentence_length=False,
        min_num_words=3,
        min_length=10,
        chunk_size=300,
        chunk_overlap=False,
        verbose=False,
        print_args=False,
        **kwargs,
    ):
        self.in_segment_separator = codecs.decode(
            in_segment_separator, "unicode_escape"
        )
        self.in_sentence_separator = codecs.decode(
            in_sentence_separator, "unicode_escape"
        )
        self.out_segment_separator = codecs.decode(
            out_segment_separator, "unicode_escape"
        )
        self.out_sentence_separator = codecs.decode(
            out_sentence_separator, "unicode_escape"
        )
        self.return_type = return_type
        self.max_split_length = max_split_length
        self.max_split_iterations = max_split_iterations
        self.keep_segment = keep_segment
        self.merge_lines = (
            merge_lines
            if isinstance(merge_lines, bool)
            else str(merge_lines).lower() in ["true", "t", "1"]
        )
        self.merge_level = merge_level
        self.empty_lines_threshold = empty_lines_threshold
        self.broken_lines_threshold = broken_lines_threshold
        self.filter_language = filter_language
        self.detection_level = detection_level
        self.languages_to_keep = list(languages_to_keep)
        self.min_language_probability = min_language_probability
        self.filter_programming_language = filter_programming_language
        self.filter_sentence_length = filter_sentence_length
        self.min_num_words = min_num_words
        self.min_length = min_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self.kwargs = kwargs
        if print_args:
            print(f"in_segment_separator = {self.in_segment_separator}")
            print(f"in_sentence_separator = {self.in_sentence_separator}")
            print(f"out_segment_separator = {self.out_segment_separator}")
            print(f"out_sentence_separator = {self.out_sentence_separator}")
            print(f"return_type = {self.return_type}")
            print(f"max_split_length = {self.max_split_length}")
            print(f"max_split_iterations = {self.max_split_iterations}")
            print(f"keep_segment = {self.keep_segment}")
            print(f"merge_lines = {self.merge_lines}")
            print(f"merge_level = {self.merge_level}")
            print(f"empty_lines_threshold = {self.empty_lines_threshold}")
            print(f"broken_lines_threshold = {self.broken_lines_threshold}")
            print(f"filter_language = {self.filter_language}")
            print(f"detection_level = {self.detection_level}")
            print(f"languages_to_keep = {self.languages_to_keep}")
            print(f"min_language_probability = {self.min_language_probability}")
            print(f"filter_programming_language = {self.filter_programming_language}")
            print(f"filter_sentence_length = {self.filter_sentence_length}")
            print(f"min_num_words = {self.min_num_words}")
            print(f"min_length = {self.min_length}")
            print(f"chunk_size = {self.chunk_size}")
            print(f"chunk_overlap = {self.chunk_overlap}")
            print(f"verbose = {self.verbose}")
            print(f"kwargs = {self.kwargs}")

        self.guess = None

    @abstractmethod
    def segment(self, text):
        raise NotImplementedError("Must override segment")

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
            print(f"num_empty_lines={num_empty_lines}")
            print(f"num_lines_endswith_period={num_lines_endswith_period}")
            print(f"num_lines={num_lines}")
            print(f"empty_lines_ratio={empty_lines_ratio}")
            print(f"broken_lines_ratio={broken_lines_ratio}")
        if empty_lines_ratio > self.empty_lines_threshold:
            return True
        if broken_lines_ratio > self.broken_lines_threshold:
            return True

    def segment_article(self, article):

        if article is None:
            return None

        article = re.sub(
            f"{self.in_segment_separator}+", self.in_segment_separator, article
        )
        if self.merge_lines == "auto":
            merge_lines = self._check_if_merge_lines(article)
        else:
            merge_lines = self.merge_lines
        if merge_lines and self.merge_level == "article":
            if self.verbose > 10:
                print("=> merging lines of the article")
            article = re.sub(r"\s+", " ", article)
        if self.keep_segment:
            in_segments = article.split(self.in_segment_separator)
        else:
            in_segments = [article]

        segments = []
        for i, segment in enumerate(in_segments):
            segment = segment.strip()
            if len(segment) > 0:
                if self.filter_programming_language:
                    if self._skip_programming_language(segment):
                        continue
                if merge_lines and self.merge_level == "segment":
                    if self.verbose > 10:
                        print("=> merging lines of the segment")
                    segment = re.sub(r"\s+", " ", segment)

                if self.filter_language and self.detection_level == "segment":
                    if self._skip_language(segment):
                        continue
                sentences = []
                for sent in segment.split(self.in_sentence_separator):
                    sent = sent.strip()
                    if len(sent) > 0:

                        if self.filter_language and self.detection_level == "sentence":
                            if self._skip_language(sent):
                                continue

                        try:
                            sentences.extend(self._split_sentences(sent))
                        except Exception as e:
                            print(f"===> skipping segment {i}: {e}")
                            continue

                sentences = self._filter_sentence_length(sentences)
                if self.return_type == "list":
                    segments.append(sentences)
                else:
                    segments.append(self.out_sentence_separator.join(sentences))

        return (
            segments
            if self.return_type == "list"
            else self.out_segment_separator.join(segments)
        )

    def _filter_sentence_length(self, sentences):
        if self.filter_sentence_length:
            sentences = [
                sent
                for sent in sentences
                if len(sent.split()) >= self.min_num_words
                and len(sent) >= self.min_length
            ]
        return sentences

    def _skip_language(self, text):
        from ftlangdetect import detect

        lang = detect(text.replace("\n", " "), low_memory=False)
        if (
            lang is not None
            and lang["lang"] not in self.languages_to_keep
            and lang["score"] < self.min_language_probability
        ):
            if self.verbose > 5:
                print(f"===> lang={lang} not in languages_to_keep")
            return True
        return False

    def _skip_programming_language(self, text):
        from ekorpkit.models.guesslang import Guess

        if self.guess is None:
            self.guess = Guess()

        # Guess the language from code
        lang = self.guess.language_name(text)
        if lang is not None:
            if self.verbose > 5:
                print(f"===> Programming language: {lang}")
            if self.verbose > 10:
                print(text)
            return True
        return False

    def _split_sentences(self, text):
        text = text.strip()
        if len(text) <= self.max_split_length:
            return self.segment(text)
        else:
            if self.verbose:
                print(f"==> too long text: {len(text)}")
            text_chunk = text[: self.max_split_length]
            next_chunk = text[self.max_split_length :]
            sentences = []
            split_iter = 0
            while len(text_chunk) > 0 and split_iter < self.max_split_iterations:
                sents = self.segment(text_chunk)
                if len(sents) > 1 and len(next_chunk) > 0:
                    sentences.extend(sents[:-1])
                    last_sent = sents[-1]
                else:
                    sentences.extend(sents)
                    last_sent = ""
                end_pos = self.max_split_length - len(last_sent)
                if end_pos >= len(next_chunk):
                    text_chunk = last_sent + next_chunk
                    next_chunk = ""
                else:
                    text_chunk = last_sent + next_chunk[:end_pos]
                    next_chunk = next_chunk[end_pos:]
                split_iter += 1
            if split_iter >= self.max_split_iterations:
                if self.verbose:
                    print(f"==> too many split iterations: {split_iter}")
                raise Exception(f"Maximum number of iterations reached: {split_iter}")
            return sentences

    def segment_articles(self, articles):
        return [self.segment_article(article) for article in articles]

    def chunk_article(self, article):

        if article is None:
            return None

        article = re.sub(
            f"{self.in_segment_separator}+", self.in_segment_separator, article
        )
        if self.keep_segment:
            in_segments = article.split(self.in_segment_separator)
        else:
            in_segments = [article]

        segments = []
        for i, segment in enumerate(in_segments):
            segment = segment.strip()
            segment = re.sub(
                f"{self.in_sentence_separator}+", self.in_sentence_separator, segment
            )
            if len(segment) > 0:
                sentences = segment.split(self.in_sentence_separator)

                chunks = self.chunk(
                    sentences,
                    max_length=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
                if self.return_type != "list":
                    chunks = [
                        self.out_sentence_separator.join(chunk) for chunk in chunks
                    ]
                segments += chunks

        return (
            segments
            if self.return_type == "list"
            else self.out_segment_separator.join(segments)
        )

    def chunk(
        self,
        sentences: list,
        max_length: int,
        overlap: bool = False,
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

        span, chunks = [], []

        for offset in _get_sentence_offsets(sentences):
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


def _get_sentence_offsets(sentences):
    sentence_offsets = []
    offset = None
    for i, sentence in enumerate(sentences):
        prev_cum_len = offset.this_cum_len if offset is not None else 0
        offset = Offset(i, i + 1, prev_cum_len, prev_cum_len + len(sentence))
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

        nltk.download("punkt")
        super().__init__(**kwargs)

    def segment(self, text):
        import nltk

        return nltk.tokenize.sent_tokenize(text)


class SimpleSegmenter(Segmenter):
    def __init__(self, separator="\n", remove_newlines=True, **kwargs):
        self.separator = codecs.decode(separator, "unicode_escape")
        self.remove_newlines = remove_newlines
        print(f"segment_separator: -->{self.separator}<--")
        if remove_newlines:
            print("Remove newlines inside segments")
        super().__init__(**kwargs)

    def segment(self, text):
        segments = text.split(self.separator)
        if self.remove_newlines:
            segments = [segment.replace("\n", " ") for segment in segments]
        return segments


class PySBDSegmenter(Segmenter):
    def __init__(self, language="en", clean=False, doc_type=None, **kwargs):
        import pysbd

        self.language = language
        self.clean = clean
        self.doc_type = doc_type
        if doc_type == "pdf":
            self.clean = True
        self.seg = pysbd.Segmenter(
            language=self.language, clean=self.clean, doc_type=self.doc_type
        )
        super().__init__(**kwargs)

    def segment(self, text):
        # import pysbd
        # seg = pysbd.Segmenter(language=self.language, clean=self.clean, doc_type=self.doc_type)
        # results = self.seg.segment(text)
        # del seg
        return self.seg.segment(text)


class KSSSegmenter(Segmenter):
    def __init__(self, use_heuristic=False, backend="fugashi", **kwargs):
        from ekorpkit.preprocessors import KSS

        self.use_heuristic = use_heuristic
        self.backend = backend
        self.kwargs = kwargs
        self.seg = KSS(
            use_heuristic=self.use_heuristic, backend=self.backend, **self.kwargs
        )
        super().__init__(**kwargs)

    def segment(self, text):
        # from ekorpkit.preprocessors import KSS
        # seg = KSS(use_heuristic=self.use_heuristic, backend=self.backend, **self.kwargs)
        return self.seg.segment(text)
