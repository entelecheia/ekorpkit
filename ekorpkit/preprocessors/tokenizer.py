import codecs
import logging
from abc import ABCMeta, abstractmethod
from .normalizer import strict_normalize, only_text

logging.basicConfig(format="[ekorpkit]: %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)


class Tokenizer:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        **kwargs,
    ):
        self.only_text = kwargs.get("only_text", True)
        self.tokenize_each_word = kwargs.get("tokenize_each_word", False)
        self.wordpieces_prefix = kwargs.get("wordpieces_prefix", "##")
        self.noun_pos = kwargs.get("noun_pos", None)
        if self.noun_pos is None:
            self.noun_pos = ["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"]
        self.exclude_pos = kwargs.get("exclude_pos", None)
        if self.exclude_pos is None:
            self.exclude_pos = ["SP"]
        self.no_space_for_non_nouns = kwargs.get("no_space_for_non_nouns", False)
        self.flatten = kwargs.get("flatten", True)
        self.join_pos = kwargs.get("join_pos", True)
        self.include_whitespace_token = kwargs.get("include_whitespace_token", True)
        self.punct_pos = kwargs.get("punct_pos", None)
        if self.punct_pos is None:
            self.punct_pos = ["SF", "SP", "SSO", "SSC", "SY"]
        self.sentence_separator = kwargs.get("sentence_separator", None)
        if self.sentence_separator is None:
            self.sentence_separator = "\n"
        self.sentence_separator = codecs.decode(
            self.sentence_separator, "unicode_escape"
        )

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError

    def tokenize_article(self, article):
        if article is None:
            return None

        tokenized_article = []
        for sent in article.split(self.sentence_separator):
            sent = sent.strip()
            if len(sent) > 0:
                tokenized_article.append(" ".join(self.tokenize(sent)))
            else:
                tokenized_article.append("")
        return self.sentence_separator.join(tokenized_article)

    def tokenize(self, text):
        if self.only_text:
            text = only_text(strict_normalize(text))
        if self.tokenize_each_word:
            term_pos = []
            for word in text.split():
                term_pos += self._tokenize_word(word)
            return term_pos
        else:
            text = " ".join(text.split())
            return self.parse(text)

    def _tokenize_word(self, word):
        tokens = self.parse(word)
        term_pos = []
        for i, (term, pos) in enumerate(tokens):
            term = f"{term}/{pos}" if self.join_pos else term
            if (
                i > 0
                and pos not in self.punct_pos
                and tokens[i - 1][1] not in self.punct_pos
            ):
                term = f"{self.wordpieces_prefix}{term}"
            term_pos.append(term)
        return term_pos

    def extract(self, article, nouns_only=False):
        if article is None:
            return None

        tokens_article = []
        for sent in article.split(self.sentence_separator):
            sent = sent.strip()
            if len(sent) > 0:
                if nouns_only:
                    tokens = extract_tokens(
                        sent, nouns_only=True, noun_pos=self.noun_pos
                    )
                else:
                    tokens = extract_tokens(
                        sent,
                        nouns_only=False,
                        exclude_pos=self.exclude_pos,
                        no_space_for_non_nouns=self.no_space_for_non_nouns,
                    )
                tokens_article.append(" ".join(tokens))
            else:
                tokens_article.append("")
        return self.sentence_separator.join(tokens_article)

    def extract_tokens(self, article):
        return self.extract(article, nouns_only=False)

    def extract_nouns(self, article):
        return self.extract(article, nouns_only=True)

    def nouns(self, text):
        tokenized_text = self.tokenize(text)
        return extract_tokens(tokenized_text, nouns_only=True, noun_pos=self.noun_pos)

    def tokens(self, text):
        tokenized_text = self.tokenize(text)
        return extract_tokens(
            tokenized_text,
            nouns_only=False,
            exclude_pos=self.exclude_pos,
            no_space_for_non_nouns=self.no_space_for_non_nouns,
        )


class PynoriTokenizer(Tokenizer):
    def __init__(
        self,
        pynori=None,
        **kwargs,
    ):
        logging.warning("Initializing Pynori...")

        super().__init__(**kwargs)
        if pynori is None:
            pynori = {}
        pynori["decompound_mode"] = "NONE" if self.flatten else "MIXED"
        pynori["infl_decompound_mode"] = "NONE" if self.flatten else "MIXED"
        pynori["discard_punctuation"] = False

        try:
            from pynori.korean_analyzer import KoreanAnalyzer

            if pynori is None:
                self._tokenizer = KoreanAnalyzer()
            else:
                self._tokenizer = KoreanAnalyzer(**pynori)
        except ImportError:
            raise ImportError(
                "\n"
                "You must install `pynori` if you want to use `pynori` backend.\n"
                "Please install using `pip install pynori`.\n"
            )

    def parse(self, text):

        join_pos = False if self.tokenize_each_word else self.join_pos
        tokens = self._tokenizer.do_analysis(text)
        term_pos = [
            f"{term}/{pos}" if join_pos else (term, pos)
            for term, pos in zip(tokens["termAtt"], tokens["posTagAtt"])
            if self.include_whitespace_token or pos != "SP"
        ]
        return term_pos


class MecabTokenizer(Tokenizer):
    def __init__(
        self,
        mecab=None,
        **kwargs,
    ):

        logging.warning("Initializing mecab...)")
        super().__init__(**kwargs)
        self.mecab = mecab
        try:
            from ..tokenizers.mecab import MeCab

            if self.mecab is None:
                _tokenizer = MeCab()
            else:
                _tokenizer = MeCab(**self.mecab)
        except ImportError:
            raise ImportError(
                "\n"
                "You must install `fugashi` and `mecab_ko_dic` if you want to use `mecab` backend.\n"
                "Please install using `pip install python-mecab-ko`.\n"
            )

    def parse(self, text):
        from ..tokenizers.mecab import MeCab

        if self.mecab is None:
            self._tokenizer = MeCab()
        else:
            self._tokenizer = MeCab(**self.mecab)

        join_pos = False if self.tokenize_each_word else self.join_pos
        return self._tokenizer.pos(
            text,
            join=join_pos,
            flatten=self.flatten,
            include_whitespace_token=self.include_whitespace_token,
        )


class BWPTokenizer(Tokenizer):
    def __init__(
        self,
        bwp=None,
        **kwargs,
    ):
        logging.warning("Initializing BertWordPieceTokenizer...")
        super().__init__(**kwargs)
        try:
            from transformers import BertTokenizerFast

            if bwp is None:
                self._tokenizer = BertTokenizerFast.from_pretrained(
                    "entelecheia/ekonbert-base"
                )
            else:
                self._tokenizer = BertTokenizerFast.from_pretrained(**bwp)

        except ImportError:
            raise ImportError(
                "\n"
                "You must install `BertWordPieceTokenizer` if you want to use `bwp` backend.\n"
                "Please install using `pip install transformers`.\n"
            )

    def parse(self, text):
        return self._tokenizer.tokenize(text)


def extract_tokens(
    tokenized_text,
    nouns_only=False,
    noun_pos=["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"],
    exclude_pos=["SP"],
    no_space_for_non_nouns=False,
    **kwargs,
):

    _tokens_pos = [
        token.split("/")
        for token in tokenized_text.split()
        if len(token.split("/")) == 2
    ]

    if nouns_only:
        _tokens = [token[0].strip() for token in _tokens_pos if token[1] in noun_pos]
    else:
        exist_sp_tag = False
        for i, token in enumerate(_tokens_pos):
            if token[1] == "SP":
                exist_sp_tag = True
                break

        _tokens = []
        if exist_sp_tag and no_space_for_non_nouns:
            prev_nonnoun_check = False
            cont_morphs = []
            i = 0
            while i < len(_tokens_pos):
                token = _tokens_pos[i]
                if not prev_nonnoun_check and token[1] in noun_pos:
                    _tokens.append(token[0])
                elif (
                    not prev_nonnoun_check
                    and token[1] not in noun_pos
                    and token[1][0] != "S"
                ):
                    prev_nonnoun_check = True
                    cont_morphs.append(token[0])
                elif (
                    prev_nonnoun_check
                    and token[1] not in noun_pos
                    and token[1][0] != "S"
                ):
                    cont_morphs.append(token[0])
                else:
                    if len(cont_morphs) > 0:
                        _tokens.append("".join(cont_morphs))
                        cont_morphs = []
                        prev_nonnoun_check = False
                    if token[1] != "SP":
                        _tokens.append(token[0])
                i += 1
            if len(cont_morphs) > 0:
                _tokens.append("".join(cont_morphs))
        else:
            _tokens = [
                token[0].strip() for token in _tokens_pos if token[1] not in exclude_pos
            ]
    return _tokens
