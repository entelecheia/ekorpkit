import codecs
import logging
from abc import ABCMeta
from ekorpkit.io.load.list import load_wordlist
from ekorpkit import eKonf

logging.basicConfig(format="[ekorpkit]: %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)


class Tokenizer:
    __metaclass__ = ABCMeta

    def __init__(
        self,
        normalize=None,
        tokenize={},
        tokenize_article={},
        extract={},
        verbose=False,
        **kwargs,
    ):

        self.verbose = verbose
        if tokenize is None:
            tokenize = {}
        if tokenize_article is None:
            tokenize_article = {}
        if extract is None:
            extract = {}

        self._normalize = normalize
        if eKonf.is_instantiatable(self._normalize):
            if self.verbose:
                print(f"[ekorpkit]: instantiating {self._normalize['_target_']}...")
            self._normalize = eKonf.instantiate(self._normalize)
            # print(f"[ekorpkit]: {self._normalize.__name__} is instantiated.")


        self._lowercase = tokenize.get("lowercase", False)
        self._flatten = tokenize.get("flatten", True)
        self._concat_surface_and_pos = tokenize.get("concat_surface_and_pos", True)
        self._include_whitespace_token = tokenize.get("include_whitespace_token", True)
        self._tokenize_each_word = tokenize.get("tokenize_each_word", False)
        self._wordpieces_prefix = tokenize.get("wordpieces_prefix", "##")
        self._punct_postags = tokenize.get("punct_postags", None)
        if self._punct_postags is None:
            self._punct_postags = ["SF", "SP", "SSO", "SSC", "SY"]

        return_type = tokenize_article.get("return_type", "str")
        self.return_type = str(return_type).lower().strip()
        if self.return_type not in ["str", "list"]:
            raise ValueError(
                f"Invalid return_type: {self.return_type}. "
                f"Valid values are 'str' and 'list'."
            )
        self._sentence_separator = tokenize_article.get("sentence_separator", None)
        if self._sentence_separator is None:
            self._sentence_separator = "\n"
        self._sentence_separator = codecs.decode(
            self._sentence_separator, "unicode_escape"
        )

        self._noun_postags = extract.get("noun_postags", None)
        if self._noun_postags is None:
            self._noun_postags = ["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"]
        self._stop_postags = extract.get("stop_postags", None)
        if self._stop_postags is None:
            self._stop_postags = ["SP"]
        self._no_space_for_non_nouns = extract.get("no_space_for_non_nouns", False)
        self._stopwords_path = extract.get("stopwords_path", None)
        if self._stopwords_path is not None:
            self._stopwords = load_wordlist(
                self._stopwords_path, lowercase=True, verbose=self.verbose
            )
            if self.verbose:
                logger.info(f"Loaded {len(self._stopwords)} stopwords")
        else:
            self._stopwords = []
        stopwords = extract.get("stopwords", None)
        if stopwords is not None:
            self._stopwords += stopwords

        if self.verbose:
            print(f"{self.__class__.__name__} initialized with:")
            print(f"\treturn_type: {self.return_type}")
            print(f"\tstopwords_path: {self._stopwords_path}")

    def parse(self, text):
        return text.split()

    def tokenize_article(self, article, return_type=None):
        if article is None:
            return None
        if return_type is None:
            return_type = self.return_type

        tokenized_article = []
        for sent in article.split(self._sentence_separator):
            sent = sent.strip()
            tokens = self.tokenize(sent, return_type=return_type)
            tokenized_article.append(tokens)
        return (
            tokenized_article
            if str(return_type) == "list"
            else self._sentence_separator.join(tokenized_article)
        )

    def tokenize(self, text, return_type="list"):
        if self._lowercase:
            text = text.lower()
        if self._normalize and callable(self._normalize):
            text = self._normalize(text)
        if len(text) > 0:
            if self._tokenize_each_word:
                term_pos = []
                for word in text.split():
                    term_pos += self._tokenize_word(word)
            else:
                text = " ".join(text.split())
                term_pos = self.parse(text)
        else:
            term_pos = []
        return term_pos if str(return_type) == "list" else " ".join(term_pos)

    def pos(self, text):
        return self.tokenize(text)

    def _tokenize_word(self, word):
        tokens = self.parse(word)
        term_pos = []
        for i, (term, pos) in enumerate(tokens):
            term = f"{term}/{pos}" if self._concat_surface_and_pos else term
            if (
                i > 0
                and pos not in self._punct_postags
                and tokens[i - 1][1] not in self._punct_postags
            ):
                term = f"{self._wordpieces_prefix}{term}"
            term_pos.append(term)
        return term_pos

    def extract(self, text, nouns_only=False):
        if len(text) > 0:
            if nouns_only:
                tokens = extract_tokens(
                    text,
                    nouns_only=True,
                    noun_postags=self._noun_postags,
                    stopwords=self._stopwords,
                )
            else:
                tokens = extract_tokens(
                    text,
                    nouns_only=False,
                    stop_postags=self._stop_postags,
                    no_space_for_non_nouns=self._no_space_for_non_nouns,
                    stopwords=self._stopwords,
                )
        else:
            tokens = []
        return tokens if self.return_type == "list" else " ".join(tokens)

    def extract_article(self, article, nouns_only=False):
        if article is None:
            return None

        tokens_article = []
        for sent in article.split(self._sentence_separator):
            sent = sent.strip()
            tokens = self.extract(sent, nouns_only=nouns_only)
            tokens_article.append(tokens)
        return (
            tokens_article
            if self.return_type == "list"
            else self._sentence_separator.join(tokens_article)
        )

    def extract_tokens(self, article):
        return self.extract_article(article, nouns_only=False)

    def extract_nouns(self, article):
        return self.extract_article(article, nouns_only=True)

    def filter_stopwords(self, text_or_tokens):
        if text_or_tokens is None:
            return None

        if isinstance(text_or_tokens, list):
            tokens = text_or_tokens
        else:
            tokens = self.tokenize(text_or_tokens)

        tokens = [token for token in tokens if token.lower() not in self._stopwords]

        return tokens if self.return_type == "list" else " ".join(tokens)

    def filter_article_stopwords(self, article):
        if article is None:
            return None

        tokens_article = []
        for sent in article.split(self._sentence_separator):
            tokens_article.append(self.filter_stopwords(sent))
        return (
            tokens_article
            if self.return_type == "list"
            else self._sentence_separator.join(tokens_article)
        )

    def nouns(self, text):
        tokenized_text = self.tokenize(text)
        return extract_tokens(tokenized_text, nouns_only=True, noun_postags=self._noun_postags)

    def tokens(self, text):
        tokenized_text = self.tokenize(text)
        return extract_tokens(
            tokenized_text,
            nouns_only=False,
            stop_postags=self._stop_postags,
            no_space_for_non_nouns=self._no_space_for_non_nouns,
        )

    def morphs(self, text):
        return self.tokens(text)


class SimpleTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse(self, text):
        return text.split()


class PynoriTokenizer(Tokenizer):
    def __init__(
        self,
        pynori={},
        **kwargs,
    ):
        logging.warning("Initializing Pynori...")

        super().__init__(**kwargs)
        if pynori is None:
            pynori = {}
        pynori["decompound_mode"] = "NONE" if self._flatten else "MIXED"
        pynori["infl_decompound_mode"] = "NONE" if self._flatten else "MIXED"
        pynori["discard_punctuation"] = False
        if pynori["path_userdict"] is None:
            pynori.pop("path_userdict")

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

        _concat_surface_and_pos = (
            False if self._tokenize_each_word else self._concat_surface_and_pos
        )
        tokens = self._tokenizer.do_analysis(text)
        term_pos = [
            f"{term}/{pos}" if _concat_surface_and_pos else (term, pos)
            for term, pos in zip(tokens["termAtt"], tokens["posTagAtt"])
            if self._include_whitespace_token or pos != "SP"
        ]
        return term_pos


class MecabTokenizer(Tokenizer):
    def __init__(
        self,
        mecab={},
        **kwargs,
    ):

        logging.warning("Initializing mecab...")
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

        concat_token_and_pos = (
            False if self._tokenize_each_word else self._concat_surface_and_pos
        )
        return self._tokenizer.pos(
            text,
            join=concat_token_and_pos,
            flatten=self._flatten,
            include_whitespace_token=self._include_whitespace_token,
        )


class BWPTokenizer(Tokenizer):
    def __init__(
        self,
        bwp={},
        **kwargs,
    ):
        logging.warning("Initializing BertWordPieceTokenizer...")
        super().__init__(**kwargs)
        try:
            from transformers import BertTokenizerFast

            if bwp is None or bwp.get("pretrained_model_name_or_path") is None:
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
    noun_postags=["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"],
    stop_postags=["SP"],
    no_space_for_non_nouns=False,
    filter_stopwords_only=False,
    stopwords=[],
    **kwargs,
):
    if isinstance(tokenized_text, str):
        tokens = tokenized_text.split()
    else:
        tokens = tokenized_text
    _tokens_pos = [token.split("/") for token in tokens if len(token.split("/")) == 2]

    if nouns_only:
        _tokens = [token[0].strip() for token in _tokens_pos if token[1] in noun_postags]
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
                if not prev_nonnoun_check and token[1] in noun_postags:
                    _tokens.append(token[0])
                elif (
                    not prev_nonnoun_check
                    and token[1] not in noun_postags
                    and token[1][0] != "S"
                ):
                    prev_nonnoun_check = True
                    cont_morphs.append(token[0])
                elif (
                    prev_nonnoun_check
                    and token[1] not in noun_postags
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
                token[0].strip() for token in _tokens_pos if token[1] not in stop_postags
            ]

    if stopwords is not None and len(stopwords) > 0:
        _tokens = [token for token in _tokens if token.lower() not in stopwords]
    return _tokens
