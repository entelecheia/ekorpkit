import codecs
import logging
from abc import ABCMeta
from ekorpkit.io.load.list import load_wordlist
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def _match_tags(token, tags):
    for tag in tags:
        if token[1].startswith(tag):
            return True
    return False


def _extract_tokens(
    tokenized_text,
    postags=[],
    stop_postags=["SP"],
    stopwords=None,
    strip_pos=True,
    postag_delim="/",
    postag_length=None,
    **kwargs,
):
    if isinstance(tokenized_text, str):
        tokens = tokenized_text.split()
    else:
        tokens = tokenized_text
    _token_pos_tuples = [
        _token_to_tuple(token, postag_delim=postag_delim, postag_length=postag_length)
        for token in tokens
    ]
    postags = [
        postag[:postag_length] if postag_length else postag
        for postag in postags
        if postag not in stop_postags
    ]
    stop_postags = [
        postag[:postag_length] if postag_length else postag for postag in stop_postags
    ]
    _tokens = []
    if len(postags) > 0:
        _tokens = [
            token_pos
            for token_pos in _token_pos_tuples
            if len(token_pos) == 1
            or (
                not _match_tags(token_pos, stop_postags)
                and _match_tags(token_pos, postags)
            )
        ]
    else:
        _tokens = [
            token_pos
            for token_pos in _token_pos_tuples
            if len(token_pos) == 1 or not _match_tags(token_pos, stop_postags)
        ]
    if stopwords is not None:
        _tokens = [token_pos for token_pos in _tokens if not stopwords(token_pos[0])]

    _tokens = [
        _tuple_to_token(
            token_pos,
            strip_pos=strip_pos,
            postag_delim=postag_delim,
            postag_length=postag_length,
        )
        for token_pos in _tokens
    ]
    return _tokens


def _tuple_to_token(token_pos, strip_pos=True, postag_delim="/", postag_length=None):
    if strip_pos or len(token_pos) == 1:
        return token_pos[0]
    return (
        token_pos[0].strip()
        + postag_delim
        + (token_pos[1][:postag_length] if postag_length else token_pos[1])
    )


def _token_to_tuple(_token, postag_delim="/", postag_length=None):
    if isinstance(_token, str):
        token_pos = _token.split(postag_delim)
        if len(token_pos) == 2:
            return (
                token_pos[0],
                token_pos[1][:postag_length] if postag_length else token_pos[1],
            )
        return tuple(token_pos)
    return _token


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
                log.info(f"instantiating {self._normalize['_target_']}...")
            self._normalize = eKonf.instantiate(self._normalize)

        self._lowercase = tokenize.get("lowercase", False)
        self._flatten = tokenize.get("flatten", True)
        self._strip_pos = tokenize.get("strip_pos", False)
        self._postag_delim = tokenize.get("postag_delim", "/")
        self._postag_length = tokenize.get("postag_length", None)
        self._include_whitespace_token = tokenize.get("include_whitespace_token", True)
        self._tokenize_each_word = tokenize.get("tokenize_each_word", False)
        self._wordpieces_prefix = tokenize.get("wordpieces_prefix", "##")
        self._punct_postags = tokenize.get("punct_postags", None)
        if self._punct_postags is None:
            self._punct_postags = ["SF", "SP", "SSO", "SSC", "SY"]

        self._sentence_separator = tokenize_article.get("sentence_separator", None)
        if self._sentence_separator is None:
            self._sentence_separator = "\n"
        self._sentence_separator = codecs.decode(
            self._sentence_separator, "unicode_escape"
        )

        self._postags = extract.get("postags", None)
        if self._postags is None:
            self._postags = []
        self._noun_postags = extract.get("noun_postags", None)
        if self._noun_postags is None:
            self._noun_postags = ["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"]
        self._stop_postags = extract.get("stop_postags", None)
        if self._stop_postags is None:
            self._stop_postags = ["SP"]
        self._extract_strip_pos = extract.get("strip_pos", True)
        self._extract_postag_delim = extract.get("postag_delim", "/")
        self._extract_postag_length = extract.get("postag_length", None)

        stopwords = kwargs.get("stopwords")
        self._stopwords = stopwords
        if eKonf.is_instantiatable(self._stopwords):
            log.info(f"instantiating {self._stopwords['_target_']}...")
            self._stopwords = eKonf.instantiate(self._stopwords)

        self._return_as_list = kwargs.get("return_as_list", False)
        if self.verbose:
            log.info(f"{self.__class__.__name__} initialized with:")
            log.info(f"\treturn_as_list: {self._return_as_list}")

    def __call__(self, text):
        """Calling a tokenizer instance like a function just calls the tokenize method."""
        return self.tokenize(text)

    def parse(self, text):
        return text.split()

    def _to_token(self, term_pos):
        if isinstance(term_pos, tuple):
            return _tuple_to_token(
                term_pos,
                self._strip_pos,
                self._postag_delim,
                self._postag_length,
            )
        return term_pos

    def tokenize_article(self, article, return_as_list=None):
        if article is None:
            return None
        if return_as_list is None:
            return_as_list = self._return_as_list

        tokenized_article = []
        for sent in article.split(self._sentence_separator):
            sent = sent.strip()
            tokens = self.tokenize(sent, return_as_list=return_as_list)
            tokenized_article.append(tokens)
        return (
            tokenized_article
            if return_as_list
            else self._sentence_separator.join(tokenized_article)
        )

    def tokenize(self, text, return_as_list=True):
        if isinstance(text, list):
            return text
        text = str(text)
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
                term_pos = [self._to_token(token) for token in self.parse(text)]
        else:
            term_pos = []
        return term_pos if return_as_list else " ".join(term_pos)

    def pos(self, text):
        return self.tokenize(text)

    def _tokenize_word(self, word):
        tokens = self.parse(word)
        if len(tokens) > 1 and isinstance(tokens[0], tuple):
            term_pos = []
            for i, (term, pos) in enumerate(tokens):
                term = self._to_token((term, pos))
                if (
                    i > 0
                    and pos not in self._punct_postags
                    and tokens[i - 1][1] not in self._punct_postags
                ):
                    term = f"{self._wordpieces_prefix}{term}"
                term_pos.append(term)
            return term_pos
        return tokens

    def extract(
        self,
        text,
        nouns_only=False,
        return_as_list=True,
        postags=None,
        stop_postags=None,
        strip_pos=None,
        postag_delim=None,
        postag_length=None,
    ):
        if strip_pos is None:
            strip_pos = self._extract_strip_pos
        if stop_postags is None:
            stop_postags = self._stop_postags
        if postags is None:
            postags = self._noun_postags if nouns_only else self._postags
        if postag_delim is None:
            postag_delim = self._extract_postag_delim
        if postag_length is None:
            postag_length = self._extract_postag_length

        tokens = _extract_tokens(
            text,
            postags=postags,
            stop_postags=stop_postags,
            stopwords=self._stopwords,
            strip_pos=strip_pos,
            postag_delim=postag_delim,
            postag_length=postag_length,
        )
        return tokens if return_as_list else " ".join(tokens)

    def extract_article(self, article, nouns_only=False):
        if article is None:
            return None

        tokens_article = []
        for sent in article.split(self._sentence_separator):
            sent = sent.strip()
            tokens = self.extract(
                sent, nouns_only=nouns_only, return_as_list=self._return_as_list
            )
            tokens_article.append(tokens)
        return (
            tokens_article
            if self._return_as_list
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

        tokens = [token for token in tokens if not self._stopwords(token)]

        return tokens if self._return_as_list else " ".join(tokens)

    def filter_article_stopwords(self, article):
        if article is None:
            return None

        token_article = []
        for sent in article.split(self._sentence_separator):
            token_article.append(self.filter_stopwords(sent))
        return (
            token_article
            if self._return_as_list
            else self._sentence_separator.join(token_article)
        )

    def nouns(self, text):
        tokens = self.tokenize(text)
        return self.extract(tokens, nouns_only=True)

    def tokens(self, text):
        tokens = self.tokenize(text)
        return self.extract(tokens, nouns_only=False)

    def morphs(self, text):
        return self.tokens(text)


class NLTKTokenizer(Tokenizer):
    def __init__(self, nltk={}, **kwargs):
        import nltk as NLTK

        NLTK.download("punkt", quiet=True)
        NLTK.download("averaged_perceptron_tagger", quiet=True)
        NLTK.download("wordnet", quiet=True)
        NLTK.download("omw-1.4", quiet=True)

        nltk = nltk or {}
        self.verbose = kwargs.get("verbose", False)

        self.lemmatizer = nltk.get("lemmatizer", None)
        if eKonf.is_instantiatable(self.lemmatizer):
            if self.verbose:
                log.info(f"instantiating {self.lemmatizer['_target_']}...")
            self.lemmatizer = eKonf.instantiate(self.lemmatizer)
        self.stemmer = nltk.get("stemmer", None)
        if eKonf.is_instantiatable(self.stemmer):
            if self.verbose:
                log.info(f"instantiating {self.stemmer['_target_']}...")
            self.stemmer = eKonf.instantiate(self.stemmer)
        do_lemmatize = nltk.get("lemmatize", False)
        do_stem = nltk.get("stem", False)
        self.do_lemmatize = do_lemmatize and self.lemmatizer is not None
        self.do_stem = do_stem and self.stemmer is not None and not self.do_lemmatize

        super().__init__(**kwargs)

    def parse(self, text):
        import nltk

        tokens = nltk.pos_tag(nltk.word_tokenize(text))
        if self.do_lemmatize:
            tokens = self.lemmatize(tokens)
        if self.do_stem:
            tokens = self.stem(tokens)

        return tokens

    def _lemmatize(self, token):
        if self.lemmatizer is None:
            return token
        token_pos = _token_to_tuple(token)
        if len(token_pos) == 2:
            return (
                self.lemmatizer.lemmatize(
                    token_pos[0], self._get_wordnet_pos(token_pos[1])
                ),
                token_pos[1],
            )
        return self.lemmatizer.lemmatize(token_pos[0])

    def _stem(self, token):
        if self.stemmer is None:
            return token
        token_pos = _token_to_tuple(token)
        if len(token_pos) == 2:
            return (self.stemmer.stem(token_pos[0]), token_pos[1])
        return self.stemmer.stem(token_pos[0])

    def lemmatize(self, tokens):
        tokens = [self._lemmatize(token) for token in tokens]
        return tokens

    def stem(self, tokens):
        tokens = [self._stem(token) for token in tokens]
        return tokens

    @staticmethod
    def _get_wordnet_pos(tag):
        from nltk.corpus import wordnet

        """Map POS tag to first character lemmatize() accepts"""
        tag = tag[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        return tag_dict.get(tag, wordnet.NOUN)


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
        log.info(f"Initializing Pynori with {pynori}...")

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

        tokens = self._tokenizer.do_analysis(text)
        term_pos = [
            (term, pos)
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

        log.info(f"Initializing mecab with {mecab}...")
        super().__init__(**kwargs)
        self.mecab = mecab
        try:
            from ..models.tokenizer.mecab import MeCab

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
        from ..models.tokenizer.mecab import MeCab

        if self.mecab is None:
            self._tokenizer = MeCab()
        else:
            self._tokenizer = MeCab(**self.mecab)

        return self._tokenizer.pos(
            text,
            concat_surface_and_pos=False,
            flatten=self._flatten,
            include_whitespace_token=self._include_whitespace_token,
        )


class BWPTokenizer(Tokenizer):
    def __init__(
        self,
        bwp={},
        **kwargs,
    ):
        log.info("Initializing BertWordPieceTokenizer...")
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


# def _extract(
#     tokenized_text,
#     postags=[],
#     noun_postags=["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"],
#     stop_postags=["SP"],
#     no_space_for_non_nouns=False,
#     stopwords=None,
#     **kwargs,
# ):
#     if isinstance(tokenized_text, str):
#         tokens = tokenized_text.split()
#     else:
#         tokens = tokenized_text
#     _tokens_pos = [
#         _token_to_tuple(token) for token in tokens if len(_token_to_tuple(token)) == 2
#     ]

#     exist_sp_tag = False
#     if no_space_for_non_nouns:
#         for i, token in enumerate(_tokens_pos):
#             if token[1] == "SP":
#                 exist_sp_tag = True
#                 break

#     _tokens = []
#     if exist_sp_tag and no_space_for_non_nouns:
#         prev_nonnoun_check = False
#         cont_morphs = []
#         i = 0
#         while i < len(_tokens_pos):
#             token = _tokens_pos[i]
#             if not prev_nonnoun_check and token[1] in noun_postags:
#                 _tokens.append(token[0])
#             elif (
#                 not prev_nonnoun_check
#                 and token[1] not in noun_postags
#                 and token[1][0] != "S"
#             ):
#                 prev_nonnoun_check = True
#                 cont_morphs.append(token[0])
#             elif (
#                 prev_nonnoun_check
#                 and token[1] not in noun_postags
#                 and token[1][0] != "S"
#             ):
#                 cont_morphs.append(token[0])
#             else:
#                 if len(cont_morphs) > 0:
#                     _tokens.append("".join(cont_morphs))
#                     cont_morphs = []
#                     prev_nonnoun_check = False
#                 if token[1] != "SP":
#                     _tokens.append(token[0])
#             i += 1
#         if len(cont_morphs) > 0:
#             _tokens.append("".join(cont_morphs))
#     else:
#         if len(postags) > 0:
#             _tokens = [
#                 token[0].strip()
#                 for token in _tokens_pos
#                 if token[1] not in stop_postags and token[1] in postags
#             ]
#         else:
#             _tokens = [
#                 token[0].strip()
#                 for token in _tokens_pos
#                 if token[1] not in stop_postags
#             ]

#     if stopwords is not None:
#         _tokens = [token for token in _tokens if not stopwords(token)]
#     return _tokens
