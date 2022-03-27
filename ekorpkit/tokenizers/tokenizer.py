import logging
from abc import ABCMeta, abstractmethod
from ..preprocessors.normalizer import strict_normalize, only_text

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

    @abstractmethod
    def parse(self, text):
        raise NotImplementedError("Must override pos")

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
            if i == 0:
                term_pos.append(f"{term}/{pos}")
            else:
                term_pos.append(f"{self.wordpieces_prefix}{term}/{pos}")
        return term_pos


class PynoriTokenizer(Tokenizer):
    def __init__(
        self,
        args=None,
        **kwargs,
    ):
        logging.warning("Initializing Pynori...")
        try:
            from pynori.korean_analyzer import KoreanAnalyzer

            self._tokenizer = KoreanAnalyzer(**args)
        except ImportError:
            raise ImportError(
                "\n"
                "You must install `pynori` if you want to use `pynori` backend.\n"
                "Please install using `pip install pynori`.\n"
            )
        super().__init__(**kwargs)

    def parse(self, text):

        tokens = self._tokenizer.do_analysis(text)
        term_pos = [
            f"{term}/{pos}" for term, pos in zip(tokens["termAtt"], tokens["posTagAtt"])
        ]
        return term_pos


class MecabTokenizer(Tokenizer):
    def __init__(
        self,
        args=None,
        **kwargs,
    ):

        logging.warning("Initializing mecab...)")
        try:
            from .mecab import MeCab

            self._tokenizer = MeCab(**args)
        except ImportError:
            raise ImportError(
                "\n"
                "You must install `fugashi` and `mecab_ko_dic` if you want to use `mecab` backend.\n"
                "Please install using `pip install python-mecab-ko`.\n"
            )
        super().__init__(**kwargs)

    def parse(self, text):
        return self._tokenizer.pos(text)


class BWPTokenizer(Tokenizer):
    def __init__(
        self,
        args=None,
        **kwargs,
    ):
        logging.warning("Initializing BertWordPieceTokenizer...")
        try:
            from transformers import BertTokenizerFast

            self._tokenizer = BertTokenizerFast.from_pretrained(**args)

        except ImportError:
            raise ImportError(
                "\n"
                "You must install `BertWordPieceTokenizer` if you want to use `bwp` backend.\n"
                "Please install using `pip install transformers`.\n"
            )
        super().__init__(**kwargs)

    def parse(self, text):
        return self._tokenizer.tokenize(text)


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

    df[text_key] = df.apply(extact_tokens_row, axis=1)

    df = df.dropna(subset=[text_key])
    return df
