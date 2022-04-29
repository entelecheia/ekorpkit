import logging
import unicodedata
import re
from ftfy import TextFixerConfig, fix_text
from .hanja import translate as hanja2hangle
from .hangle import compose, decompose


log = logging.getLogger(__name__)


doublespace_pattern = re.compile(r"\s+")
repeatchars_pattern = re.compile(r"(\w)\\1{2,}")
number_pattern = re.compile(r"[0-9]")
punctuation_pattern = re.compile(r"[,\.\?\!]")
symbol_pattern = re.compile(r"[()\[\]\{\}`]")
hangle_pattern = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]")
alphabet_pattern = re.compile(r"[a-zA-Z]")

hangle_filter = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣]")
hangle_number_filter = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9]")
text_filter = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9,\.\?\!&·\"'\(\)\[\]\{\}+\-\\\/\*×%]")
# text_filter = re.compile(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9,\.\?\!&·\"\'-()\[\]\{\}]')
text_filter_for_lrgraph = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9,\.\?\!&\"'-()\[\]\{\}]")


#: Control characters.
CONTROLS = {
    "\u0001",
    "\u0002",
    "\u0003",
    "\u0004",
    "\u0005",
    "\u0006",
    "\u0007",
    "\u0008",
    "\u000e",
    "\u000f",
    "\u0011",
    "\u0012",
    "\u0013",
    "\u0014",
    "\u0015",
    "\u0016",
    "\u0017",
    "\u0018",
    "\u0019",
    "\u001a",
    "\u001b",
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    "-",  # \u002d Hyphen-minus
    "‐",  # \u2010 Hyphen
    "‑",  # \u2011 Non-breaking hyphen
    "⁃",  # \u2043 Hyphen bullet
    "‒",  # \u2012 figure dash
    "–",  # \u2013 en dash
    "—",  # \u2014 em dash
    "―",  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    "-",  # \u002d Hyphen-minus
    "−",  # \u2212 Minus
    "－",  # \uff0d Full-width Hyphen-minus
    "⁻",  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    "+",  # \u002b Plus
    "＋",  # \uff0b Full-width Plus
    "⁺",  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    "/",  # \u002f Solidus
    "⁄",  # \u2044 Fraction slash
    "∕",  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    "~",  # \u007e Tilde
    "˜",  # \u02dc Small tilde
    "⁓",  # \u2053 Swung dash
    "∼",  # \u223c Tilde operator
    "∽",  # \u223d Reversed tilde
    "∿",  # \u223f Sine wave
    "〜",  # \u301c Wave dash
    "～",  # \uff5e Full-width tilde
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    "’",  # \u2019
    "՚",  # \u055a
    "Ꞌ",  # \ua78b
    "ꞌ",  # \ua78c
    "＇",  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    "‘",  # \u2018
    "’",  # \u2019
    "‚",  # \u201a
    "‛",  # \u201b
}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    "“",  # \u201c
    "”",  # \u201d
    "„",  # \u201e
    "‟",  # \u201f
}

#: Accent characters.
ACCENTS = {
    "`",  # \u0060
    "´",  # \u00b4
}

#: Prime characters.
PRIMES = {
    "′",  # \u2032
    "″",  # \u2033
    "‴",  # \u2034
    "‵",  # \u2035
    "‶",  # \u2036
    "‷",  # \u2037
    "⁗",  # \u2057
}

#: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES

#: Uppercase and lowercase greek letters.
GREEK = {
    "Α",  # \u0391
    "Β",  # \u0392
    "Γ",  # \u0393
    "Δ",  # \u0394
    "Ε",  # \u0395
    "Ζ",  # \u0396
    "Η",  # \u0397
    "Θ",  # \u0398
    "Ι",  # \u0399
    "Κ",  # \u039a
    "Λ",  # \u039b
    "Μ",  # \u039c
    "Ν",  # \u039d
    "Ξ",  # \u039e
    "Ο",  # \u039f
    "Π",  # \u03a0
    "Ρ",  # \u03a1
    "Σ",  # \u03a3
    "Τ",  # \u03a4
    "Υ",  # \u03a5
    "Φ",  # \u03a6
    "Χ",  # \u03a7
    "Ψ",  # \u03a8
    "Ω",  # \u03a9
    "α",  # \u03b1
    "β",  # \u03b2
    "γ",  # \u03b3
    "δ",  # \u03b4
    "ε",  # \u03b5
    "ζ",  # \u03b6
    "η",  # \u03b7
    "θ",  # \u03b8
    "ι",  # \u03b9
    "κ",  # \u03ba
    "λ",  # \u03bb
    "μ",  # \u03bc
    "ν",  # \u03bd
    "ξ",  # \u03be
    "ο",  # \u03bf
    "π",  # \u03c0
    "ρ",  # \u03c1
    "σ",  # \u03c3
    "τ",  # \u03c4
    "υ",  # \u03c5
    "φ",  # \u03c6
    "χ",  # \u03c7
    "ψ",  # \u03c8
    "ω",  # \u03c9
}

#: Names of greek letters spelled out as words.
GREEK_WORDS = {
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Xi",
    "Omicron",
    "Pi",
    "Rho",
    "Sigma",
    "Tau",
    "Upsilon",
    "Phi",
    "Chi",
    "Psi",
    "Omega",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lamda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
}

#: Words that should not be capitalized in titles.
SMALL = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "en",
    "for",
    "if",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "v",
    "v",
    "via",
    "vs",
    "vs",
}

#: Words that should not be capitalized in names.
NAME_SMALL = {
    "abu",
    "bon",
    "bin",
    "da",
    "dal",
    "de",
    "del",
    "der",
    "de",
    "di",
    u"dí",
    "ibn",
    "la",
    "le",
    "san",
    "st",
    "ste",
    "van",
    "vel",
    "von",
    "y",
}

# This isn't every possible TLD, just the most common, to avoid false positives.
TLDS = {
    "aero",
    "asia",
    "biz",
    "cat",
    "com",
    "coop",
    "edu",
    "eu",
    "gov",
    "info",
    "int",
    "jobs",
    "mil",
    "mobi",
    "museum",
    "name",
    "net",
    "org",
    "pro",
    "tel",
    "travel",
    "xxx",
    "ad",
    "as",
    "ar",
    "au",
    "br",
    "bz",
    "ca",
    "cc",
    "cd",
    "co",
    "ch",
    "cn",
    "de",
    "dj",
    "es",
    "fr",
    "fm",
    "it",
    "io",
    "jp",
    "la",
    "ly",
    "me",
    "ms",
    "nl",
    "no",
    "nu",
    "ru",
    "sc",
    "se",
    "sr",
    "su",
    "tk",
    "tv",
    "uk",
    "us",
    "ws",
}

#: A variety of numbers, spelled out as words.
NUMBERS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
}
LEFT_PARENTHESES = {"(", "[", "{", "&lt;"}
RIGHT_PARENTHESES = {")", "]", "}", "&gt;"}
#: Regular expression that matches email addresses.
EMAIL_RE = re.compile(r"([\w\-\.\+%]+@(\w[\w\-]+\.)+[\w\-]+)", re.I | re.U)
#: Regular expression that matches DOIs.
DOI_RE = re.compile(r"^10\.\d{4,9}/[-\._;()/:A-Z0-9]+$", re.U)
#: Regular expression that matches ISSNs.
ISSN_RE = re.compile(r"^\d{4}-\d{3}[\dX]$", re.U)
#: Regular expression that matches control characters not allowed in XML.
CONTROL_RE = re.compile(
    "[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\u10000-\u10FFFF]+"
)


# def get_encoding(input_string, guesses=None, is_html=False):
#     """Return the encoding of a byte string. Uses bs4 UnicodeDammit.
#     :param string input_string: Encoded byte string.
#     :param list[string] guesses: (Optional) List of encoding guesses to prioritize.
#     :param bool is_html: Whether the input is HTML.
#     """
#     converted = UnicodeDammit(input_string, override_encodings=[guesses] if guesses else [], is_html=is_html)
#     return converted.original_encoding


def levenshtein(s1, s2, allow_substring=False):
    """Return the Levenshtein distance between two strings.
    The Levenshtein distance (a.k.a "edit difference") is the number of characters that need to be substituted,
    inserted or deleted to transform s1 into s2.
    Setting the `allow_substring` parameter to True allows s1 to be a
    substring of s2, so that, for example, "hello" and "hello there" would have a distance of zero.
    :param string s1: The first string
    :param string s2: The second string
    :param bool allow_substring: Whether to allow s1 to be a substring of s2
    :returns: Levenshtein distance.
    :rtype int
    """
    len1, len2 = len(s1), len(s2)
    lev = []
    for i in range(len1 + 1):
        lev.append([0] * (len2 + 1))
    for i in range(len1 + 1):
        lev[i][0] = i
    for j in range(len2 + 1):
        lev[0][j] = 0 if allow_substring else j
    for i in range(len1):
        for j in range(len2):
            lev[i + 1][j + 1] = min(
                lev[i][j + 1] + 1, lev[i + 1][j] + 1, lev[i][j] + (s1[i] != s2[j])
            )
    return min(lev[len1]) if allow_substring else lev[len1][len2]


def bracket_level(text, open={"(", "[", "{"}, close={")", "]", "}"}):
    """Return 0 if string contains balanced brackets or no brackets."""
    level = 0
    for c in text:
        if c in open:
            level += 1
        elif c in close:
            level -= 1
    return level


def is_punct(text):
    for char in text:
        if not unicodedata.category(char).startswith("P"):
            return False
    else:
        return True


def is_ascii(text):
    for char in text:
        if ord(char) >= 128:
            return False
    else:
        return True


def like_url(text):
    if len(text) < 1:
        return False
    if text.startswith("http://"):
        return True
    elif text.startswith("www.") and len(text) >= 5:
        return True
    if len(text) < 2 or text[0] == "." or text[-1] == "." or "." not in text:
        return False
    tld = text.rsplit(".", 1)[1].split(":", 1)[0]
    if tld.endswith("/"):
        return True
    if tld.isalpha() and tld in TLDS:
        return True
    return False


def like_number(text):
    text = text.replace(",", "").replace(".", "")
    if text.isdigit():
        return True
    if text.count("/") == 1:
        num, denom = text.split("/")
        if like_number(num) and like_number(denom):
            return True
    if text in NUMBERS:
        return True
    return False


def word_shape(text):
    prev_m = ""
    seq = 0
    shape = []
    for c in text:
        if c.isdigit():
            m = "d"  # Digits
        elif c in GREEK:
            m = "g"  # Greek letters
        elif c.isalpha():
            m = "X" if c.isupper() else "x"  # Uppercase or lowercase alphabetical
        elif c in QUOTES:
            m = "'"  # Quotes and apostrophes
        elif c in {":", ";"}:
            m = ":"  # Colons and semicolons
        elif c in {"!", "?", "."}:
            m = "."  # Sentence ends
        elif c in {"(", "[", "{", ")", "]", "}"}:
            m = "b"  # Brackets
        elif c in {"°", "%"}:
            m = "u"  # units
        elif c in {"■", "◼", "●", "▲", "○", "◆", "▼", "⧫", "△", "◇", "▽", "⬚", "□"}:
            m = "l"  # list markers
        elif c in {",", "$", "&", "-"}:
            m = c  # Stay the same
        else:
            m = "*"  # Everything else, symbols etc: {'=', '+', '*', '_', '|', '@', '×', '÷', '±', '<', '≤', '>', '≥', '≦', '≡', '≅', '≈', '≃', '≲', '→', '←', '⇄', '≪', '≫', '↔', '≠', '∝', '∈', '⇌', '⇋', '⋯', '~', '·', '•', '√', '⊃', '∑', '∏', '®', '∞', '∂', '∫', '∇', '∧', '⟨', '⟩'}
        if m == prev_m:
            seq += 1
        else:
            seq = 0
            prev_m = m
        if seq < 3:
            shape.append(m)
    return "".join(shape)


def remove_doublespace(sent):
    return doublespace_pattern.sub(" ", sent)


def repeat_normalize(sent, num_repeats=2):
    if num_repeats > 0:
        sent = repeatchars_pattern.sub("\\1" * num_repeats, sent)
    sent = doublespace_pattern.sub(" ", sent)
    return sent.strip()


def emoticon_normalize(sent, num_repeats=2):
    if not sent:
        return sent

    # Pattern matching ㅋ쿠ㅜ
    def pattern(idx):
        # Jaum: 0, Moum: 1, Complete: 2, else -1
        if 12593 <= idx <= 12622:
            return 0
        elif 12623 <= idx <= 12643:
            return 1
        elif 44032 <= idx <= 55203:
            return 2
        else:
            return -1

    idxs = [pattern(ord(c)) for c in sent]
    sent_ = []
    last_idx = len(idxs) - 1
    for i, (idx, c) in enumerate(zip(idxs, sent)):
        if (i > 0 and i < last_idx) and (
            idxs[i - 1] == 0 and idx == 2 and idxs[i + 1] == 1
        ):
            cho, jung, jong = decompose(c)
            if (cho == sent[i - 1]) and (jung == sent[i + 1]) and (jong == " "):
                sent_.append(cho)
                sent_.append(jung)
            else:
                sent_.append(c)
        elif (i < last_idx) and (idx == 2) and (idxs[i + 1] == 0):
            cho, jung, jong = decompose(c)
            if jong == sent[i + 1]:
                sent_.append(compose(cho, jung, " "))
                sent_.append(jong)
        elif (i > 0) and (idx == 2 and idxs[i - 1] == 0):
            cho, jung, jong = decompose(c)
            if cho == sent[i - 1]:
                sent_.append(cho)
                sent_.append(jung)
        else:
            sent_.append(c)
    return repeat_normalize("".join(sent_), num_repeats)


def hanja_to_hangle(sent):
    return hanja2hangle(sent, "substitution")


def only_hangle(sent):
    sent = hanja_to_hangle(sent)
    return doublespace_pattern.sub(" ", hangle_filter.sub(" ", sent)).strip()


def only_hangle_number(sent):
    sent = hanja_to_hangle(sent)
    return doublespace_pattern.sub(" ", hangle_number_filter.sub(" ", sent)).strip()


def only_text(sent):
    sent = hanja_to_hangle(sent)
    return doublespace_pattern.sub(" ", text_filter.sub(" ", sent)).strip()


def remain_hangle_on_last(eojeol):
    matchs = list(hangle_pattern.finditer(eojeol))
    if not matchs:
        return ""
    last_index = matchs[-1].span()[1]
    return eojeol[:last_index].strip()


def normalize_sent_for_lrgraph(sent):
    sent = hanja_to_hangle(sent)
    sent = text_filter_for_lrgraph.sub(" ", sent)
    sent = symbol_pattern.sub(" ", sent)
    sent_ = [remain_hangle_on_last(eojeol) for eojeol in sent.split()]
    sent_ = [eojeol for eojeol in sent_ if eojeol]
    if not sent_:
        return ""
    return " ".join(sent_)


class BaseNormalizer:
    """Abstract normalizer class from which all normalizers inherit.
    Subclasses must implement a ``normalize()`` method.
    """

    def normalize(self, text):
        """Normalize the text.
        :param string text: The text to normalize.
        :returns: Normalized text.
        :rtype: string
        """
        return text

    def __call__(self, text):
        """Calling a normalizer instance like a function just calls the normalize method."""
        return self.normalize(text)


class Normalizer(BaseNormalizer):
    """Main Normalizer class for generic English text.
    Normalize unicode, hyphens, quotes, whitespace.
    By default, the normal form NFKC is used for unicode normalization. This applies a compatibility decomposition,
    under which equivalent characters are unified, followed by a canonical composition. See Python docs for information
    on normal forms: http://docs.python.org/2/library/unicodedata.html#unicodedata.normalize
    """

    def __init__(
        self,
        ftfy={},
        spaces={},
        special_characters={},
        hanja2hangle=True,
        num_repeats=2,
        **kwargs
    ):
        """
        :param string form: Normal form for unicode normalization.
        :param bool strip: Whether to strip whitespace from start and end.
        :param bool hyphens: Whether to normalize all hyphens, minuses and dashes to the ASCII hyphen-minus character.
        :param bool quotes: Whether to normalize all apostrophes, quotes and primes to the ASCII quote character.
        :param bool ellipsis: Whether to normalize ellipses to three full stops.
        :param bool slashes: Whether to normalize slash characters to the ASCII slash character.
        :param bool tildes: Whether to normalize tilde characters to the ASCII tilde character.
        """
        self.ftfy = ftfy or {}
        self.special_characters = special_characters or {}
        self.spaces = spaces or {}

        self._uncurl_quotes = self.ftfy.get("uncurl_quotes", True)
        self._remove_control_chars = self.ftfy.get("remove_control_chars", True)

        self._strip = self.spaces.get("strip", True)
        self._fix_whitespaces = self.spaces.get("fix_whitespaces", True)
        self._collapse_whitespaces = self.spaces.get("collapse_whitespaces", True)
        self._replace_tabs = self.spaces.get("replace_tabs", True)
        self._replacement_spaces = " " * self.spaces.get("replacement_spaces", 4)

        self._fix_hyphens = self.special_characters.get("fix_hyphens", True)
        self._fix_ellipsis = self.special_characters.get("fix_ellipsis", True)
        self._fix_slashes = self.special_characters.get("fix_slashes", True)
        self._fix_tildes = self.special_characters.get("fix_tildes", True)
        self._fix_emoticons = self.special_characters.get("fix_emoticons", False)
        self._single_quotes_only = self.special_characters.get(
            "single_quotes_only", False
        )
        self._regular_parentheses_only = self.special_characters.get(
            "regular_parentheses_only", False
        )

        self._hanja2hangle = hanja2hangle
        self._num_repeats = num_repeats

        self._ftfy_cfg = TextFixerConfig(**self.ftfy)

    def fix_text(self, text):
        if text is None:
            return None
        return fix_text(str(text), self._ftfy_cfg)

    def remove_control_chars(self, text):
        """
        Strip out any control characters (they occasionally creep in somehow)
        """
        if text is None:
            return None
        for control in CONTROLS:
            text = text.replace(control, "")
        return text

    def fix_hyphens(self, text):
        """
        Normalize all hyphens, minuses and dashes to ascii hyphen-minus and remove soft hyphen entirely
        """
        # TODO: Better normalization of em/en dashes to '--' if surrounded by spaces or start/end?
        if text is None:
            return None
        for hyphen in HYPHENS | MINUSES:
            text = text.replace(hyphen, "-")
        text = text.replace("\u00ad", "")
        return text

    def uncurl_quotes(self, text):
        """
        Normalize all quotes and primes to ascii apostrophe and quotation mark
        """
        if text is None:
            return None
        for double_quote in DOUBLE_QUOTES:
            text = text.replace(double_quote, '"')  # \u0022
        for single_quote in SINGLE_QUOTES | APOSTROPHES | ACCENTS:
            text = text.replace(single_quote, "'")  # \u0027
        text = text.replace("′", "'")  # \u2032 prime
        text = text.replace("‵", "'")  # \u2035 reversed prime
        text = text.replace("″", "''")  # \u2033 double prime
        text = text.replace("‶", "''")  # \u2036 reversed double prime
        text = text.replace("‴", "'''")  # \u2034 triple prime
        text = text.replace("‷", "'''")  # \u2037 reversed triple prime
        text = text.replace("⁗", "''''")  # \u2057 quadruple prime
        return text

    def fix_ellipsis(self, text):
        """
        Normalize ellipses to three full stops
        """
        if text is None:
            return None
        text = text.replace("…", "...").replace(" . . . ", " ... ")  # \u2026
        return text

    def fix_slashes(self, text):
        """
        Normalize slash characters to ascii slash
        """
        if text is None:
            return None
        for slash in SLASHES:
            text = text.replace(slash, "/")
        return text

    def fix_tildes(self, text):
        """
        Normalize tilde characters to ascii tilde
        """
        if text is None:
            return None
        for tilde in TILDES:
            text = text.replace(tilde, "~")
        return text

    def replace_tabs(self, text, replacement_spaces=" " * 4):
        """
        Replace tabs with spaces
        """
        if text is None:
            return None
        text = text.replace("\t", replacement_spaces)
        return text

    def fix_whitespaces(self, text):
        """
        Normalize unusual whitespace not caught by unicodedata
        """
        if text is None:
            return None
        text = (
            text.replace("\u000b", " ").replace("\u000c", " ").replace(u"\u0085", " ")
        )
        return text

    def collapse_whitespaces(self, text):
        """
        Collapse all whitespace to a single space
        """
        if text is None:
            return None

        text = re.sub(r" +", " ", text)
        return text

    def single_quotes_only(self, text):
        """
        Replace all double quotes with single quotes
        """
        if text is None:
            return None
        for quote in QUOTES:
            text = text.replace(quote, "'")
        return text

    # Convert all brackets to regular parentheses
    def regular_parentheses_only(self, text):
        """
        Replace all curly brackets with regular parentheses
        """
        if text is None:
            return None
        for ob in LEFT_PARENTHESES:
            text = text.replace(ob, "(")
        for cb in RIGHT_PARENTHESES:
            text = text.replace(cb, ")")
        return text

    def hanja2hangle(self, text):
        """
        Convert all hanja to hangle
        """
        if text is None:
            return None
        text = hanja_to_hangle(text)
        return text

    def fix_emoticons(self, text, num_repeats=2):
        """
        Replace emoticons with their text equivalents
        """
        if text is None:
            return None
        text = emoticon_normalize(text, num_repeats=num_repeats)
        return text

    def normalize(self, text):
        """Run the Normalizer on a string.
        :param text: The string to normalize.
        """
        if text is None:
            return None
        text = self.fix_text(text)

        # Normalize to canonical unicode (using NFKC by default)
        # if self.form is not None:
        #     text = unicodedata.normalize(self.form, text)

        if self._remove_control_chars:
            text = self.remove_control_chars(text)

        # if self.fix_line_breaks:
        #     text = text.replace('\u2028', '\n').replace('\u2029', '\n').replace('\r\n', '\n').replace('\r', '\n')

        if self._fix_hyphens:
            text = self.fix_hyphens(text)

        if self._uncurl_quotes:
            text = self.uncurl_quotes(text)

        if self._fix_ellipsis:
            text = self.fix_ellipsis(text)

        if self._fix_slashes:
            text = self.fix_slashes(text)

        if self._fix_tildes:
            text = self.fix_tildes(text)

        if self._replace_tabs:
            text = self.replace_tabs(text, replacement_spaces=self._replacement_spaces)

        if self._fix_whitespaces:
            text = self.fix_whitespaces(text)

        if self._collapse_whitespaces:
            text = self.collapse_whitespaces(text)

        if self._strip:
            text = text.strip()

        if self._single_quotes_only:
            text = self.single_quotes_only(text)

        if self._regular_parentheses_only:
            text = self.regular_parentheses_only(text)

        if self._hanja2hangle:
            text = self.hanja2hangle(text)

        if self._fix_emoticons:
            text = self.emoticon_normalize(text, num_repeats=self._num_repeats)

        return text


#: Default normalize that canonicalizes unicode and fixes whitespace.
base_normalize = Normalizer(
    strip=True,
    fix_whitespaces=False,
    fix_hyphens=False,
    fix_ellipsis=False,
)
#: More aggressive normalize that also standardizes hyphens, and quotes.
strict_normalize = Normalizer(
    strip=True,
    fix_whitespaces=False,
    fix_hyphens=True,
    fix_ellipsis=True,
    fix_tildes=True,
)


# class ExcessNormalizer(Normalizer):
#     """Excessive string normalization.
#     This is useful when doing fuzzy string comparisons. A common use case is to run this before calculating the
#     Levenshtein distance between two strings, so that only "important" differences are counted.
#     """

#     def __init__(self, form='NFKC', strip=True, collapse=False, hyphens=True, quotes=True, ellipsis=True, tildes=True, lower=True):
#         """"""
#         super(ExcessNormalizer, self).__init__(form, strip=strip, fix_whitespaces=collapse, fix_hyphens=hyphens, uncurl_quotes=quotes,
#                                             fix_ellipsis=ellipsis, fix_tildes=tildes)
#         self.lower = lower

#     def normalize(self, text):
#         # Lowercase and normalize unicode
#         text = super(ExcessNormalizer, self).normalize(text.lower() if self.lower else text)
#         # Remove all whitespace
#         # text = ''.join(text.split())
#         # Convert all apostrophes, quotes, accents, primes to single ascii apostrophe
#         for quote in QUOTES:
#             text = text.replace(quote, "'")
#         # Convert all brackets to regular parentheses
#         for ob in {'(', '<', '[', '{', '&lt;'}:
#             text = text.replace(ob, '(')
#         for cb in {')', '>', ']', '}', '&gt;'}:
#             text = text.replace(cb, ')')
#         return text
