import logging
from ekorpkit.io.load.list import load_wordlist
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class Stopwords:
    def __init__(self, **kwargs):
        kwargs = eKonf.to_dict(kwargs)

        self.name = kwargs.get("name", "stopwords")
        self._lowercase = kwargs.get("lowercase", False)
        self.verbose = kwargs.get("verbose", False)

        self._stopwords_path = kwargs.get("stopwords_path", None)
        if self._stopwords_path is not None:
            self._stopwords_list = load_wordlist(
                self._stopwords_path, lowercase=self._lowercase, verbose=self.verbose
            )
            if self.verbose:
                log.info(
                    f"Loaded {len(self._stopwords_list)} stopwords from {self._stopwords_path}"
                )
        else:
            self._stopwords_list = []

        self._stopwords_fn = lambda x: False
        stopwords = kwargs.get("stopwords")

        if callable(stopwords):
            self._stopwords_fn = stopwords
            if self.verbose:
                log.info(f"Using custom stopwords function: {str(self._stopwords_fn)}")
        elif isinstance(stopwords, list):
            if self._lowercase:
                stopwords = [w.lower() for w in stopwords]
            if self.verbose:
                log.info(f"Loaded {len(stopwords)} stopwords from list")
            self._stopwords_list += stopwords

        nltk_stopwords = kwargs.get("nltk_stopwords")
        if nltk_stopwords:
            self._stopwords_list += self._load_nltk_stopwords(nltk_stopwords)
            if self.verbose:
                log.info(f"Loaded {len(self._stopwords_list)} stopwords in total")

    def __call__(self, word):
        """Calling a stopwords instance like a function just calls the is_stopword method."""
        return self.is_stopword(word)

    def is_stopword(self, word):
        """
        :type word: str
        :returns: bool
        """
        _word = word.lower() if self._lowercase else word
        return self._stopwords_fn(_word) or _word in self._stopwords_list

    def _load_nltk_stopwords(self, language="english"):
        """
        :type language: str
        :returns: list
        """
        import nltk
        from nltk.corpus import stopwords

        nltk.download("stopwords", quiet=True)
        if language in stopwords.fileids():
            if self.verbose:
                log.info(f"Loading stopwords for {language} from NLTK")
            return stopwords.words(language)

        log.warning(f"No stopwords for {language} in NLTK")
        return []
