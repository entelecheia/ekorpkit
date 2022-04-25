import numpy as np
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.file import load_dataframe


class Lexicon:
    def __init__(self, **args) -> None:
        args = eKonf.to_dict(args)
        self.args = args
        self.name = args["name"]
        self.source = args.get("source", None)
        self.lang = args.get("lang", None)
        self._lexicon_ngram_delim = args.get("ngram_delim", ";")
        self._lexicon_tags = args["lexicon_tags"]
        self._lexicon_path = args["lexicon_path"]
        self._word_key = args.get("word_key", "word")
        self._arr_word_key = "_" + self._word_key
        self._ngram_len_key = "_ngram_len"
        self._token_count_key = "count"
        self._ignore_pos = self.args.get("ignore_pos", False)
        self._lexicon = None
        self._analyze_args = args.get("analyze", {})
        self._tags = self._analyze_args.get("tags", None)
        self._ngram_distiance_tolerance = self._analyze_args.get(
            "ngram_distiance_tolerance", 0
        )
        self._ngram_delim = self._analyze_args.get("ngram_delim", ";")

        self._load_lexicon()

    def _load_lexicon(self, lexicon_path=None):
        lexicon_path = lexicon_path or self._lexicon_path
        df = load_dataframe(lexicon_path)
        df = df.dropna(subset=[self._word_key])
        df[self._word_key] = df[self._word_key].apply(self._prepare_word)
        if self._lexicon_ngram_delim:
            df[self._arr_word_key] = df[self._word_key].str.split(
                self._lexicon_ngram_delim
            )
            df[self._ngram_len_key] = df[self._arr_word_key].apply(len)
            df.sort_values(by=self._ngram_len_key, ascending=False, inplace=True)
            self._lexicon = df[
                [self._word_key, self._arr_word_key, self._ngram_len_key]
                + self._lexicon_tags
            ]
        else:
            self._lexicon = df[[self._word_key] + self._lexicon_tags]

    def _prepare_word(self, word, ngram_delim=None, strip_pos=None, as_list=False):
        ngram_delim = ngram_delim or self._ngram_delim
        strip_pos = strip_pos or self._ignore_pos
        if ngram_delim:
            _word = [
                token.split("/")[0] if strip_pos else token
                for token in word.lower().split(ngram_delim)
            ]
            return _word if as_list else ngram_delim.join(_word)
        else:
            return word.lower().split("/")[0] if strip_pos else word.lower()

    def _prepare_tokens(
        self, tokens, ngram_delim=None, strip_pos=None, ngram_distiance_tolerance=None
    ):
        ngram_delim = ngram_delim or self._ngram_delim
        strip_pos = strip_pos or self._ignore_pos
        ngram_distiance_tolerance = (
            ngram_distiance_tolerance or self._ngram_distiance_tolerance
        )

        token_dict = {}
        ngrams_dict = {}
        for token in tokens:
            token = self._prepare_word(token, ngram_delim, strip_pos, False)
            if token in token_dict:
                token_dict[token][self._token_count_key] += 1
            else:
                token_dict[token] = {self._token_count_key: 1}
            if ngram_distiance_tolerance > 0 and ngram_delim:
                ngram = self._prepare_word(token, ngram_delim, strip_pos, True)
                if len(ngram) > 1:
                    ngrams_dict[token] = ngram
        return token_dict, ngrams_dict

    @property
    def n_tags(self):
        return len(self._lexicon_tags)

    def __len__(self):
        return len(self._lexicon)

    def __getitem__(self, key):
        return self._lexicon.loc[key]

    def __contains__(self, key):
        return key in self._lexicon

    def __iter__(self):
        return iter(self._lexicon)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname}\n----------\n"
        s += f"name: {self.name}\n"
        s += f"source: {self.source}\n"
        s += f"lang: {self.lang}\n"
        s += f"tags: {self._lexicon_tags}\n"
        s += f"lexicon_path: {self._lexicon_path}\n"
        s += f"word_key: {self._word_key}\n"
        s += f"ignore_pos: {self._ignore_pos}\n"
        return s

    @staticmethod
    def check_subset_tokens(word, ngrams, ngram_distiance_tolerance=0):
        for token, ngram in ngrams.items():
            if (
                set(word).issubset(set(ngram))
                and len(ngram) >= len(word)
                and len(ngram) - len(word) <= ngram_distiance_tolerance
            ):
                return token
        return np.nan

    def analyze(
        self,
        tokens: list,
        tags=None,
        ngram_distiance_tolerance=None,
        ngram_delim=None,
        **kwargs,
    ):
        tags = tags or self._lexicon_tags
        ignore_pos = self._ignore_pos
        ngram_distiance_tolerance = (
            ngram_distiance_tolerance or self._ngram_distiance_tolerance
        )
        ngram_delim = ngram_delim or self._ngram_delim

        token_dict, ngrams_dict = self._prepare_tokens(
            tokens, ngram_delim, ignore_pos, ngram_distiance_tolerance
        )
        df = self._lexicon
        if len(ngrams_dict) > 0:
            df_token = df[df[self._ngram_len_key] == 1]
            df_ngram = df[df[self._ngram_len_key] > 1]
        else:
            df_token = df
            df_ngram = None
        senti_tokens = df_token[df_token[self._word_key].isin(token_dict.keys())]
        if df_ngram is not None:
            check = df_ngram._word.apply(
                lambda x: self.check_subset_tokens(
                    x, ngrams_dict, ngram_distiance_tolerance
                )
            )
            senti_ngrams = df_ngram.loc[list(check.dropna().index), :]
            senti_ngrams[self._word_key] = check
            senti_tokens = pd.concat(
                [senti_tokens, senti_ngrams], axis=0
            ).drop_duplicates(subset=self._word_key, keep="first")

        senti_dict = senti_tokens.set_index(self._word_key)[tags].to_dict("index")

        return_dict = {}
        for token, sentiment in senti_dict.items():
            sentiment.update(token_dict[token])
            return_dict[token] = sentiment
        return return_dict
