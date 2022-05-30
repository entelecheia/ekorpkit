import numpy as np
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.file import load_dataframe


class Lexicon:
    def __init__(self, **args) -> None:
        args = eKonf.to_dict(args)
        self.args = args
        self.name = args["name"]
        self.fullname = args["fullname"]
        self.source = args.get("source", None)
        self.lang = args.get("lang", None)
        self._lexicon_ngram_delim = args.get("ngram_delim")
        self._lexicon_ngram_max_n = args.get("ngram_max_n")
        self._lexicon_features = args["lexicon_features"]
        self._lexicon_path = args["lexicon_path"]
        self._word_key = args.get("word_key", "word")
        self._arr_word_key = "_" + self._word_key
        self._ngram_len_key = "_ngram_len"
        self._token_count_key = "count"
        self._ignore_pos = self.args.get("ignore_pos", False)
        self._postag_length = self.args.get("postag_length")
        if not self._postag_length:
            self._postag_length = 1
        self._postag_delim = self.args.get("postag_delim")
        if not self._postag_delim:
            self._postag_delim = "/"
        self._lexicon = None
        self._analyze_args = args.get("analyze", {})
        self._features = self._analyze_args.get("features", None)
        self._ngram_distiance_tolerance = self._analyze_args.get(
            "ngram_distiance_tolerance", 0
        )
        self._ngram_delim = self._analyze_args.get("ngram_delim")

        self._load_lexicon()

    def _load_lexicon(self, lexicon_path=None):
        lexicon_path = lexicon_path or self._lexicon_path
        df = load_dataframe(lexicon_path)
        df = df.dropna(subset=[self._word_key])
        df[self._word_key] = df[self._word_key].apply(self._prepare_token)
        if self._lexicon_ngram_delim:
            df[self._arr_word_key] = df[self._word_key].str.split(
                self._lexicon_ngram_delim
            )
            df[self._ngram_len_key] = df[self._arr_word_key].apply(len)
            df.sort_values(by=self._ngram_len_key, ascending=False, inplace=True)
            self._lexicon = df[
                [self._word_key, self._arr_word_key, self._ngram_len_key]
                + self._lexicon_features
            ]
        else:
            self._lexicon = df[[self._word_key] + self._lexicon_features]

    def _prepare_token(self, token, ngram_delim=None, strip_pos=None, as_list=False):
        ngram_delim = ngram_delim or self._ngram_delim
        strip_pos = strip_pos or self._ignore_pos
        token = token.lower()

        def _get_word(token):
            token_pos = token.split(self._postag_delim)
            if strip_pos:
                return token_pos[0]
            return (
                token
                if len(token_pos) == 1
                else token_pos[0]
                + self._postag_delim
                + token_pos[1][: self._postag_length]
            )

        if ngram_delim:
            _token = [_get_word(subtoken) for subtoken in token.split(ngram_delim)]
            return _token if as_list else ngram_delim.join(_token)
        else:
            return _get_word(token)

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
            token = self._prepare_token(token, ngram_delim, strip_pos, False)
            if token in token_dict:
                token_dict[token][self._token_count_key] += 1
            else:
                token_dict[token] = {self._token_count_key: 1}
            if ngram_distiance_tolerance > 0 and ngram_delim:
                ngram = self._prepare_token(token, ngram_delim, strip_pos, True)
                if len(ngram) > 1:
                    ngrams_dict[token] = ngram
        return token_dict, ngrams_dict

    @property
    def n_features(self):
        return len(self._lexicon_features)

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
        s += f"fullname: {self.fullname}\n"
        s += f"source: {self.source}\n"
        s += f"lang: {self.lang}\n"
        s += f"lexicon_features: {self._lexicon_features}\n"
        s += f"lexicon_path: {self._lexicon_path}\n"
        s += f"word_key: {self._word_key}\n"
        s += f"ignore_pos: {self._ignore_pos}\n"
        return s

    @staticmethod
    def check_subset_tokens(word, ngrams, ngram_distance_tolerance=0):
        for token, ngram in ngrams.items():
            if (
                set(word).issubset(set(ngram))
                and len(ngram) >= len(word)
                and len(ngram) - len(word) <= ngram_distance_tolerance
            ):
                return token
        return np.nan

    def analyze(
        self,
        tokens: list,
        features=None,
        ngram_distiance_tolerance=None,
        ngram_delim=None,
        **kwargs,
    ):
        features = features or self._features or self._lexicon_features
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
        token_features = df_token[df_token[self._word_key].isin(token_dict.keys())]
        if df_ngram is not None:
            check = df_ngram[self._arr_word_key].apply(
                lambda x: self.check_subset_tokens(
                    x, ngrams_dict, ngram_distiance_tolerance
                )
            )
            ngram_features = df_ngram.loc[list(check.dropna().index), :]
            if not ngram_features.empty:
                ngram_features[self._word_key] = check
                token_features = pd.concat(
                    [token_features, ngram_features], axis=0
                ).drop_duplicates(subset=self._word_key, keep="first")

        feature_dict = token_features.set_index(self._word_key)[features].to_dict(
            "index"
        )

        return_dict = {}
        for token, features in feature_dict.items():
            if not isinstance(token, str):
                continue
            features.update(token_dict[token])
            return_dict[token] = features
        return return_dict
