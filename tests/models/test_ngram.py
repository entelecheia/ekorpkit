import os
import pytest
from ekorpkit import eKonf


def test_ngram_trainer():
    corpus_cfg = eKonf.compose("corpus")
    corpus_cfg.path.cache.uri = (
        "https://github.com/entelecheia/ekorpkit-book/raw/main/assets/data/bok_minutes.zip"
    )
    corpus_cfg.data_dir = corpus_cfg.path.cached_path
    corpus_cfg.name = "bok_minutes"
    corpus_cfg.auto.merge = True

    ngram_cfg = eKonf.compose("model/ngram=npmi")
    ngram_cfg.data.corpus = corpus_cfg
    ngram_cfg.verbose = False
    ngram_cfg.auto.load = True
    ngram_cfg.force.train = True
    ngram_cfg.candidates.threshold = 0.1
    ngram = eKonf.instantiate(ngram_cfg)

    _ngrams = ngram.find_ngrams(
        ngram.sentences[0],
        strip_pos=False,
        surface_delim=";",
        threshold=0.1,
        apply_postag_rules=False,
    )
    assert len(_ngrams) > 0
