import os
from ekorpkit import eKonf


def test_setiment_lexicon():
    config_group = "model/sentiment/lexicon=mpko_lex"
    cfg = eKonf.compose(config_group=config_group)
    cfg.ignore_pos = True
    cfg.analyze.ngram_distiance_tolerance = 1
    lexicon = eKonf.instantiate(cfg)

    tokens = ["투기", "억제", "금리", "인상", "인상", "투기;;억제", "금리;인상"]
    sentiments = lexicon.analyze(tokens, tags=["label", "polarity"])
    assert len(sentiments) == 4

    config_group = "model/sentiment/lexicon=lm"
    cfg = eKonf.compose(config_group=config_group)
    lexicon = eKonf.instantiate(cfg)

    tokens = ["Bad", "Fraud", "Good", "Sound", "uncertain", "beat", "wrong"]
    sentimetns = lexicon.analyze(tokens, tags=["Positive", "Negative", "Uncertainty"])
    assert len(sentimetns) == len(tokens)
