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


def test_predict_sentiments():
    config_group = "model/sentiment=lm"
    model_cfg = eKonf.compose(config_group=config_group)
    model_cfg.preprocessor.tokenizer.nltk.lemmatize = True

    ds_cfg = eKonf.compose(config_group="dataset=dataset")
    ds_cfg.name = "financial_phrasebank"
    ds_cfg.data_dir = "./data/tmp"

    cfg = eKonf.compose(config_group="pipeline=pipeline")
    cfg.output_dir = "./data/tmp/predict"
    cfg.dataset = ds_cfg
    cfg._pipeline_ = ["subset", "predict"]
    cfg.subset.sample_n = 5
    cfg.predict.model = model_cfg
    cfg.predict.output_file = f"{ds_cfg.name}.parquet"
    cfg.num_workers = 1
    df = eKonf.instantiate(cfg)

    assert len(df) == 15
