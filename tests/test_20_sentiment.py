import os
import pytest
from ekorpkit import eKonf


def test_setiment_lexicon():
    config_group = "model/sentiment/lexicon=mpko_lex"
    cfg = eKonf.compose(config_group=config_group)
    cfg.verbose = True
    cfg.ignore_pos = True
    cfg.analyze.ngram_distiance_tolerance = 1
    lexicon = eKonf.instantiate(cfg)

    tokens = ["투기", "억제", "금리", "인상", "인상", "투기;;억제", "금리;인상"]
    sentiments = lexicon.analyze(tokens, tags=["label", "polarity"])
    assert len(sentiments) == 4

    config_group = "model/sentiment/lexicon=lm"
    cfg = eKonf.compose(config_group=config_group)
    cfg.verbose = True
    lexicon = eKonf.instantiate(cfg)

    tokens = ["Bad", "Fraud", "Good", "Sound", "uncertain", "beat", "wrong"]
    sentimetns = lexicon.analyze(tokens, tags=["Positive", "Negative", "Uncertainty"])
    assert len(sentimetns) == len(tokens)


def test_predict_sentiments():
    config_group = "model/sentiment=lm"
    model_cfg = eKonf.compose(config_group=config_group)
    model_cfg.verbose = True
    model_cfg.preprocessor.tokenizer.nltk.lemmatize = True

    ds_cfg = eKonf.compose(config_group="dataset=dataset")
    ds_cfg.verbose = True
    ds_cfg.name = "financial_phrasebank"
    ds_cfg.data_dir = "${cached_path:'https://github.com/entelecheia/ekorpkit-config/raw/main/data/financial_phrasebank.zip',true,false}"
    ds_cfg.use_name_as_subdir = True

    cfg = eKonf.compose(config_group="pipeline")
    cfg.verbose = True
    cfg.output_dir = "./data/tmp/predict"
    cfg.dataset = ds_cfg
    cfg._pipeline_ = ["subset", "predict"]
    cfg.subset.sample_frac = 0.5
    cfg.predict.model = model_cfg
    cfg.predict.output_file = f"{ds_cfg.name}.parquet"
    cfg.num_workers = 1
    df = eKonf.instantiate(cfg)

    assert len(df) > 0


def test_eval_sentiments():
    eval_cfg = eKonf.compose(config_group="model/eval=classification")
    eval_cfg.verbose = True
    eval_cfg.to_eval.actual = "labels"
    eval_cfg.to_eval.predicted = "polarity_label"
    eval_cfg.data_dir = "./data/tmp/predict"
    eval_cfg.data_file = "financial_phrasebank*.parquet"
    eval_cfg.output_dir = "./data/tmp/eval"
    eKonf.instantiate(eval_cfg)

    assert os.path.exists(eval_cfg.output_dir)


@pytest.mark.skip(reason=".")
def test_eval_fomc_sentiments():
    cfg = eKonf.compose(config_group="corpus")
    cfg.name = "fomc"
    cfg.data_dir = "${cached_path:'https://github.com/entelecheia/ekorpkit-config/raw/main/data/fomc.zip',true,false}"
    cfg.automerge = True
    fomc = eKonf.instantiate(cfg)

    fomc_statements = fomc.data[fomc.data.content_type == "fomc_statement"]
    fomc_statements.set_index("timestamp", inplace=True)

    assert True 

    config_group = "model/sentiment=lm"
    model_cfg = eKonf.compose(config_group=config_group)
    model_cfg.preprocessor.tokenizer.nltk.lemmatize = True

    cfg = eKonf.compose(config_group="pipeline/predict")
    cfg.name = "fomc_sentiments"
    cfg.model = model_cfg
    cfg.output_dir = "./data/tmp/predict"
    cfg.output_file = f"{cfg.name}-lm.parquet"
    cfg.num_workers = 100
    fomc_sentiments = eKonf.pipe(cfg, fomc_statements)

    assert True 

    cfg = eKonf.compose(config_group="visualize/plot=lineplot")
    cfg.dataset.y = "num_tokens"
    cfg.plot.figsize = (15, 8)
    cfg.figure.title = "The number of words in the FOMC statements"
    cfg.figure.legend = None
    cfg.output_dir = "./data/tmp/visualize"
    eKonf.instantiate(cfg, data=fomc_sentiments)

    assert os.path.exists(cfg.output_dir)
