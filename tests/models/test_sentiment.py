import os
import pytest
from ekorpkit import eKonf


def test_setiment_lexicon():
    ngram_cfg = eKonf.compose(config_group="model/ngram=mpko_lex")
    ngram_cfg.verbose = True
    ngram_cfg.auto.load = True
    ngram = eKonf.instantiate(ngram_cfg)

    sentence = "투기를 억제하기 위해 금리를 인상해야 한다."
    tokens = ngram.ngramize_sentence(sentence)
    assert len(tokens) == 4

    ngram_cfg = eKonf.compose(config_group="model/ngram=lm")
    ngram_cfg.verbose = True
    ngram_cfg.auto.load = True
    ngram = eKonf.instantiate(ngram_cfg)

    sentence = "Beyond the improved voice capabilities, customers now have a streamlined way to comply with recalls and other traceability requirements, providing them with a competitive advantage."
    _features = ngram.find_features(sentence)
    assert len(_features) == 22


def test_predict_sentiments():
    config_group = "model/sentiment=lm"
    model_cfg = eKonf.compose(config_group=config_group)

    ds_cfg = eKonf.compose(config_group="dataset")
    ds_cfg.verbose = True
    ds_cfg.name = "financial_phrasebank"
    ds_cfg.path.cache.uri = "https://github.com/entelecheia/ekorpkit-book/raw/main/data/financial_phrasebank.zip"
    ds_cfg.data_dir = ds_cfg.path.cached_path
    ds_cfg.use_name_as_subdir = True

    cfg = eKonf.compose(config_group="pipeline")
    # cfg.output_dir = "./data/tmp/predict"
    cfg.data.dataset = ds_cfg
    cfg._pipeline_ = ["subset", "predict"]
    cfg.subset.sample_frac = 0.5
    cfg.predict.model = model_cfg
    cfg.predict.output_file = f"{ds_cfg.name}.parquet"
    cfg.num_workers = 1
    df = eKonf.instantiate(cfg)

    assert len(df) > 0


def test_eval_sentiments():
    eval_cfg = eKonf.compose(config_group="model/eval=classification")
    eval_cfg.columns.actual = 'labels'
    eval_cfg.columns.predicted = 'polarity_label'
    eval_cfg.labels = ['positive', 'negative', 'neutral']
    # eval_cfg.data_dir = "./data/tmp/predict"
    eval_cfg.data_file = "financial_phrasebank*.parquet"
    # eval_cfg.output_dir = "./data/tmp/eval"
    eKonf.instantiate(eval_cfg)

    assert os.path.exists(eval_cfg.output_dir)


@pytest.mark.skip(reason=".")
def test_eval_fomc_sentiments():
    cfg = eKonf.compose(config_group="corpus")
    cfg.name = "fomc"
    cfg.path.cache.uri = (
        "https://github.com/entelecheia/ekorpkit-book/raw/main/data/fomc.zip"
    )
    cfg.data_dir = cfg.path.cached_path
    cfg.auto.merge = True
    fomc = eKonf.instantiate(cfg)

    fomc_statements = fomc.data[fomc.data.content_type == "fomc_statement"]
    fomc_statements.set_index(eKonf.Keys.TIMESTAMP, inplace=True)

    assert True

    config_group = "model/sentiment=lm"
    model_cfg = eKonf.compose(config_group=config_group)
    model_cfg.preprocessor.tokenizer.nltk.lemmatize = True

    cfg = eKonf.compose(config_group="pipeline/predict")
    cfg.name = "fomc_sentiments"
    cfg.model = model_cfg
    # cfg.output_dir = "./data/tmp/predict"
    cfg.output_file = f"{cfg.name}-lm.parquet"
    cfg.num_workers = 100
    fomc_sentiments = eKonf.pipe(fomc_statements, cfg)

    assert True
