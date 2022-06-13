import os
from ekorpkit import eKonf


def test_dummy_corpus():
    cfg = eKonf.compose(config_group="io/fetcher=_dummy")
    cfg.verbose = True
    cfg.name = "fomc_minutes"
    eKonf.instantiate(cfg)
    output_file = cfg.output_file
    assert os.path.exists(output_file)
    os.remove(output_file)
    assert not os.path.exists(output_file)


def test_build_corpora():
    cfg = eKonf.compose(config_group="corpus/builtin=_dummy_fomc_minutes")
    cfg.verbose = True
    cfg.data_dir = "./data/tmp/fomc_minutes"
    db = eKonf.instantiate(cfg)
    db.build()

    cfg = eKonf.compose(config_group="corpus/builtin=_dummy_bok_minutes")
    cfg.verbose = True
    cfg.data_dir = "./data/tmp/bok_minutes"
    db = eKonf.instantiate(cfg)
    db.build()

    cfg = eKonf.compose(config_group="corpus=corpora")
    cfg.verbose = True
    cfg.name = ["bok_minutes", "fomc_minutes"]
    cfg.data_dir = "./data/tmp"
    crps = eKonf.instantiate(cfg)
    # crps.concat_corpora()

    assert len(crps.corpora) == 2


def test_corpus_task():
    corpus_cfg = eKonf.compose(config_group="corpus")
    corpus_cfg.verbose = True
    corpus_cfg.name = "bok_minutes"
    corpus_cfg.auto.merge = True
    corpus_cfg.data_dir = "./data/tmp"

    cfg = eKonf.compose(config_group="pipeline")
    cfg.verbose = True
    cfg.data.corpus = corpus_cfg
    cfg._pipeline_ = ["filter_query", "save_dataframe"]
    cfg.filter_query.query = "filename in ['BOK_20181130_20181218']"
    cfg.save_dataframe.output_dir = "./data/tmp"
    cfg.save_dataframe.output_file = "corpus_filtered.parquet"
    eKonf.instantiate(cfg)

    assert os.path.exists("./data/tmp/corpus_filtered.parquet")


def test_corpora_task():
    corpus_cfg = eKonf.compose(config_group="corpus=corpora")
    corpus_cfg.verbose = True
    corpus_cfg.name = ["bok_minutes", "fomc_minutes"]
    corpus_cfg.auto.merge = True
    corpus_cfg.data_dir = "./data/tmp"

    cfg = eKonf.compose(config_group="pipeline")
    cfg.verbose = True
    cfg.data.corpus = corpus_cfg
    cfg._pipeline_ = ["filter_query", "save_dataframe"]
    cfg.filter_query.query = "id == 0"
    cfg.save_dataframe.output_dir = "./data/tmp"
    cfg.save_dataframe.output_file = "corpora_filtered.parquet"
    eKonf.instantiate(cfg)

    assert os.path.exists("./data/tmp/corpora_filtered.parquet")
