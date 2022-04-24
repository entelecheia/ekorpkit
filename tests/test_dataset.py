import os
from ekorpkit import eKonf


def test_build_datasets():
    cfg = eKonf.compose(config_group="dataset/simple=financial_phrasebank")
    cfg["data_dir"] = "./data/tmp/financial_phrasebank"
    cfg.fetch.data_dir = cfg.data_dir
    cfg.fetch.overwrite = True
    cfg.fetch.calculate_stats = True
    db = eKonf.instantiate(cfg)
    db.build()

    cfg = eKonf.compose(config_group="dataset/simple=sst2")
    cfg["data_dir"] = "./data/tmp/sst2"
    cfg.fetch.data_dir = cfg.data_dir
    cfg.fetch.overwrite = True
    cfg.fetch.calculate_stats = True
    db = eKonf.instantiate(cfg)
    db.build()

    cfg = eKonf.compose(config_group="dataset=datasets")
    cfg["name"] = ["financial_phrasebank", "sst2"]
    cfg["data_dir"] = "./data/tmp"
    ds = eKonf.instantiate(cfg)
    ds.persist()

    assert len(ds.datasets) == 2


def test_datafame_pipeline():
    cfg = eKonf.compose(config_group="pipeline=dataframe_pipeline")
    cfg.verbose = True
    cfg.data_dir = "./data/tmp/financial_phrasebank"
    cfg.data_file = "financial_phrasebank-train.csv"
    cfg._pipeline_ = ["load_dataframe", "summary_stats"]
    cfg.summary_stats.output_file = "stats.yaml"
    eKonf.instantiate(cfg)

    assert os.path.exists(f"{cfg.data_dir}/{cfg.summary_stats.output_file}")
