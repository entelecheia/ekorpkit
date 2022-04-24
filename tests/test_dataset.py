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
