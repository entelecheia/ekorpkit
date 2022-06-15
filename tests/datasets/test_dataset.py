import os
import pytest
from ekorpkit import eKonf


@pytest.mark.skip(reason="data source not available")
def test_build_financial_phrasebank():
    cfg = eKonf.compose("dataset/simple=financial_phrasebank")
    cfg.data_dir = "./data/tmp/financial_phrasebank"
    cfg.io.data_dir = cfg.data_dir
    cfg.io.force.build = True
    cfg.io.force.summarize = True
    db = eKonf.instantiate(cfg)
    assert True


def test_build_datasets():

    cfg = eKonf.compose("dataset/simple=sst2")
    cfg.data_dir = "./data/tmp/sst2"
    cfg.io.data_dir = cfg.data_dir
    cfg.io.force.build = True
    cfg.io.force.summarize = True
    eKonf.instantiate(cfg)

    cfg = eKonf.compose("dataset=datasets")
    cfg.name = ["sst2"]
    cfg.data_dir = "./data/tmp"
    ds = eKonf.instantiate(cfg)
    ds.persist()

    assert len(ds.datasets) == 1


def test_datafame_pipeline():
    cfg = eKonf.compose("pipeline")
    cfg.data_dir = "./data/tmp/sst2"
    cfg.data_file = "sst2-train.parquet"
    cfg._pipeline_ = ["summary_stats"]
    cfg.summary_stats.output_file = "stats.yaml"
    eKonf.instantiate(cfg)

    assert os.path.exists(f"{cfg.data_dir}/{cfg.summary_stats.output_file}")
