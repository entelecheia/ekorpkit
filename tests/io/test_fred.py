import pytest
from ekorpkit import eKonf


@pytest.mark.local()
def test_fred():
    cfg = eKonf.compose(config_group="io/fetcher=quandl")
    cfg.series_name = "DFEDTAR"
    cfg.series_id = ["DFEDTAR", "DFEDTARU"]
    cfg.force_download = True

    fred = eKonf.instantiate(cfg)

    cfg = eKonf.compose(config_group="visualize/plot=lineplot")
    cfg.series.y = "DFEDTAR"
    cfg.plot.figsize = (15, 8)
    cfg.figure.title = "Fed Rate"
    eKonf.instantiate(cfg, data=fred.data)

    assert True
