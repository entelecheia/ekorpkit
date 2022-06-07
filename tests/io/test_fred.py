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
    cfg.plots[0].y = "DFEDTAR"
    cfg.figure.figsize = (15, 8)
    cfg.ax.title = "Fed Rate"
    eKonf.instantiate(cfg, data=fred.data)

    assert True
