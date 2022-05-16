import pytest
from ekorpkit import eKonf


@pytest.mark.local()
def test_fred():
    cfg = eKonf.compose(config_group="io/fetcher=fred")
    cfg.name = "fed_rate"
    cfg.series_id = ["DFEDTAR", "DFEDTARU"]

    fred = eKonf.instantiate(cfg)

    cfg = eKonf.compose(config_group="visualize/plot=lineplot")
    cfg.dataset.y = "fed_rate"
    cfg.dataset.form = "wide"
    cfg.plots[0].y = "fed_rate"
    cfg.plot.figsize = (15, 8)
    cfg.figure.title = "Fed Rate"
    eKonf.instantiate(cfg, data=fred.data)
