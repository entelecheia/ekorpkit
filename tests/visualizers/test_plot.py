from ekorpkit import eKonf


def test_plot():

    cfg = eKonf.compose("dataset=feature")
    cfg.name = "fomc_features_small"
    cfg.path.cache.uri = "https://github.com/entelecheia/ekorpkit-book/raw/main/assets/data/fomc_features_small.zip"
    cfg.data_dir = cfg.path.cached_path
    f_small = eKonf.instantiate(cfg)

    cfg = eKonf.compose("visualize/plot=lineplot")
    cfg.plots[0].x = "date"
    cfg.plots[0].y = "PMI"
    cfg.figure.figsize = (15, 8)
    cfg.ax.title = "PMI"
    eKonf.instantiate(cfg, data=f_small.data)

    assert True
