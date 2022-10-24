from ekorpkit import eKonf


def test_automl():

    fs_cfg = eKonf.compose("dataset=feature")
    fs_cfg.name = "fomc_features_small"
    fs_cfg.path.cache.uri = "https://github.com/entelecheia/ekorpkit-book/raw/main/assets/data/fomc_features_small.zip"
    fs_cfg.data_dir = fs_cfg.path.cached_path

    model_cfg = eKonf.compose("model/automl=classification")
    model_cfg.dataset = fs_cfg
    model_cfg.config.time_budget = 60
    model_cfg.verbose = False
    model = eKonf.instantiate(model_cfg)

    model.fit()

    model.save()
    model.load()
    model.plot_learning_curve()
    model.eval()
    model.plot_feature_importance()

    assert True
