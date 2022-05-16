from ekorpkit import eKonf


def test_compose_config():
    cfg = eKonf.compose()
    cfg = eKonf.to_dict(cfg)
    assert type(cfg) == dict


def test_about():
    from ekorpkit.cli import about

    cfg = eKonf.compose()
    about(**cfg)
    assert True


def test_dependencies():
    deps = eKonf.dependencies("all")

    assert type(deps) == set
    assert len(deps) == 40
