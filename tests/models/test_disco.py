import os
import pytest
from ekorpkit import eKonf


@pytest.mark.gpu()
def test_disco():

    cfg = eKonf.compose("model/disco")
    cfg.diffuse.n_samples = 1
    disco = eKonf.instantiate(cfg)
    disco.imagine()
    assert True
