import os
import pytest
from ekorpkit import eKonf


@pytest.mark.gpu()
def test_disco():

    cfg = eKonf.compose("model/disco")
    disco = eKonf.instantiate(cfg)
    disco.imagine(n_samples=1)
    assert True
