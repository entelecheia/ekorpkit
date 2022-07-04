import os
import pytest
from ekorpkit import eKonf


@pytest.mark.gpu()
def test_disco():

    cfg = eKonf.compose("model/disco")
    cfg.basic.batch_name = "dovish"
    cfg.diffuse.n_batches = 1
    disco = eKonf.instantiate(cfg)

    disco.load()
    disco.diffuse()

    assert True
