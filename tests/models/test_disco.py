import pytest
from ekorpkit import eKonf


@pytest.mark.gpu()
def test_disco():

    cfg = eKonf.compose("model/disco")
    disco = eKonf.instantiate(cfg)

    text_prompts = "A beautiful water painting of Jeju Island."
    batch_name = "jeju"

    results = disco.imagine(
        text_prompts,
        batch_name=batch_name,
        n_samples=1,
        steps=100,
    )
    assert results is not None
