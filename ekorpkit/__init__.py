from hyfi import about, global_config

from . import _version
from .ekonf import eKonf
from ._version import __version__

about.name = "ekorpkit"
about.author = "Young Joon Lee"
about.description = "ekorpkit provides a flexible interface for NLP and ML research pipelines \
such as extraction, transformation, tokenization, training, and visualization."
about.homepage = "https://entelecheia.cc"
about.version = __version__
global_config.hyfi_package_config_path = "pkg://ekorpkit.conf"


def get_version() -> str:
    """This is the cli function of the package"""
    return __version__
