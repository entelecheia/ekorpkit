from .ekonf import eKonf
from .hyfi import global_config, about
from . import _version

__version__ = _version.get_versions()["version"]
about.name = "ekorpkit"
about.author = "Young Joon Lee"
about.description = "ekorpkit provides a flexible interface for NLP and ML research pipelines such as extraction, transformation, tokenization, training, and visualization."
about.website = "https://entelecheia.cc"
about.version = __version__
global_config.hyfi_package_config_path = "pkg://ekorpkit.conf"
