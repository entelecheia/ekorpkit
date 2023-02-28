from .ekonf import eKonf
from .hyfi import global_config
from . import _version

__version__ = _version.get_versions()["version"]
global_config.hyfi_package_config_path = "pkg://ekorpkit.conf"
