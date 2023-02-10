from .ekonf import eKonf
from .hyfi import global_env
from . import _version

__version__ = _version.get_versions()["version"]
global_env.hyfi_package_config_path = "pkg://ekorpkit.conf"
