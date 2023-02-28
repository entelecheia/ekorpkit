from .base import _SPLITS, __ekorpkit_path__, _Defaults, _Keys
from .hyfi import HyFI


class eKonf(HyFI):
    """ekorpkit config primary class"""

    __ekorpkit_path__ = __ekorpkit_path__()
    book_repo = "https://github.com/entelecheia/ekorpkit-book/raw/main/"
    book_repo_assets = book_repo + "assets/"
    book_url = "https://entelecheia.github.io/ekorpkit-book/"
    book_assets_url = book_url + "assets/"

    Defaults = _Defaults
    Keys = _Keys
    SPLITS = _SPLITS
