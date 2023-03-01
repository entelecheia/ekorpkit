"""Plotting functions"""
import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

from ..utils.logging import getLogger

logger = getLogger(__name__)


def get_plot_font(
    set_font_for_matplot=True, fontpath=None, fontname=None, verbose=False
):
    """Get font for plot"""
    if fontname and not fontname.endswith(".ttf"):
        fontname += ".ttf"
    if not fontpath:
        if platform.system() == "Darwin":
            fontname = fontname or "AppleGothic.ttf"
            fontpath = os.path.join("/System/Library/Fonts/Supplemental/", fontname)
        elif platform.system() == "Windows":
            fontname = fontname or "malgun.ttf"
            fontpath = os.path.join("c:/Windows/Fonts/", fontname)
        elif platform.system() == "Linux":
            fontname = fontname or "NanumGothic.ttf"
            if fontname.lower().startswith("nanum"):
                fontpath = os.path.join("/usr/share/fonts/truetype/nanum/", fontname)
            else:
                fontpath = os.path.join("/usr/share/fonts/truetype/", fontname)
        if fontpath and not Path(fontpath).is_file():
            paths = find_font_file(fontname)
            if len(paths) > 0:
                fontpath = paths[0]
            else:
                fontpath = None
        if verbose:
            logger.info(f"Font path: {fontpath}")

    if fontpath is None or not Path(fontpath).is_file():
        fontname = None
        fontpath = None
        logger.warning(f"Font file does not exist at {fontpath}")
        if platform.system() == "Linux":
            font_install_help = """
            apt install fontconfig
            apt install fonts-nanum
            fc-list | grep -i nanum
            """
            print(font_install_help)
    else:
        font_manager.fontManager.addfont(fontpath)
        fontname = font_manager.FontProperties(fname=fontpath).get_name()

        if set_font_for_matplot and fontname:
            rc("font", family=fontname)
            plt.rcParams["axes.unicode_minus"] = False
            font_family = plt.rcParams["font.family"]
            if verbose:
                logger.info(f"font family: {font_family}")
        if verbose:
            logger.info(f"font name: {fontname}")
    return fontname, fontpath


def find_font_file(query):
    """Find font file by query string"""
    matches = list(
        filter(
            lambda path: query in os.path.basename(path), font_manager.findSystemFonts()
        )
    )
    return matches
