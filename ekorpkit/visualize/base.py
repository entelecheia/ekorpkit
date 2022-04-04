import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pathlib import Path


def _get_font_name(set_font_for_matplot=True, fontpath=None, verbose=False):
    if not fontpath:
        if platform.system() == "Darwin":
            fontpath = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        elif platform.system() == "Windows":
            fontpath = "c:/Windows/Fonts/malgun.ttf"
        elif platform.system() == "Linux":
            fontpath = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    if not Path(fontpath).is_file():
        fontname = None
        fontpath = None
        print(f"Font file does not exist at {fontpath}")
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
    if verbose:
        print("font family: ", plt.rcParams["font.family"])
        print("font path: ", fontpath)
    return fontname, fontpath
