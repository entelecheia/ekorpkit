import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from pathlib import Path


def _get_font_name(set_font_for_matplot=True, font_path=None):
    if not font_path:
        if platform.system() == "Darwin":
            font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        elif platform.system() == "Windows":
            font_path = "c:/Windows/Fonts/malgun.ttf"
        elif platform.system() == "Linux":
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    if not Path(font_path).is_file():
        font_name = None
        font_path = None
        print(f"Font file does not exist at {font_path}")
        if platform.system() == "Linux":
            font_install_help = """
            apt install fontconfig
            apt install fonts-nanum
            fc-list | grep -i nanum
            """
            print(font_install_help)
    else:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        if set_font_for_matplot and font_name:
            rc("font", family=font_name)
            plt.rcParams["axes.unicode_minus"] = False
    print("font family: ", plt.rcParams["font.family"])
    print("font path: ", font_path)
    return font_name, font_path
