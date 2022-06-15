import logging
import platform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc
from pathlib import Path


log = logging.getLogger(__name__)


def save_figure(fig, fname=None, **kwargs):
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname=fname, **kwargs)
        log.info(f"Saved figure to {fname}")


def add_decorations(ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    axvspans = [] or kwargs.get("axvspans")
    annotations = [] or kwargs.get("annotations")

    if isinstance(axvspans, dict):
        axvspans = [axvspans]
    if isinstance(annotations, dict):
        annotations = [annotations]

    for span in axvspans:
        if isinstance(span, dict):
            ax.axvspan(**span)
    for annot in annotations:
        if isinstance(annot, dict):
            x = annot.pop("x")
            y = annot.pop("y")
            if x and y:
                annot["xy"] = (x, y)
            xtext = annot.pop("xtext")
            ytext = annot.pop("ytext")
            if xtext and ytext:
                annot["xytext"] = (xtext, ytext)
            ax.annotate(**annot)


def set_super(
    fig,
    xlabel=None,
    ylabel=None,
    title=None,
    **kwargs,
):
    if xlabel is not None:
        if isinstance(xlabel, str):
            fig.supxlabel(xlabel)
        else:
            fig.supxlabel(**xlabel)
    if ylabel is not None:
        if isinstance(ylabel, str):
            fig.supylabel(ylabel)
        else:
            fig.supylabel(**ylabel)
    if title is not None:
        if isinstance(title, str):
            fig.suptitle(title)
        else:
            fig.suptitle(**title)


def set_figure(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    grid=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    xtickmajorformatterfunc=None,
    ytickmajorformatterfunc=None,
    **kwargs,
):
    if xlabel is not None:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(**xlabel)
    if ylabel is not None:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(**ylabel)
    if title is not None:
        if isinstance(title, str):
            ax.set_title(title)
        else:
            ax.set_title(**title)
    if legend is not None:
        if isinstance(legend, (str, list)):
            ax.legend(legend)
        elif legend.get("labels") is not None:
            ax.legend(**legend)
    if grid is not None:
        if isinstance(grid, bool):
            ax.grid(grid)
        elif isinstance(grid, dict):
            ax.grid(**grid)
    if xticks is not None:
        if isinstance(xticks, str):
            ax.set_xticks(eval(xticks))
        else:
            if xticks.get("labels", None) or xticks.get("ticks", None):
                ax.set_xticks(**xticks)
    if yticks is not None:
        if isinstance(yticks, str):
            ax.set_yticks(eval(yticks))
        else:
            if yticks.get("labels", None) or yticks.get("ticks", None):
                ax.set_yticks(**yticks)
    if xticklabels is not None:
        if not xticklabels.get("labels", None):
            xticklabels["labels"] = ax.get_xticks().tolist()
        ax.set_xticklabels(**xticklabels)
    if yticklabels is not None:
        if not yticklabels.get("labels", None):
            yticklabels["labels"] = ax.get_yticks().tolist()
        ax.set_yticklabels(**yticklabels)
    if xlim is not None:
        if isinstance(xlim, str):
            ax.set_xlim(eval(xlim))
        else:
            ax.set_xlim(**xlim)
    if ylim is not None:
        if isinstance(ylim, str):
            ax.set_ylim(eval(ylim))
        else:
            ax.set_ylim(**ylim)
    if xtickmajorformatterfunc is not None:
        ax.xaxis.set_major_formatter(eval(xtickmajorformatterfunc))
    if ytickmajorformatterfunc is not None:
        ax.yaxis.set_major_formatter(eval(ytickmajorformatterfunc))


def set_style(style, rcParams, fontpath=None, language=None, **kwargs):
    if language or (fontpath and Path(fontpath).is_file()):
        fontname, fontpath = _configure_font(fontpath=fontpath)
        if fontname:
            rcParams["font.family"] = fontname

    plt.style.use(style)
    plt.rcParams.update(rcParams)


def _configure_font(set_font_for_matplot=True, fontpath=None, verbose=False):
    if not fontpath:
        if platform.system() == "Darwin":
            fontpath = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        elif platform.system() == "Windows":
            fontpath = "c:/Windows/Fonts/malgun.ttf"
        elif platform.system() == "Linux":
            fontpath = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if verbose:
            print("Font path:", fontpath)

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
