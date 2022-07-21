import os
import pandas as pd
import omegaconf
import logging
import platform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc
from pathlib import Path
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def prepare_data(data, _query=None, _set_index=None, **kwargs):
    data = data.copy()
    if _query is not None:
        if isinstance(data, pd.DataFrame):
            data = data.query(_query, engine="python")
        else:
            data = [d for d in data if eval(_query)]
    if _set_index is not None:
        if isinstance(data, pd.DataFrame):
            if _set_index == "reset":
                data = data.reset_index()
            else:
                data = data.set_index(_set_index)
    return data


def prepare_plot_args(data, verbose=False, **kwargs):
    kwargs = eKonf.to_dict(kwargs)
    _plots = kwargs.get("plots")
    _axes = kwargs.get("axes")
    _figure = kwargs.get("figure") or {}
    _style = kwargs.get("style") or {}
    _gridspec = kwargs.get("gridspec") or {}
    _subplots = kwargs.get("subplots") or {}
    _savefig = kwargs.get("savefig") or {}
    _theme = kwargs.get("theme") or {}
    _grid = kwargs.get("grid") or {}
    _fig_super = _figure.pop("super", None) or {}
    _fig_rcParams = _figure.pop("rcParams", None) or {}

    _style["rcParams"] = _style.get("rcParams") or {}
    _style["rcParams"].update(_fig_rcParams)
    _gridspec["nrows"] = _subplots.pop("nrows", 1)
    _gridspec["ncols"] = _subplots.pop("ncols", 1)

    if data is None:
        log.info("No data to plot")
    if isinstance(data, omegaconf.listconfig.ListConfig):
        data = list(data)
    if verbose:
        log.info(f"type of data: {type(data)}")

    tight_layout = _figure.pop("tight_layout", True)
    figsize = _figure.pop("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)

    return (
        _axes,
        _fig_super,
        _grid,
        _gridspec,
        _plots,
        _savefig,
        _style,
        _subplots,
        _theme,
        data,
        figsize,
        tight_layout,
    )


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
    xtick_params=None,
    ytick_params=None,
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
    if xtick_params is not None:
        ax.tick_params(axis="x", **xtick_params)
    if ytick_params is not None:
        ax.tick_params(axis="y", **ytick_params)


def set_style(style, rcParams, fontpath=None, language=None, **kwargs):
    if language or (fontpath and Path(fontpath).is_file()):
        fontname, fontpath = _configure_font(fontpath=fontpath)
        if fontname:
            rcParams["font.family"] = fontname

    plt.style.use(style)
    plt.rcParams.update(rcParams)


def find_font_file(query):
    matches = list(
        filter(
            lambda path: query in os.path.basename(path), font_manager.findSystemFonts()
        )
    )
    return matches


def _configure_font(
    set_font_for_matplot=True, fontpath=None, fontname=None, verbose=False
):
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
            log.info(f"Font path: {fontpath}")

    if fontpath is None or not Path(fontpath).is_file():
        fontname = None
        fontpath = None
        log.warning(f"Font file does not exist at {fontpath}")
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
                log.info(f"font family: {font_family}")
        if verbose:
            log.info(f"font name: {fontname}")
    return fontname, fontpath
