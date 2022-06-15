import logging
import omegaconf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cmath import isinf
from pathlib import Path
from .base import set_style, set_figure, set_super, add_decorations, save_figure
from .classification import confusion_matrix
from .yellowbrick import yellowbrick_features
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def plot(data, verbose=False, **kwargs):
    kwargs = eKonf.to_dict(kwargs)
    _plots = kwargs.get("plots")
    _axes = kwargs.get("axes")
    # _series = {} or kwargs.get("series")
    _figure = {} or kwargs.get("figure")
    _gridspec = {} or kwargs.get("gridspec")
    _subplots = {} or kwargs.get("subplots")
    _savefig = {} or kwargs.get("savefig")

    if data is None:
        log.info("No data to plot")
    if isinstance(data, omegaconf.listconfig.ListConfig):
        data = list(data)
    if verbose:
        log.info(f"type of data: {type(data)}")

    set_style(**_figure)
    figsize = _figure.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)

    if verbose and isinstance(_plots, list) and len(_plots) > 1:
        log.info("Plotting multiple plots'")

    sencondary_axes = {}
    fig = plt.figure(figsize=figsize, tight_layout=_figure["tight_layout"])

    _gridspec["nrows"] = _subplots.pop("nrows", 1)
    _gridspec["ncols"] = _subplots.pop("ncols", 1)
    gs = fig.add_gridspec(**_gridspec)
    axes = gs.subplots(**_subplots)
    if _gridspec["nrows"] > 1 or _gridspec["ncols"] > 1:
        ax = None
        axes = axes.flatten()
    else:
        axes = [axes]
        ax = axes[0]

    if isinstance(_plots, dict):
        _plots = [_plots]
    if not isinstance(_plots, list) or data is None:
        log.info("No plots to plot")
        _plots = []
    for i, _plot_cfg_ in enumerate(_plots):
        _func_ = eval(_plot_cfg_.pop(eKonf.Keys.FUNC))
        _x = _plot_cfg_.pop("x", None)
        _y = _plot_cfg_.pop("y", None)
        _secondary_y = _plot_cfg_.pop("secondary_y", False)
        _secondary_to = _plot_cfg_.pop("secondary_to", 0)
        _axno = _plot_cfg_.pop("axno", 0)
        _datano = _plot_cfg_.pop("datano", 0)
        _query = _plot_cfg_.pop("query", None)
        _set_index = _plot_cfg_.pop("set_index", None)

        if _secondary_y:
            if _secondary_to in sencondary_axes:
                ax = sencondary_axes[_secondary_to]
            else:
                ax = axes[_secondary_to].twinx()
                log.info(f"Creating secondary axis to axis[{_secondary_to}]")
                sencondary_axes[_secondary_to] = ax
        else:
            ax = axes[_axno]
        if isinstance(data, list):
            _data = data[_datano]
            log.debug(f"Plotting data[{_datano}]")
        else:
            _data = data
        _data = prepare_data(_data, _query, _set_index)
        _func_(ax, _x, _y, _data, **_plot_cfg_)

    if _axes is None:
        _axes = []
    elif not isinstance(_axes, (dict, list)):
        _axes = []
    elif isinstance(_axes, dict):
        _axes = [_axes]
    for _ax_cfg_ in _axes:
        _secondary_y = _ax_cfg_.get("secondary_y", False)
        if _secondary_y:
            _secondary_to = _ax_cfg_.get("secondary_to", 0)
            if _secondary_to in sencondary_axes:
                ax = sencondary_axes[_secondary_to]
            else:
                ax = None
        else:
            _axno = _ax_cfg_.get("axno", 0)
            ax = axes[_axno]
        add_decorations(ax, **_ax_cfg_)
        set_figure(ax, **_ax_cfg_)

    set_super(fig, **_figure["super"])

    save_figure(fig, **_savefig)


def grid(data, verbose=False, **kwargs):
    kwargs = eKonf.to_dict(kwargs)
    _theme = kwargs.get("theme")
    _figure = {} or kwargs.get("figure")
    _grid = {} or kwargs.get("grid")
    _gridspec = {} or kwargs.get("gridspec")
    _subplots = {} or kwargs.get("subplots")
    _savefig = {} or kwargs.get("savefig")

    if data is None:
        log.warning("No data to plot")
        return
    if isinstance(data, omegaconf.listconfig.ListConfig):
        data = list(data)
    if verbose:
        log.info(f"type of data: {type(data)}")

    set_style(**_figure)
    figsize = _figure.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)

    if isinstance(_theme["rc"], dict):
        _theme["rc"]["figure.figsize"] = figsize
    else:
        _theme["rc"] = {"figure.figsize": figsize}
    sns.set_theme(**_theme)

    _gridspec["nrows"] = _subplots.pop("nrows", 1)
    _gridspec["ncols"] = _subplots.pop("ncols", 1)

    _query = _grid.pop("query", None)
    _set_index = _grid.pop("set_index", None)
    _func_ = eval(_grid.get(eKonf.Keys.FUNC))
    _data = prepare_data(data, _query, _set_index)
    g = _func_(_data, **_grid)

    save_figure(plt, **_savefig)


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


def facetgrid(data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    _name_ = kwargs.get(eKonf.Keys.NAME_KEY)
    _map = kwargs.get(_name_)
    _map_fn = _map.get(eKonf.Keys.FUNC)
    _map_parms = _map.get(eKonf.Keys.rcPARAMS)

    g = sns.FacetGrid(data, **rcParams)
    map_fn = getattr(sns, _map_fn)
    getattr(g, _name_)(map_fn, **_map_parms)
    if kwargs.get("add_legend", False):
        g.add_legend()
    return g


def snsplot(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = kwargs.pop(eKonf.Keys.rcPARAMS, {}) or {}
    _name_ = kwargs.pop(eKonf.Keys.NAME_KEY)
    if ax is None:
        ax = plt.gca()
    if isinstance(y, list):
        if x is not None and x in data.columns:
            data.set_index(x, inplace=True)
            x = None
        data = data[y]
        y = None
    if x is not None:
        rcParams["x"] = x
    if y is not None:
        rcParams["y"] = y
    _fn = getattr(sns, _name_)
    if isinstance(kwargs, dict):
        rcParams.update(kwargs)
    log.info(f"Plotting {_name_} with {rcParams}")
    _fn(data=data, ax=ax, **rcParams)


def heatmap(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    sns.heatmap(data=data, ax=ax, **rcParams)


def lineplot(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    sns.lineplot(x=x, y=y, data=data, ax=ax, **rcParams)


def countplot(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    sns.countplot(x=x, y=y, data=data, ax=ax, **rcParams)


def histplot(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    sns.histplot(x=x, y=y, data=data, ax=ax, **rcParams)


def stackplot(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    if x is None or x not in data.columns:
        x = data.index
    elif isinstance(x, str):
        x = data[x]
    plt.stackplot(x, data[y].T, **rcParams)


def scatter(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    sns.scatterplot(x=x, y=y, data=data, ax=ax, **rcParams)
