import logging
import matplotlib.pyplot as plt
import seaborn as sns
from cmath import isinf
from pathlib import Path
from .base import (
    set_style,
    set_figure,
    set_super,
    add_decorations,
    save_figure,
    prepare_plot_args,
    prepare_data,
)
from .classification import confusion_matrix
from .yellowbrick import yellowbrick_features
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def plot(data, verbose=False, **kwargs):
    (
        _axes,
        _fig_super,
        _,
        _gridspec,
        _plots,
        _savefig,
        _style,
        _subplots,
        _,
        data,
        figsize,
        tight_layout,
    ) = prepare_plot_args(data, verbose, **kwargs)

    set_style(**_style)

    if verbose and isinstance(_plots, list) and len(_plots) > 1:
        log.info("Plotting multiple plots'")

    sencondary_axes = {}
    fig = plt.figure(figsize=figsize, tight_layout=tight_layout)

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

    set_super(fig, **_fig_super)

    save_figure(fig, **_savefig)


def grid(data, verbose=False, **kwargs):
    (
        _,
        _,
        _grid,
        _,
        _,
        _savefig,
        _style,
        _,
        _theme,
        data,
        figsize,
        _,
    ) = prepare_plot_args(data, verbose, **kwargs)

    set_style(**_style)

    if isinstance(_theme["rc"], dict):
        _theme["rc"]["figure.figsize"] = figsize
    else:
        _theme["rc"] = {"figure.figsize": figsize}
    sns.set_theme(**_theme)

    _query = _grid.pop("query", None)
    _set_index = _grid.pop("set_index", None)
    _func_ = eval(_grid.get(eKonf.Keys.FUNC))
    _data = prepare_data(data, _query, _set_index)
    g = _func_(_data, **_grid)

    save_figure(plt, **_savefig)


def facetgrid(data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    _name_ = kwargs.get(eKonf.Keys.METHOD_NAME)
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
    _name_ = kwargs.pop(eKonf.Keys.METHOD_NAME)
    if ax is None:
        ax = plt.gca()
    if isinstance(y, list) and len(y) == 1:
        y = y[0]
    if isinstance(y, list):
        if x is not None and x in data.columns:
            data.set_index(x, inplace=True)
        data = data[y]
        x = None
        y = None
    elif isinstance(y, str):
        if x is None:
            if data.index.name is not None:
                x = data.index.name
            else:
                x = data.index
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


def stackplot(ax=None, x=None, y=None, data=None, **kwargs):
    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    if ax is None:
        ax = plt.gca()
    if x is None or x not in data.columns:
        x = data.index
    elif isinstance(x, str):
        x = data[x]
    plt.stackplot(x, data[y].T, **rcParams)
