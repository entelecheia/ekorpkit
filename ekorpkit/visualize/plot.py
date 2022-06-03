import logging
import omegaconf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cmath import isinf
from pathlib import Path
from .base import set_style, set_figure, set_super
from .classification import confusion_matrix
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
    for i, _plot_cfg_ in enumerate(_plots):
        _func_ = eval(_plot_cfg_.get(eKonf.Keys.FUNC))
        _x = _plot_cfg_.pop("x", None)
        _y = _plot_cfg_.pop("y", None)
        secondary_y = _plot_cfg_.get("secondary_y", False)
        if secondary_y:
            secondary_to = _plot_cfg_.get("secondary_to", 0)
            if secondary_to in sencondary_axes:
                ax = sencondary_axes[secondary_to]
            else:
                ax = axes[secondary_to].twinx()
                log.info(f"Creating secondary axis to axis[{secondary_to}]")
                sencondary_axes[secondary_to] = ax
        else:
            axno = _plot_cfg_.get("axno", 0)
            ax = axes[axno]
        datano = _plot_cfg_.get("datano", 0)
        if isinstance(data, list):
            _data = data[datano]
            log.debug(f"Plotting data[{datano}]")
        else:
            _data = data
        _data = prepare_data(_data, **_plot_cfg_)
        _func_(ax, _x, _y, _data, **_plot_cfg_)

    if _axes is None:
        _axes = []
    elif not isinstance(_axes, (dict, list)):
        _axes = []
    elif isinstance(_axes, dict):
        _axes = [_axes]
    for _ax_cfg_ in _axes:
        secondary_y = _ax_cfg_.get("secondary_y", False)
        if secondary_y:
            secondary_to = _ax_cfg_.get("secondary_to", 0)
            if secondary_to in sencondary_axes:
                ax = sencondary_axes[secondary_to]
            else:
                ax = None
        else:
            axno = _ax_cfg_.get("axno", 0)
            ax = axes[axno]
        add_decorations(ax, **_ax_cfg_)
        set_figure(ax, **_ax_cfg_)

    set_super(fig, **_figure["super"])

    fname = _savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**_savefig)
        log.info(f"Saved figure to {fname}")


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

    _func_ = eval(_grid.get(eKonf.Keys.FUNC))
    _data = prepare_data(data, **_grid)
    g = _func_(_data, **_grid)

    fname = _savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**_savefig)
        log.info(f"Saved figure to {fname}")


def prepare_data(data, **kwargs):
    _query = kwargs.get("query", None)
    _set_index = kwargs.get("set_index", None)
    if _query is not None:
        if isinstance(data, pd.DataFrame):
            data = data.copy().query(_query, engine="python")
        else:
            data = [d for d in data if eval(_query)]
    if _set_index is not None:
        if isinstance(data, pd.DataFrame):
            if _set_index == "reset":
                data = data.copy().reset_index()
            else:
                data = data.copy().set_index(_set_index)
    return data


def facetgrid(data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    _name_ = kwargs.get(eKonf.Keys.NAME)
    _map = kwargs.get(_name_)
    _map_fn = _map.get(eKonf.Keys.FUNC)
    _map_parms = _map.get(eKonf.Keys.PARMS)

    g = sns.FacetGrid(data, **_parms_)
    map_fn = getattr(sns, _map_fn)
    getattr(g, _name_)(map_fn, **_map_parms)
    if kwargs.get("add_legend", False):
        g.add_legend()
    return g


def snsplot(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    _name_ = kwargs.get(eKonf.Keys.NAME)
    if ax is None:
        ax = plt.gca()
    if isinstance(y, list):
        data = data[y]
        y = None
    if x is not None:
        _parms_["x"] = x
    if y is not None:
        _parms_["y"] = y
    getattr(sns, _name_)(data=data, ax=ax, **_parms_)


def heatmap(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    if ax is None:
        ax = plt.gca()
    sns.heatmap(data=data, ax=ax, **_parms_)


def lineplot(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    if ax is None:
        ax = plt.gca()
    sns.lineplot(x=x, y=y, data=data, ax=ax, **_parms_)


def countplot(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    if ax is None:
        ax = plt.gca()
    sns.countplot(x=x, y=y, data=data, ax=ax, **_parms_)


def histplot(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    if ax is None:
        ax = plt.gca()
    sns.histplot(x=x, y=y, data=data, ax=ax, **_parms_)


def stackplot(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = data.index
    elif isinstance(x, str):
        x = data[x]
    plt.stackplot(x, data[y].T, **_parms_)


def scatter(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    if ax is None:
        ax = plt.gca()
    sns.scatterplot(x=x, y=y, data=data, ax=ax, **_parms_)


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


def treemap(
    df,
    fig_filepath,
    scale=1.5,
    columns=None,
    treemap=None,
    layout=None,
    update_layout=None,
    **kwargs,
):
    # textinfo = "label+value+percent parent+percent root"
    # import plotly.express as px
    import plotly.graph_objects as go

    labels = list(df[columns.label].to_list())
    values = list(df[columns.value].to_list())
    parents = list(df[columns.parent].to_list())

    layout = go.Layout(**layout)
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            **treemap,
        ),
        layout=layout,
    )
    fig.update_layout(
        **update_layout,
    )

    fig.write_image(fig_filepath, scale=scale)


# def barplot(df, columns=None, savefig={}, plot={}, figure={}, verbose=False, **kwargs):
#     if df is None:
#         if verbose:
#             print("No data to plot")
#         return
#     set_style(**plot)
#     figsize = plot.get("figsize", None)
#     if figsize is not None and isinstance(figsize, str):
#         figsize = eval(figsize)
#     ycols = columns.yvalue
#     xcol = columns.xvalue
#     index = columns.get("index", None)
#     if index:
#         df.index = list(index)
#         xcol = None
#     data = df

#     # plt.figure(figsize=figsize, tight_layout=True)
#     stacked = plot.get("stacked", False)
#     ax = data.plot(x=xcol, y=ycols, kind="bar", stacked=stacked, figsize=figsize)
#     set_figure(ax, **figure)
#     plt.tight_layout()

#     fname = savefig.get("fname", None)
#     if fname:
#         Path(fname).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(**savefig)
#         if verbose:
#             print(f"Saved figure to {fname}")
def t(econ_train_large, econ_train_small):

    # Retail sales growth ratio is difficult to estimate. Though it is not ideal, simply use the average
    econ_train_small["RSALES_diff_year"].fillna(
        econ_train_small["RSALES_diff_year"].mean(), inplace=True
    )
    econ_train_large["RSALES_diff_prev"].fillna(
        econ_train_large["RSALES_diff_prev"].mean(), inplace=True
    )
    econ_train_large["RSALES_diff_year"].fillna(
        econ_train_large["RSALES_diff_year"].mean(), inplace=True
    )

    econ_train_small["Inertia_diff"].fillna(
        econ_train_small["Inertia_diff"].mean(), inplace=True
    )
    econ_train_small["Balanced_diff"].fillna(
        econ_train_small["Balanced_diff"].mean(), inplace=True
    )
    econ_train_large["Inertia_diff"].fillna(
        econ_train_large["Inertia_diff"].mean(), inplace=True
    )
    econ_train_large["Balanced_diff"].fillna(
        econ_train_large["Balanced_diff"].mean(), inplace=True
    )
    econ_train_large["Taylor_diff"].fillna(
        econ_train_large["Taylor_diff"].mean(), inplace=True
    )
