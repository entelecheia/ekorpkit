from cmath import isinf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .base import set_style, set_figure
from ekorpkit import eKonf


def plot(data, verbose=False, **kwargs):
    kwargs = eKonf.to_dict(kwargs)
    plots = kwargs.get("plots")
    dataset = {} or kwargs.get("dataset")
    plot = {} or kwargs.get("plot")
    figure = {} or kwargs.get("figure")
    figure2 = {} or kwargs.get("figure2")
    savefig = {} or kwargs.get("savefig")

    if data is None:
        if verbose:
            print("No data to plot")
        return

    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)

    dataform = dataset.get("form", "individual")
    xcol = dataset["x"]
    if isinstance(xcol, list):
        xcol = xcol[0]
    x = data[xcol] if xcol in data.columns else data.index
    ycols = dataset["y"]
    if isinstance(ycols, str):
        ycols = [ycols]

    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()
    ax2 = None

    if dataform == "individual":
        if isinstance(plots, dict):
            plots = [plots]
        for plot_args in plots:
            _function = eval(plot_args.get("function"))
            y = plot_args.pop("y")
            secondary_y = plot_args.get("secondary_y", False)
            if secondary_y and ax2 is None:
                if verbose:
                    print("Creating secondary axis")
                ax2 = ax.twinx()
            _function(ax2 if secondary_y else ax, x, y, data, **plot_args)
    else:
        if isinstance(plots, list):
            plot_args = plots[0]
        else:
            plot_args = plots
        _function = eval(plot_args.get("function"))
        if len(ycols) == 1:
            ycols = ycols[0]
        y = plot_args.pop("y")
        if plot_args.get("dataform"):
            dataform = plot_args.pop("dataform")
        if dataform == "wide":
            if xcol and xcol in data.columns:
                _data = data.set_index(xcol)[ycols]
            else:
                _data = data[ycols]
            _function(ax, x=None, y=None, data=_data, **plot_args)
        else:
            _function(ax, x=xcol, y=ycols, data=data, **plot_args)

    add_decorations(ax, **kwargs)
    set_figure(ax, **figure)
    if ax2 is not None and figure2 is not None:
        set_figure(ax2, **figure2)

    fname = savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**savefig)
        if verbose:
            print(f"Saved figure to {fname}")


def lineplot(ax=None, x=None, y=None, data=None, args=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    sns.lineplot(x=x, y=y, data=data, ax=ax, **args)


def stackplot(ax=None, x=None, y=None, data=None, args=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = data.index
    elif isinstance(x, str):
        x = data[x]
    plt.stackplot(x, data[y].T, **args)


def scatter(ax=None, x=None, y=None, data=None, args=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    sns.scatterplot(x=x, y=y, data=data, ax=ax, **args)


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
            if annot.get("xy") is not None and isinstance(annot.get("xy"), str):
                annot["xy"] = eval(annot["xy"])
            if annot.get("xytext") is not None and isinstance(annot.get("xytext"), str):
                annot["xytext"] = eval(annot["xytext"])
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


def barplot(df, columns=None, savefig={}, plot={}, figure={}, verbose=False, **kwargs):
    if df is None:
        if verbose:
            print("No data to plot")
        return
    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    ycols = columns.yvalue
    xcol = columns.xvalue
    index = columns.get("index", None)
    if index:
        df.index = list(index)
        xcol = None
    data = df

    # plt.figure(figsize=figsize, tight_layout=True)
    stacked = plot.get("stacked", False)
    ax = data.plot(x=xcol, y=ycols, kind="bar", stacked=stacked, figsize=figsize)
    set_figure(ax, **figure)
    plt.tight_layout()

    fname = savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**savefig)
        if verbose:
            print(f"Saved figure to {fname}")
