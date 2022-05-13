import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .base import set_style, set_figure


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


def lineplot(
    df,
    columns=None,
    savefig={},
    lineplot={},
    plot={},
    figure={},
    verbose=False,
    **kwargs,
):

    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    ycols = columns.yvalue
    xcol = columns.xvalue
    data = df.set_index(xcol) if xcol in df.columns else df

    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()
    sns.lineplot(data=data[ycols], **lineplot)
    set_figure(ax, **figure)
    fname = savefig.get("fname", None)

    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**savefig)
        if verbose:
            print(f"Saved figure to {fname}")


def stackplot(
    df, columns=None, savefig={}, plot={}, figure={}, verbose=False, **kwargs
):
    if df is None:
        if verbose:
            print("No data to plot")
        return
    data = df

    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    ycols = columns.yvalue
    xcol = columns.xvalue
    if isinstance(xcol, list):
        xcol = xcol[0]
    x = data[xcol] if xcol in data.columns else data.index

    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()
    labels = figure.get("legend", {}).get("labels", None)
    if labels is None:
        labels = ycols
    plt.stackplot(x, data[ycols].T, labels=labels)
    set_figure(ax, **figure)

    fname = savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**savefig)
        if verbose:
            print(f"Saved figure to {fname}")


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
