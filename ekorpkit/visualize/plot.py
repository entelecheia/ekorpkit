import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from pathlib import Path
from .base import _get_font_name


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


def lineplot(df, columns=None, savefig={}, plot={}, figure={}, verbose=False, **kwargs):

    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    ycols = columns.yvalue
    xcol = columns.xvalue
    data = df.set_index(xcol) if xcol in df.columns else df

    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()
    linewidth = plot.get("linewidth", None)
    sns.lineplot(data=data[ycols], linewidth=linewidth)
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
    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    ycols = columns.yvalue
    xcol = columns.xvalue
    data = df

    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()
    labels = figure.get("legend", {}).get("labels", None)
    if labels is None:
        labels = ycols
    plt.stackplot(data[xcol], data[ycols].T, labels=labels)
    set_figure(ax, **figure)
    fname = savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**savefig)
        if verbose:
            print(f"Saved figure to {fname}")


def set_figure(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None,
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
        ax.legend(**legend)
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


def set_style(style, rcParams, fontpath=None, **kwargs):
    fontname, fontpath = _get_font_name(fontpath=fontpath)
    rcParams["font.family"] = fontname

    plt.style.use(style)
    plt.rcParams.update(rcParams)
