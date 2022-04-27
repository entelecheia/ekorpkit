import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from .base import _configure_font

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def generate_wordclouds(
    wordclouds_args,
    fig_output_dir,
    fig_filename_format,
    title_fontsize=20,
    title_color="green",
    ncols=5,
    nrows=1,
    dpi=300,
    figsize=(20, 20),
    save=True,
    save_each=False,
    save_masked=False,
    mask_dir=None,
    fontpath=None,
    colormap="PuBu",
    verbose=True,
    **kwargs,
):
    """Wrapper function that generates wordclouds
    ** Inputs **

    ** Returns **
    wordclouds as plots
    """

    fontname, fontpath = _configure_font(fontpath=fontpath)
    plt.rcParams["font.family"] = fontname

    # for individual masked wordclouds
    masked_wc_file_format = fig_filename_format + "_{}_masked.png"
    for k, wc_args in wordclouds_args.items():
        mask_file = wc_args.get("mask_file", None)
        if mask_file and mask_dir:
            if verbose:
                print(f"Creating masked wordcloud #{k}")
            wc_args["fontpath"] = fontpath
            wc_args["dpi"] = dpi
            wc_args["colormap"] = colormap

            wc_file = masked_wc_file_format.format(k)
            fig_filepath = Path(fig_output_dir) / wc_file
            wc_args["fig_filepath"] = fig_filepath
            wc_args["mask_path"] = f"{mask_dir}/{mask_file}"
            wc_args["save"] = True
            create_wordcloud(**wc_args)

    if figsize is None:
        figsize = (nrows * 4, ncols * 5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    cnt = 0
    p = 1
    wc_file_format = fig_filename_format + "_{}.png"
    wc_page_file_format = fig_filename_format + "_p{}.png"
    num_clouds = len(wordclouds_args)
    for k, wc_args in wordclouds_args.items():
        r, c = divmod(cnt, ncols)

        if verbose:
            print(f"Creating wordcloud #{k}")

        wc_args["fontpath"] = fontpath
        wc_args["dpi"] = dpi
        wc_args["colormap"] = colormap
        wc_args["mask_path"] = None
        wc_args["fig"] = fig
        wc_args["ax"] = axes[r, c]
        wc_args["save"] = save_each

        wc_file = wc_file_format.format(k)
        fig_filepath = Path(fig_output_dir) / wc_file
        wc_args["fig_filepath"] = fig_filepath
        create_wordcloud(**wc_args)

        axes[r, c].set_title(
            wc_args["title"], fontsize=title_fontsize, color=title_color
        )
        cnt += 1
        if cnt == nrows * ncols:
            if save:
                wc_file = wc_page_file_format.format(p)
                fig_filepath = Path(fig_output_dir) / wc_file
                save_subplots(fig, fig_filepath, transparent=True, dpi=dpi, verbose=verbose)
            if k < num_clouds - 1:
                p += 1
                cnt = 0
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if save and cnt < nrows * ncols:
        while cnt < nrows * ncols:
            r, c = divmod(cnt, ncols)
            axes[r, c].set_visible(False)
            cnt += 1
        wc_file = wc_page_file_format.format(p)
        fig_filepath = Path(fig_output_dir) / wc_file
        save_subplots(fig, fig_filepath, transparent=True, dpi=dpi, verbose=verbose)


def savefig(fig_filepath, transparent=True, dpi=300, **kwargs):
    plt.savefig(fig_filepath, transparent=transparent, dpi=dpi, **kwargs)


def save_subplots(fig, fig_filepath, transparent=True, dpi=300, verbose=True):
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.00, hspace=0.00
    )  # make the figure look better
    fig.tight_layout()
    Path(fig_filepath).parent.mkdir(parents=True, exist_ok=True)
    fig_filepath = str(fig_filepath)
    plt.savefig(fig_filepath, transparent=transparent, dpi=dpi)
    if verbose:
        print(f"Saved {fig_filepath}")


def create_wordcloud(
    word_freq,
    fig=None,
    ax=None,
    save=False,
    fig_filepath=None,
    mask_path=None,
    contour_width=0,
    contour_color="steelblue",
    dpi=300,
    figsize=(10, 10),
    facecolor="k",
    fontpath=None,
    colormap="PuBu",
    verbose=True,
    **kwargs,
):
    """Wrapper function that generates individual wordclouds

    ** Inputs **
    fig, ax: obj -> pyplot objects from subplots method
    save: bool -> If the user would like to save the images

    ** Returns **
    wordclouds as plots"""
    from wordcloud import WordCloud

    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    # if not fontpath:
    fontpath, fontpath = _configure_font(fontpath=fontpath)

    if mask_path is not None and Path(mask_path).is_file():
        if verbose:
            print(f"Using mask {mask_path}")
        mask = np.array(Image.open(mask_path))
        wc = WordCloud(
            font_path=fontpath,
            background_color="white",
            colormap=colormap,
            mask=mask,
            width=mask.shape[1],
            height=mask.shape[0],
            contour_width=contour_width,
            contour_color=contour_color,
        )

    else:
        wc = WordCloud(font_path=fontpath, background_color="white", colormap=colormap)

    img = wc.generate_from_frequencies(word_freq)
    if ax is not None:
        ax.imshow(img, interpolation="bilinear")
        ax.axis("off")
    else:
        plt.figure(figsize=figsize, facecolor=facecolor, dpi=dpi)
        plt.imshow(img, interpolation="bilinear")
        plt.tight_layout(pad=0)
        plt.axis("off")

    if save and fig_filepath:
        Path(fig_filepath).parent.mkdir(parents=True, exist_ok=True)
        fig_filepath = str(fig_filepath)
        if verbose > 5:
            print(f"Saving wordcloud to {fig_filepath}")
        if fig is not None:
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig(fig_filepath, bbox_inches=extent.expanded(1.1, 1.2), dpi=dpi)
        else:
            wc.to_file(fig_filepath)
            plt.savefig(fig_filepath)
