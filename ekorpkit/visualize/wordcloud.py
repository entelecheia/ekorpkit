import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from .base import _get_font_name

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
    verbose=True,
    **kwargs,
):
    """Wrapper function that generates wordclouds
    ** Inputs **

    ** Returns **
    wordclouds as plots
    """

    fontname, _ = _get_font_name()
    plt.rcParams["font.family"] = fontname
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
        wc_file = wc_file_format.format(k)
        fig_filepath = Path(fig_output_dir) / wc_file
        mask_file = wc_args.get("mask_file", None)
        wc_args["fig"] = fig
        wc_args["ax"] = axes[r, c]
        wc_args["save"] = False if save_masked else save_each
        wc_args["fig_filepath"] = fig_filepath
        wc_args["fontname"] = fontname
        wc_args["dpi"] = dpi
        if mask_file and mask_dir:
            wc_args["mask_path"] = f"{mask_dir}/{mask_file}"
            wc_args["save"] = True
        create_wordcloud(**wc_args)

        axes[r, c].set_title(
            wc_args["title"], fontsize=title_fontsize, color=title_color
        )
        cnt += 1
        if cnt == nrows * ncols:
            if save:
                wc_file = wc_page_file_format.format(p)
                fig_filepath = Path(fig_output_dir) / wc_file
                save_subplots(fig, fig_filepath, transparent=True, dpi=dpi)
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
        save_subplots(fig, fig_filepath, transparent=True, dpi=dpi)


def savefig(fig_filepath, transparent=True, dpi=300, **kwargs):
    plt.savefig(fig_filepath, transparent=transparent, dpi=dpi, **kwargs)


def save_subplots(fig, fig_filepath, transparent=True, dpi=300):
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.00, hspace=0.00
    )  # make the figure look better
    fig.tight_layout()
    Path(fig_filepath).parent.mkdir(parents=True, exist_ok=True)
    fig_filepath = str(fig_filepath)
    plt.savefig(fig_filepath, transparent=transparent, dpi=dpi)


def create_wordcloud(
    word_freq,
    fig=None,
    ax=None,
    save=False,
    fig_filepath=None,
    fontname=None,
    mask_path=None,
    contour_width=0,
    contour_color="steelblue",
    dpi=300,
    figsize=(10, 10),
    facecolor="k",
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
    if not fontname:
        fontname, _ = _get_font_name()

    if mask_path is not None and Path(mask_path).is_file():
        save_masked = True
        if verbose:
            print(f"Using mask {mask_path}")
        mask = np.array(Image.open(mask_path))
        wc = WordCloud(
            font_path=fontname,
            background_color="white",
            mask=mask,
            width=mask.shape[1],
            height=mask.shape[0],
            contour_width=contour_width,
            contour_color=contour_color,
        )

    else:
        save_masked = False
        wc = WordCloud(font_path=fontname, background_color="white")

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
        if not save_masked:
            if verbose > 5:
                print("No mask provided, skipping saving")
            return
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
