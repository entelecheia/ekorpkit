import logging
import matplotlib.pyplot as plt
from pathlib import Path
from .base import _get_font_name

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def generate_wordclouds(
    wordcloud_args_list,
    fig_output_dir,
    fig_filename_format,
    title_fontsize=20,
    title_color="green",
    ncols=5,
    nrows=1,
    dpi=300,
    save=True,
    figsize=(20, 20),
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
    num_clouds = len(wordcloud_args_list)
    for i, wc_args in enumerate(wordcloud_args_list):
        r, c = divmod(cnt, ncols)
        k = i
        print(f"Creating wordcloud #{k}")
        wc_file = wc_file_format.format(k)
        fig_filepath = Path(fig_output_dir) / wc_file
        create_wordcloud(
            wc_args["word_freq"],
            fig,
            axes[r, c],
            save=False,
            fig_filepath=fig_filepath,
            fontname=fontname,
        )
        axes[r, c].set_title(
            wc_args["title"], fontsize=title_fontsize, color=title_color
        )
        cnt += 1
        if cnt == nrows * ncols:
            if save:
                wc_file = wc_page_file_format.format(p)
                fig_filepath = Path(fig_output_dir) / wc_file
                save_subplots(fig, fig_filepath, transparent=True, dpi=dpi)
            if i < num_clouds - 1:
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
    word_freq, fig, ax, save=False, fig_filepath=None, fontname=None, **kwargs
):
    """Wrapper function that generates individual wordclouds

    ** Inputs **
    fig, ax: obj -> pyplot objects from subplots method
    save: bool -> If the user would like to save the images

    ** Returns **
    wordclouds as plots"""
    from wordcloud import WordCloud

    if not fontname:
        fontname, _ = _get_font_name()
    wc = WordCloud(font_path=fontname, background_color="white")

    img = wc.generate_from_frequencies(word_freq)
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")
    if save and fig_filepath:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        Path(fig_filepath).parent.mkdir(parents=True, exist_ok=True)
        fig_filepath = str(fig_filepath)
        plt.savefig(fig_filepath, bbox_inches=extent.expanded(1.1, 1.2))
