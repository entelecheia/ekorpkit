"""Collage of images.""" ""
import io
import logging
import os
import textwrap
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel

from .utils import get_image_font, load_image, load_images, scale_image

log = logging.getLogger(__name__)


class Collage(BaseModel):
    """Collage of images."""

    image: Image.Image
    width: int
    height: int
    ncols: int
    nrows: int
    filepath: str = None

    class Config:
        arbitrary_types_allowed = True


def collage(
    images_or_uris,
    collage_filepath=None,
    ncols=3,
    max_images=12,
    collage_width=1200,
    padding: int = 10,
    bg_color: str = "black",
    crop_to_min_size=False,
    show_filename=False,
    filename_offset=(5, 5),
    fontname=None,
    fontsize=12,
    fontcolor="#000",
    **kwargs,
) -> Collage:
    """
    Create a collage of images.
    """

    if not isinstance(images_or_uris, list):
        images_or_uris = [images_or_uris]
    images_or_uris = [
        uri if isinstance(uri, Image.Image) else str(uri) for uri in images_or_uris
    ]
    if len(images_or_uris) < 1:
        log.info("No images provided")
        return None

    if max_images is not None:
        max_images = min(max_images, len(images_or_uris))
    else:
        max_images = len(images_or_uris)
    if max_images < 1:
        raise ValueError("max_images must be greater than 0")
    if ncols is not None and ncols > max_images:
        ncols = max_images
    if ncols is not None and max_images % ncols != 0:
        max_images = (max_images // ncols) * ncols
    # calc number of columns and rows from max_images_per_collage
    if ncols is None or ncols > max_images or ncols < 1:
        ncols = max_images // 2
    log.info(
        f"Creating collage of {max_images} images with {ncols} columns from {len(images_or_uris)} images"
    )
    img_width = collage_width // ncols
    collage_width = ncols * img_width + padding * (ncols + 1)

    # load images
    images = load_images(
        images_or_uris[:max_images],
        resize_to_multiple_of=None,
        crop_to_min_size=crop_to_min_size,
        max_width=img_width,
        **kwargs,
    )
    filenames = [
        os.path.basename(image_or_uri) if isinstance(image_or_uri, str) else None
        for image_or_uri in images_or_uris[:max_images]
    ]
    # convert images
    images = [
        convert_image(
            image,
            show_filename=show_filename,
            filename=filename,
            filename_offset=filename_offset,
            fontname=fontname,
            fontsize=fontsize,
            fontcolor=fontcolor,
            return_as_array=False,
        )
        for image, filename in zip(images, filenames)
    ]

    collage = grid_of_images(images, ncols, padding, bg_color=bg_color)
    if collage_filepath is not None:
        collage_filepath = str(collage_filepath)
        os.makedirs(os.path.dirname(collage_filepath), exist_ok=True)
        collage.image.save(collage_filepath)
        collage.filepath = collage_filepath
        log.info(f"Saved collage to {collage_filepath}")
    return collage


def label_collage(
    collage: Collage,
    collage_filepath=None,
    title=None,
    title_fontsize=10,
    xlabel=None,
    ylabel=None,
    xticklabels=None,
    yticklabels=None,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    dpi=100,
    fg_fontcolor="white",
    bg_color="black",
    caption=None,
    **kwargs,
) -> str:
    """
    Create a collage of images.
    """
    figsize = (collage.width / dpi, collage.height / dpi)
    ncols, nrows = collage.ncols, collage.nrows

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    plt.imshow(np.array(collage.image))
    ax = plt.gca()
    plt.grid(False)
    if xlabel is None and ylabel is None:
        plt.axis("off")
    if title is not None:
        title = "\n".join(
            sum(
                [
                    textwrap.wrap(
                        t, width=int(collage.width / 15 * 12 / title_fontsize)
                    )
                    for t in title.split("\n")
                ],
                [],
            )
        )
        ax.set_title(title, fontsize=title_fontsize, color=fg_fontcolor)
    if xlabel is not None:
        # plt.xlabel(xlabel, fontdict={"fontsize": xlabel_fontsize})
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, color=fg_fontcolor)
    if ylabel is not None:
        # plt.ylabel(ylabel, fontdict={"fontsize": ylabel_fontsize})
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, color=fg_fontcolor)
    if xticklabels is not None:
        # get ncols number of xticks from xlim
        xlim = ax.get_xlim()
        xticks = np.linspace(xlim[0], xlim[1], ncols + 1)
        xticks = xticks - (xticks[1] - xticks[0]) / 2
        xticks[0] = xlim[0]
        ax.set_xticks(xticks, color=fg_fontcolor)
        xticklabels = [""] + xticklabels
        ax.set_xticklabels(xticklabels, fontsize=xlabel_fontsize, color=fg_fontcolor)
    if yticklabels is not None:
        # get nrows number of yticks from ylim
        ylim = ax.get_ylim()
        yticks = np.linspace(ylim[0], ylim[1], nrows + 1)
        yticks = yticks - (yticks[1] - yticks[0]) / 2
        yticks[0] = ylim[0]
        ax.set_yticks(yticks, color=fg_fontcolor)
        yticklabels = [""] + yticklabels
        ax.set_yticklabels(yticklabels, fontsize=ylabel_fontsize, color=fg_fontcolor)

    plt.tight_layout()
    if caption is not None:
        print(f"[{caption}]")
    img = fig2img(fig, dpi=dpi)
    img = scale_image(img, max_width=collage.width)
    plt.close()

    if collage_filepath is not None:
        collage_filepath = str(collage_filepath)
        os.makedirs(os.path.dirname(collage_filepath), exist_ok=True)
        # fig.savefig(collage_filepath, dpi=dpi, bbox_inches="tight", pad_inches=0)
        img.save(collage_filepath)
        # collage_image.save(collage_filepath)
        # log.info(f"Saved collage to {collage_filepath}")

    return Collage(
        image=img,
        filepath=collage_filepath,
        width=img.width,
        height=img.height,
        ncols=ncols,
        nrows=nrows,
    )


def grid_of_images(
    images: List[Image.Image],
    ncols: int = 3,
    padding: int = 10,
    bg_color: str = "black",
) -> Collage:
    """
    Create a grid of images.
    """
    nrows = len(images) // ncols
    assert len(images) == nrows * ncols
    width, height = images[0].size
    grid_width = ncols * width + padding * (ncols + 1)
    grid_height = nrows * height + padding * (nrows + 1)
    collage = Image.new("RGB", size=(grid_width, grid_height), color=bg_color)
    for j, image in enumerate(images):
        x = j % ncols
        y = j // ncols
        collage.paste(
            image, (x * width + padding * (x + 1), y * height + padding * (y + 1))
        )
    return Collage(
        image=collage,
        width=grid_width,
        height=grid_height,
        ncols=ncols,
        nrows=nrows,
    )


def fig2img(fig, dpi=300):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    return Image.open(buf)


def convert_image(
    image_or_uri,
    show_filename=False,
    filename=None,
    filename_offset=(5, 5),
    fontname=None,
    fontsize=12,
    fontcolor=None,
    return_as_array=False,
):
    """
    Convert an image to a PIL Image.
    """
    img = load_image(image_or_uri)
    if isinstance(image_or_uri, str) and filename is None:
        filename = os.path.basename(image_or_uri)
    if show_filename and filename is not None:
        font = get_image_font(fontname, fontsize)
        draw = ImageDraw.Draw(img)
        draw.text(filename_offset, filename, font=font, fill=fontcolor)

    # img = img.convert("RGB")
    if return_as_array:
        img = np.array(img)
    return img


def gallery(array, ncols=7):
    """
    Create a gallery of images from a numpy array.
    """
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (
        array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )

    return result
