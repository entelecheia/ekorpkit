import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from .base import _configure_font
from PIL import Image, ImageDraw, ImageFont
from ekorpkit.io.file import get_filepaths


log = logging.getLogger(__name__)


def convert_image(
    img_file,
    show_filename=False,
    filename_offset=(5, 5),
    fontname=None,
    fontsize=12,
    fontcolor=None,
):
    img = Image.open(img_file)
    if show_filename:
        fontname, fontpath = _configure_font(
            set_font_for_matplot=False, fontname=fontname
        )
        if fontpath:
            font = ImageFont.truetype(fontpath, fontsize)
        else:
            font = None
        draw = ImageDraw.Draw(img)
        draw.text(
            filename_offset, os.path.basename(img_file), font=font, fill=fontcolor
        )

    img = img.convert("RGB")
    img = np.asarray(img)
    return img


def gallery(array, ncols=7):
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


def collage(
    image_filepaths=None,
    filename_patterns=None,
    base_dir=None,
    output_filepath=None,
    ncols=7,
    num_images=None,
    figsize=(30, 20),
    dpi=300,
    title=None,
    title_fontsize=12,
    show_filename=False,
    filename_offset=(5, 5),
    fontname=None,
    fontsize=12,
    fontcolor="#000",
    **kwargs,
):
    verbose = kwargs.get("verbose", False)
    if image_filepaths is None:
        image_filepaths = sorted(get_filepaths(filename_patterns, base_dir=base_dir))
    if not image_filepaths:
        log.warning("no images found")
        return

    img_arr = []
    for filepath in image_filepaths:
        img_arr.append(
            convert_image(
                filepath, show_filename, filename_offset, fontname, fontsize, fontcolor
            )
        )
    if num_images is not None:
        num_images = min(num_images, len(img_arr))
    else:
        num_images = len(img_arr)
    ncols = min(ncols, num_images)
    nrows = num_images // ncols
    num_images = nrows * ncols

    img_arr = img_arr[:num_images]
    array = np.array(img_arr)
    result = gallery(array, ncols=ncols)

    plt.figure(figsize=figsize)
    plt.imshow(result)
    plt.grid(False)
    plt.axis("off")
    if title is not None:
        plt.title(title, fontdict={"fontsize": title_fontsize})
    plt.show()
    if output_filepath is not None:
        if base_dir is not None:
            output_filepath = os.path.join(base_dir, output_filepath)
        plt.savefig(output_filepath, dpi=dpi)

    return output_filepath
