import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ekorpkit.io.file import get_filepaths


def convert_image(img_file):
    img = Image.open(img_file).convert("RGB")
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
    **kwargs,
):
    verbose = kwargs.get("verbose", False)
    if image_filepaths is None:
        image_filepaths = get_filepaths(filename_patterns, base_dir=base_dir)
    if not image_filepaths:
        raise ValueError("No files found")

    img_arr = []
    for filepath in image_filepaths:
        img_arr.append(convert_image(filepath))
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
    plt.show()
    if output_filepath is not None:
        if base_dir is not None:
            output_filepath = os.path.join(base_dir, output_filepath)
        plt.savefig(output_filepath, dpi=dpi)
