import io
import numpy as np
from PIL import Image, ImageFont
from .base import get_plot_font
from ekorpkit.io.file import read


def scale_image(
    image: Image.Image,
    max_width: int = None,
    max_height: int = None,
    max_pixels: int = None,
    scale: float = 1.0,
    resize_to_multiple_of: int = 8,
    resample: int = Image.LANCZOS,
) -> Image.Image:
    """Scale image to have at most `max_pixels` pixels."""
    w, h = image.size

    if max_width is None and max_height is not None:
        max_width = int(w * max_height / h)
    elif max_height is None and max_width is not None:
        max_height = int(h * max_width / w)

    if max_width is not None and max_height is not None:
        max_pixels = max_width * max_height
    if max_pixels is not None:
        scale = np.sqrt(max_pixels / (w * h))

    max_width = int(w * scale)
    max_height = int(h * scale)
    if resize_to_multiple_of is not None:
        max_width = (max_width // resize_to_multiple_of) * resize_to_multiple_of
        max_height = (max_height // resize_to_multiple_of) * resize_to_multiple_of

    if scale < 1.0 or w > max_width or h > max_height:
        image = image.resize((max_width, max_height), resample=resample)
    return image


def load_image(
    image_or_uri,
    max_width: int = None,
    max_height: int = None,
    max_pixels: int = None,
    scale: float = 1.0,
    resize_to_multiple_of: int = None,
    crop_box=None,
    mode="RGB",
    **kwargs
) -> Image.Image:
    from PIL import Image

    if isinstance(image_or_uri, Image.Image):
        img = image_or_uri.convert(mode)
    else:
        img = Image.open(io.BytesIO(read(image_or_uri, **kwargs))).convert(mode)
    img = scale_image(
        img,
        max_width=max_width,
        max_height=max_height,
        max_pixels=max_pixels,
        scale=scale,
        resize_to_multiple_of=resize_to_multiple_of,
    )
    if crop_box is not None:
        img = img.crop(crop_box)
    return img


def load_images(
    images_or_uris,
    max_width=None,
    max_height=None,
    max_pixels=None,
    scale=1.0,
    resize_to_multiple_of: int = None,
    crop_to_min_size=False,
    mode="RGB",
    **kwargs
):
    imgs = [
        load_image(
            image_or_uri,
            max_width=max_width,
            max_height=max_height,
            max_pixels=max_pixels,
            scale=scale,
            resize_to_multiple_of=resize_to_multiple_of,
            mode=mode,
            **kwargs
        )
        for image_or_uri in images_or_uris
    ]
    if crop_to_min_size:
        min_width = min(img.width for img in imgs)
        min_height = min(img.height for img in imgs)
        if resize_to_multiple_of is not None:
            min_width = (min_width // resize_to_multiple_of) * resize_to_multiple_of
            min_height = (min_height // resize_to_multiple_of) * resize_to_multiple_of
        imgs = [img.crop((0, 0, min_width, min_height)) for img in imgs]

    return imgs


def get_image_font(fontname=None, fontsize=12):
    fontname, fontpath = get_plot_font(set_font_for_matplot=False, fontname=fontname)
    if fontpath:
        font = ImageFont.truetype(fontpath, fontsize)
    else:
        font = None
    return font
