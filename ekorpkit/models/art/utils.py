from typing import List, Optional
from PIL import Image


def make_collages(
    images: List[Image.Image],
    cols: Optional[int] = None,
    max_per_collage: int = 6,
    padding: int = 10,
    bg_color: str = "black",
) -> List[Image.Image]:
    """Make a list of image grids from a list of images."""

    # calc columns and rows from max_per_collage with max cols
    if cols is None or cols > max_per_collage or cols < 1:
        cols = max_per_collage // 2
        rows = max_per_collage // cols
    else:
        rows = max_per_collage // cols

    # split images into sublists of max_per_collage
    image_sublists = [
        images[i : i + max_per_collage] for i in range(0, len(images), max_per_collage)
    ]

    width, height = images[0].size

    total_w = cols * width + padding * (cols + 1)
    total_h = rows * height + padding * (rows + 1)

    collages = []
    for image_sublist in image_sublists:
        collage = Image.new("RGB", size=(total_w, total_h), color=bg_color)
        for j, image in enumerate(image_sublist):
            x = j % cols
            y = j // cols
            collage.paste(
                image, (x * width + padding * (x + 1), y * height + padding * (y + 1))
            )
        collages.append(collage)

    return collages
