import os
import logging
import subprocess
from ekorpkit.io.file import get_filepaths
from ekorpkit.base import _display, _display_image


log = logging.getLogger(__name__)


def make_gif(
    image_filepaths=None,
    filename_patterns=None,
    base_dir=None,
    output_filepath=None,
    duration=100,
    loop=0,
    width=None,
    optimize=True,
    quality=50,
    show=False,
    force=False,
    **kwargs,
):
    from PIL import Image

    log.info(f"Making GIF from {filename_patterns}")
    if os.path.exists(output_filepath) and not force:
        log.info(f"Skipping GIF creation, already exists: {output_filepath}")
        log.info("If you want to re-create the GIF, set force=True")
    else:
        if image_filepaths is None:
            image_filepaths = sorted(
                get_filepaths(filename_patterns, base_dir=base_dir)
            )
        if not image_filepaths:
            log.warning("no images found")
            return
        frames = [Image.open(image) for image in image_filepaths]
        if len(frames) > 0:
            frame_one = frames[0]
            frame_one.save(
                output_filepath,
                format="GIF",
                append_images=frames,
                save_all=True,
                duration=duration,
                loop=loop,
                optimize=optimize,
                quality=quality,
            )
            print(f"Saved GIF to {output_filepath}")
        else:
            log.warning(f"No frames found for {filename_patterns}")

    if show and os.path.exists(output_filepath):
        _display_image(data=open(output_filepath, "rb").read(), width=width)

    return output_filepath


def create_video(
    base_dir, mp4_path, input_url, fps, start_number, vframes, force=False
):

    log.info(f"Creating video from {input_url}")
    if os.path.exists(mp4_path) and not force:
        log.info(f"Skipping video creation, already exists: {mp4_path}")
        log.info("If you want to re-create the video, set force=True")
        return mp4_path

    cmd = [
        "ffmpeg",
        "-y",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        input_url,
        "-frames:v",
        str(vframes),
        "-c:v",
        "libx264",
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",
        "-preset",
        "veryslow",
        mp4_path,
    ]

    process = subprocess.Popen(
        cmd,
        cwd=f"{base_dir}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    else:
        print(f"The video is ready and saved to {mp4_path}")

    return mp4_path
