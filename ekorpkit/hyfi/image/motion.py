"""Motion image processing functions."""
import os
import subprocess
from pathlib import Path

from ..io.file import get_filepaths
from ..utils.logging import getLogger
from ..utils.notebook import display_image

logger = getLogger(__name__)


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
    """
    Create a GIF from a list of images or a list of filenames.
    """
    from PIL import Image

    logger.info(f"Making GIF from {filename_patterns}")
    if os.path.exists(output_filepath) and not force:
        logger.info(f"Skipping GIF creation, already exists: {output_filepath}")
        logger.info("If you want to re-create the GIF, set force=True")
    else:
        if image_filepaths is None:
            image_filepaths = sorted(
                get_filepaths(filename_patterns, base_dir=base_dir)
            )
        if not image_filepaths:
            logger.warning("no images found")
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
            logger.warning(f"No frames found for {filename_patterns}")

    if show and os.path.exists(output_filepath):
        display_image(data=open(output_filepath, "rb").read(), width=width)

    return output_filepath


def extract_frames(
    video_path, extract_nth_frame, extracted_frame_dir, frame_filename="%04d.jpg"
):
    """
    Extract frames from a video.
    """
    logger.info(f"Exporting Video Frames (1 every {extract_nth_frame})...")
    try:
        for f in Path(f"{extracted_frame_dir}").glob("*.jpg"):
            f.unlink()
    except FileNotFoundError:
        logger.info(f"No video frames found in {extracted_frame_dir}")
    vf = f"select=not(mod(n\,{extract_nth_frame}))"

    ffmpeg_path = "/usr/bin/ffmpeg"
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = "ffmpeg"
    if os.path.exists(video_path):
        subprocess.run(
            [
                ffmpeg_path,
                "-i",
                f"{video_path}",
                "-vf",
                f"{vf}",
                "-vsync",
                "vfr",
                "-q:v",
                "2",
                "-loglevel",
                "error",
                "-stats",
                f"{extracted_frame_dir}/{frame_filename}",
            ],
            stdout=subprocess.PIPE,
        ).stdout.decode("utf-8")
    else:
        logger.warning(
            f"WARNING!\n\nVideo not found: {video_path}.\nPlease check your video path."
        )


def create_video(
    base_dir, video_path, input_url, fps, start_number, vframes, force=False
):
    """
    Create a video from a list of images.
    """

    logger.info(f"Creating video from {input_url}")
    if os.path.exists(video_path) and not force:
        logger.info(f"Skipping video creation, already exists: {video_path}")
        logger.info("If you want to re-create the video, set force=True")
        return video_path

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
        video_path,
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
        print(f"The video is ready and saved to {video_path}")

    return video_path
