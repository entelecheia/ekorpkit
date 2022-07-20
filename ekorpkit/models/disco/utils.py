import os
import logging
import hashlib
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from ekorpkit.io.fetch.web import web_download, web_download_unzip


log = logging.getLogger(__name__)


def _download_models(
    name,
    path,
    archive_path,
    link,
    link_fb=None,
    SHA=None,
    check_model_SHA=False,
    zip_path=None,
    unzip=False,
    **kwargs,
):

    if os.path.exists(path) and check_model_SHA and SHA:
        log.info(f"Checking {name} File")
        with open(path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == SHA:
            log.info(f"Model {name} SHA matches")
        else:
            log.info(f"Model {name} SHA doesn't match, redownloading...")
            os.remove(path)
            _download_models(
                name, path, archive_path, link, link_fb, SHA, check_model_SHA
            )
    elif os.path.exists(path):
        log.info(
            f"Model {name} already downloaded, set check_model_SHA to true if the file is corrupt"
        )
    elif os.path.exists(archive_path) and check_model_SHA and SHA:
        log.info(f"Checking {name} File")
        with open(archive_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == SHA:
            log.info(f"Model {name} SHA matches")
            log.info(f"Copying {name} File to {path} from {archive_path}")
            shutil.copyfile(archive_path, path)
        else:
            log.info(f"Model {name} SHA doesn't match, redownloading...")
            os.remove(archive_path)
            _download_models(name, path, archive_path, link, link_fb)
    elif os.path.exists(archive_path):
        log.info(f"Copying {name} File to {path} from {archive_path}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive_path, path)
    else:
        if unzip:
            web_download_unzip(link, zip_path)
        else:
            web_download(link, archive_path)
        if os.path.exists(archive_path):
            log.info(f"Copying {name} File to {path} from {archive_path}")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive_path, path)
        elif not os.path.exists(archive_path) and link_fb:
            log.info("First URL Failed using FallBack")
            _download_models(name, path, archive_path, link_fb)
        else:
            log.warning(f"{name} File not found")


def move_files(start_num, end_num, old_folder, new_folder, batch_name, batch_num):
    for i in range(start_num, end_num):
        old_file = old_folder + f"/{batch_name}({batch_num})_{i:04}.png"
        new_file = new_folder + f"/{batch_name}({batch_num})_{i:04}.png"
        os.rename(old_file, new_file)


def split_prompts(prompts, max_frames):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        if isinstance(prompt, str):
            prompt =[prompt]
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.

    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.

    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re

    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()["frame"])
        param = match_object.groupdict()["param"]
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def get_inbetweens(key_frames, max_frames, interp_spline, integer=False):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.

    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.

    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = interp_spline

    if interp_method == "Cubic" and len(key_frames.items()) <= 3:
        interp_method = "Quadratic"

    if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
        interp_method = "Linear"

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[
        key_frame_series.last_valid_index()
    ]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction="both"
    )
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series
