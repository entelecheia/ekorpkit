import logging
import numpy as np
import pandas as pd
import random
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
from pydantic import BaseModel, validator, PrivateAttr


log = logging.getLogger(__name__)


class ImagineMode(str, Enum):
    """Diffuse mode"""

    GENERATE = "generate"
    INPAINT = "inpaint"
    STITCH = "stitch"


class SchedulerType(str, Enum):
    K_LMS = "k_lms"
    PNDM = "pndm"
    DDIM = "ddim"


class ImagineConfig(BaseModel):
    text_prompts: dict = None
    num_inference_steps: int = 100
    num_samples: int = 2
    num_images_per_prompt: int = 2
    init_image: str = None
    init_strength: Optional[float] = None
    init_max_pixels: int = 262144
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    increase_seed_by: Optional[int] = 1
    display_image: bool = True
    save_image: bool = True
    save_image_config: bool = True
    make_collage: bool = False
    display_collage: bool = False
    save_collage: bool = True

    def __init__(self, text_prompts, **kwargs):
        super().__init__(method=self.__class__.__name__.lower(), **kwargs)
        if isinstance(text_prompts, str):
            text_prompts = {0: [text_prompts]}
        elif isinstance(text_prompts, list):
            text_prompts = {0: text_prompts}
        self.text_prompts = text_prompts

    @validator("width")
    def width_divisible_by_8(cls, v):
        if v % 8 != 0:
            log.warning(f"Width {v} is not divisible by 8. Setting to {v // 8 * 8}")
        return (v // 8) * 8

    @validator("height")
    def height_divisible_by_8(cls, v):
        if v % 8 != 0:
            log.warning(f"Height {v} is not divisible by 8. Setting to {v // 8 * 8}")
        return (v // 8) * 8

    @validator("num_images_per_prompt")
    def num_images_per_prompt_less_than_num_samples_and_divisible(cls, v, values):
        num_samples = values["num_samples"]
        if v > num_samples:
            msg = (
                f"num_images_per_prompt {v} is greater than num_samples {num_samples}. "
            )
            msg += f"Setting to {num_samples}"
            log.warning(msg)
            v = num_samples
        if num_samples % v != 0:
            msg = f"num_samples {num_samples} is not divisible by num_images_per_prompt {v}. "
            msg += f"Setting to {num_samples // v * v}"
            log.warning(msg)
            v = num_samples // v * v
        return v

    @property
    def num_prompts(self):
        return len(self.text_prompts)

    @property
    def num_iterations(self):
        return int(self.num_samples / self.num_images_per_prompt)


class StableImagineConfig(ImagineConfig):
    mode: ImagineMode = ImagineMode.GENERATE
    scheduler: SchedulerType = SchedulerType.K_LMS
    NSFW_retry: int = 0
    mask_image: str = None
    inpaint_strength: float = 7.5


class CollageConfig(BaseModel):
    cols: int = -1
    max_images: int = 6
    padding: int = 5
    bg_color: str = "black"


class StableRunConfig(BaseModel):
    batch_name: str
    batch_num: int
    batch_dir: Path
    imagine: StableImagineConfig
    collage: CollageConfig = CollageConfig()
    image_ext: str = "png"
    _text_series: Optional[pd.Series] = PrivateAttr(None)

    @property
    def batch_file_prefix(self):
        return f"{self.batch_name}({self.batch_num})"

    def set_seed(self, seed: int):
        if seed is None or seed < 0:
            random.seed()
            seed = random.randint(0, 2**32 - 1)
        if self.imagine.seed is None:
            self.imagine.seed = seed
        else:
            log.info(f"Seed already set to {self.imagine.seed}")

    def get_sample_path(
        self, name: Union[int, str], seed: int = None, ext: str = None
    ) -> str:
        if ext is None:
            ext = self.image_ext
        if isinstance(name, int):
            name = f"{name:04d}"
        else:
            name = str(name)
        _seed = "" if seed is None else f"_{seed}"
        filename = f"{self.batch_file_prefix}_{name}{_seed}.{ext}"
        return str(self.batch_dir / filename)

    def save(self, img, name: Union[int, str], seed: int = None):
        img_filepath = self.get_sample_path(name, seed, ext=self.image_ext)
        if self.imagine.save_image:
            img.save(img_filepath)
            log.info(f"Saved image to {img_filepath}")
        if self.imagine.save_image_config:
            self.save_config(name, seed)
        return img_filepath

    def save_config(self, name: Union[int, str], seed: int = None):
        rc_filepath = self.get_sample_path(name, seed, ext="json")
        with open(rc_filepath, "w") as f:
            f.write(self.json(indent=4, sort_keys=True, ensure_ascii=False))
        log.info(f"Saved run config to {rc_filepath}")

    def get_prompt(self, sample_num: int = 0) -> List[str]:
        text_series = self.prompt_series
        if text_series is not None and sample_num >= len(text_series):
            text_prompt = text_series[-1]
        elif text_series is not None:
            text_prompt = text_series[sample_num]
        else:
            text_prompt = []
        return text_prompt

    @property
    def prompt_series(self):
        """Get the prompt series"""
        if self._text_series is not None:
            return self._text_series
        text_prompts = self.imagine.text_prompts
        text_series = (
            split_prompts(text_prompts, self.imagine.num_samples)
            if text_prompts
            else None
        )
        self._text_series = text_series
        return text_series


def split_prompts(prompts, max_frames):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        if isinstance(prompt, str):
            prompt = [prompt]
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series
