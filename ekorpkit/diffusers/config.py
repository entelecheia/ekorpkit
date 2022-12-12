import logging
import numpy as np
import pandas as pd
import random
from enum import Enum
from pathlib import Path
from typing import Optional, Union, List
from pydantic import BaseModel, validator, PrivateAttr
from ekorpkit.config import BaseBatchConfig
from ekorpkit import eKonf


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
    clear_output: bool = True
    display_image: bool = True
    save_image: bool = True
    save_image_config: bool = True
    display_collage: bool = False
    save_collage: bool = True
    image_ext: str = "png"
    _text_series: Optional[pd.Series] = PrivateAttr(None)

    def __init__(self, **values):
        text_prompts = values.get("text_prompts")
        if isinstance(text_prompts, str):
            text_prompts = {0: [text_prompts]}
        elif isinstance(text_prompts, list):
            text_prompts = {0: text_prompts}
        values["text_prompts"] = text_prompts
        super().__init__(**values)

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
        text_prompts = self.text_prompts
        text_series = (
            split_prompts(text_prompts, self.num_samples) if text_prompts else None
        )
        self._text_series = text_series
        return text_series

    def get_str_prompt(self):
        return get_str_prompt(self.text_prompts)


class StableImagineConfig(ImagineConfig):
    mode: ImagineMode = ImagineMode.GENERATE
    scheduler: SchedulerType = SchedulerType.K_LMS
    NSFW_retry: int = 0
    mask_image: str = None
    inpaint_strength: float = 7.5


class CollageConfig(BaseModel):
    clear_output = True
    show_prompt = True
    ncols = 3
    max_images = 12
    collage_width = 1200
    padding: int = 5
    bg_color: str = "black"
    crop_to_min_size = False
    show_filename = False
    filename_offset = (5, 5)
    fontname: str = None
    fontsize: int = 12
    fontcolor: str = "#000"


class AnimMode(str, Enum):
    """Animation mode"""

    NONE = "None"
    ANIM_2D = "2D"
    ANIM_3D = "3D"
    VIDEO_INPUT = "Video Input"


class BatchConfig(BaseBatchConfig):
    max_display_image_width: int = 1000
    collage_dirname = "collages"
    collage_filesfx = "collage.png"

    def get_imagine_config(self, pair_args: dict = None, **imagine_args):
        args = imagine_args.copy()
        if pair_args is not None:
            args["num_samples"] = self.num_samples
            args.update(pair_args)
        return args

    @property
    def collage_dir(self):
        collage_dir = self.batch_dir / self.collage_dirname
        collage_dir.mkdir(parents=True, exist_ok=True)
        return collage_dir

    @property
    def collage_filename(self):
        return f"{self.file_prefix}_{self.collage_filesfx}"

    @property
    def collage_filepath(self):
        return self.collage_dir / self.collage_filename


class BatchImagineResult(BaseModel):
    run_configs: dict
    collage_filepaths: dict
    batch_prompts: dict


class ImagineResult(BaseModel):
    batch_num: int = None
    batch_args: dict = None
    config_file: str = None
    image_filepaths: list = []
    stitched_image_path: str = None


class BatchRunConfig(BaseModel):
    batch_name: str
    batch_run_pair: dict
    imagine_results: list = []
    run_config_path: Path = None

    def __init__(self, **values):
        run_config_path = values.get("run_config_path")
        if run_config_path is not None:
            run_config_path = Path(run_config_path)
            if run_config_path.exists():
                run_config = eKonf.load(run_config_path)
                values.update(eKonf.to_dict(run_config))
        values["run_config_path"] = run_config_path
        super().__init__(**values)

    @validator("imagine_results")
    def validate_imagine_results(cls, v):
        return [ImagineResult(**d) if isinstance(d, dict) else d for d in v]

    def append(self, batch_args, imagine_result: ImagineResult):
        imagine_result.batch_args = batch_args
        self.imagine_results.append(imagine_result)

    @property
    def collage_dir(self):
        return self.run_config_path.parent.parent

    def collage_filename(self, suffix=None):
        sfx = f"_{suffix}" if suffix else ""
        return self.run_config_path.name.replace(".yaml", f"{sfx}.png")

    def collage_filepath(self, suffix=None):
        return self.collage_dir / self.collage_filename(suffix)


class BatchImagineConfig(BatchConfig):
    batch_run_params: dict = None
    batch_run_pairs: list = None
    num_samples: int = 1
    num_images_per_prompt: int = 1
    clear_output: bool = True
    display_collage: bool = False
    save_collage: bool = True
    run_config_file: str = "run_configs.yaml"
    batch_run_name_delimiter: str = "."
    run_pair_name_delimiter: str = "-"

    @validator("batch_run_params")
    def batch_run_params_is_dict(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("batch_run_params must be a dict")
        if len(v) < 1:
            raise ValueError("batch_run_params must have at least 1 element")
        for _, val in v.items():
            if not isinstance(v, (list, tuple)):
                val = [val]
        return v

    @validator("batch_run_pairs")
    def batch_run_pairs_is_list_of_tuples(cls, v, values):
        if v is None:
            log.info("No batch_run_pairs provided. Using each key in batch_run_params")
            batch_run_params = values.get("batch_run_params")
            if batch_run_params:
                batch_run_params = cls.batch_run_params_is_dict(batch_run_params)
                v = [[arg_name] for arg_name in batch_run_params.keys()]
            else:
                v = [[]]
        if not isinstance(v, list):
            raise ValueError("batch_run_pairs must be a list")
        # Check that each element is a list or tuple, if an element is a string, convert to list
        for i, pair in enumerate(v):
            if isinstance(pair, str):
                v[i] = [pair]
            elif not isinstance(pair, (list, tuple)):
                raise ValueError("batch_run_pairs must be a list of lists or tuples")
        return v

    @property
    def batch_run_configs(self):
        for run_pair in self.batch_run_pairs:
            batch_run_pair = {name: self.batch_run_params[name] for name in run_pair}
            batch_run_name = (
                self.batch_name
                + self.batch_run_name_delimiter
                + self.run_pair_name_delimiter.join(run_pair)
            )
            yield BatchRunConfig(
                batch_name=batch_run_name,
                batch_run_pair=batch_run_pair,
                results=[],
            )

    def get_run_args(self, batch_run_pair):
        return eKonf.dict_product(batch_run_pair)

    def get_imagine_config(self, arg_pair: dict = None, **imagine_args):
        args = imagine_args.copy()
        if arg_pair is not None:
            # args["num_samples"] = self.num_samples
            # args["increase_seed_by"] = 0
            if "text_prompts" not in arg_pair:
                args["display_collage"] = False
            args.update(arg_pair)
        return args

    def save_run_config(self, run_config: BatchRunConfig):
        """Save the settings"""
        batch_name = run_config.batch_name
        filename = f"{batch_name}({self.batch_num})_{self.run_config_file}"
        config_path = self.config_dir / filename
        log.info(f"Saving batch run config to {config_path}")
        eKonf.save(run_config.dict(), config_path)
        return config_path

    def load_run_config(self, run_config_path):
        """Load the settings"""
        log.info(f"Loading batch run config from {run_config_path}")
        run_config = eKonf.load(run_config_path)
        return BatchRunConfig(**run_config)


class RunConfig(BaseModel):
    batch: BatchConfig
    imagine: ImagineConfig
    collage: CollageConfig = CollageConfig()

    def __init__(self, **values):
        super().__init__(**values)
        self.set_seed()

    @property
    def batch_file_prefix(self):
        return self.batch.file_prefix

    @property
    def batch_dir(self):
        return self.batch.batch_dir

    @property
    def image_ext(self):
        return self.imagine.image_ext

    def set_seed(self, seed: int = None):
        if seed is None:
            seed = self.batch.seed
        if seed is None or seed < 0:
            random.seed()
            seed = random.randint(0, 2**32 - 1)
        if self.imagine.seed is None:
            log.info(f"Setting imagine seed to {seed}")
            self.imagine.seed = seed
        else:
            log.info(f"Seed already set to {self.imagine.seed}")

    @property
    def image_filepattern(self):
        return f"{self.batch_file_prefix}*{self.image_ext}"

    @property
    def image_filepaths(self):
        return sorted(self.batch_dir.glob(self.image_filepattern))

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
        filename = f"{self.batch.file_prefix}_{name}{_seed}.{ext}"
        return str(self.batch.batch_dir / filename)

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


class StableRunConfig(RunConfig):
    imagine: StableImagineConfig


def split_prompts(prompts, max_frames):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        if isinstance(prompt, str):
            prompt = [prompt]
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


def get_str_prompt(prompts):
    prompts = eKonf.to_dict(prompts)
    if isinstance(prompts, str):
        return prompts
    elif isinstance(prompts, list):
        return ", ".join(prompts)
    elif isinstance(prompts, dict):
        if 0 in prompts:
            prompts = prompts[0]
        else:
            prompts = prompts.values()[0]
        return get_str_prompt(prompts)
