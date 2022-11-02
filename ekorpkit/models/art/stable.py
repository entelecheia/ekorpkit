import os
import logging
import random
import torch
from glob import glob
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from tqdm.auto import tqdm
from enum import Enum
from .base import BaseModel
from PIL import Image, ImageDraw
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.models.disco.utils import split_prompts
from .settings import DiffuseMode, ImageFormat, Prompt, RunSettings, SchedulerType

log = logging.getLogger(__name__)


class StableDiffusion(BaseModel):
    def __init__(self, **args):
        super().__init__(**args)
        self._device = self.args.device
        if self.args.hf_user_access_token:
            self.hf_user_access_token = self.args.hf_user_access_token
        else:
            self.hf_user_access_token = eKonf.osenv("HF_USER_ACCESS_TOKEN")
        if self.auto.load:
            self.load()

    def imagine(
        self,
        text_prompts=None,
        mode: DiffuseMode = None,
        num_samples=None,
        show_collage=True,
        resize_ratio=0.5,
        return_including_init_image=False,
        **args,
    ):
        """Generate images"""

        if text_prompts is not None:
            args.update(dict(text_prompts=text_prompts))
        if mode is not None:
            args.update(dict(mode=mode))
        if num_samples is not None:
            args.update(dict(num_samples=num_samples))
        log.info("> loading settings...")
        args = self.load_config(**args)

        prompts = self._get_prompt_series(args)
        batch_dir = self._output.batch_dir
        results = {}
        sample_imagepaths = []
        with elapsed_timer(format_time=True) as elapsed:
            if args.mode == DiffuseMode.GENERATE:
                sample_imagepaths = self.generate_images(prompts, batch_dir, args)
            elif args.mode == DiffuseMode.INPAINT:
                sample_imagepaths = self.inpaint_images(
                    prompts, batch_dir, args, return_including_init_image
                )
            elif args.mode == DiffuseMode.STITCH:
                sample_imagepaths = self.generate_images(prompts, batch_dir, args)
                stitched_image_path = self.stitch_images(
                    sample_imagepaths, prompts, batch_dir, args
                )
                results["stitched_image_path"] = stitched_image_path
                if show_collage:
                    eKonf.clear_output(wait=True)
                    eKonf.display_image(stitched_image_path)
                show_collage = False

            log.info(" >> elapsed time to imagine: {}".format(elapsed()))

        if len(sample_imagepaths) == 0:
            log.info(" >> no images generated")
            return results

        if show_collage:
            eKonf.clear_output(wait=True)
            self.collage(image_filepaths=sample_imagepaths, resize_ratio=resize_ratio)

        results.update(
            {
                "image_filepaths": sample_imagepaths,
                "config_file": self.save_config(args),
                "config": eKonf.to_dict(args),
            }
        )
        return results

    def generate_images(self, prompts, batch_dir, cfg, **kwargs):
        # Generate images
        images = []
        sample_no = 0
        with torch.autocast("cuda"):
            for i in tqdm(range(cfg.num_samples)):
                log.info(f"> generating image {i+1}/{cfg.num_samples}")
                imgs = self.generating(
                    prompt=self._get_prompt(prompts, i),
                    width=cfg.width,
                    height=cfg.height,
                    guidance_scale=cfg.guidance_scale,
                    num_images_per_prompt=cfg.num_images_per_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                )
                for img in imgs:
                    filename = f"{cfg.batch_name}({cfg.batch_num})_{sample_no:04}.png"
                    _img_path = os.path.join(batch_dir, filename)
                    img.save(_img_path)
                    eKonf.clear_output(wait=True)
                    eKonf.display_image(_img_path)
                    images.append(_img_path)
                    sample_no += 1
        return images

    def generating(
        self,
        prompt,
        width=512,
        height=512,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        num_inference_steps=50,
        **kwargs,
    ):
        """Generate images from a prompt"""
        if height % 8 != 0 or width % 8 != 0:
            height = (height // 8) * 8
            width = (width // 8) * 8

        images = self._generate(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            generator=self._generator,
            **kwargs,
        ).images
        return images

    def inpaint_images(
        self,
        prompts,
        batch_dir,
        cfg,
        return_including_init_image=False,
        **kwargs,
    ):
        # Inpaint images
        images = []
        sample_no = 0

        init_image = eKonf.load_image(cfg.init_image)
        mask_image = eKonf.load_image(cfg.mask_image)

        with torch.autocast("cuda"):
            for i in tqdm(range(cfg.num_samples)):
                log.info(f"> inpainting image {i+1}/{cfg.num_samples}")
                imgs = self.inpainting(
                    prompt=self._get_prompt(prompts, i),
                    init_image=init_image,
                    mask_image=mask_image,
                    guidance_scale=cfg.inpaint_strength,
                    num_images_per_prompt=cfg.num_images_per_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                )
                for img in imgs:
                    filename = f"{cfg.batch_name}({cfg.batch_num})_{sample_no:04}.png"
                    _img_path = os.path.join(batch_dir, filename)
                    img.save(_img_path)
                    eKonf.clear_output(wait=True)
                    eKonf.display_image(_img_path)
                    images.append(_img_path)
                    sample_no += 1
        # insert initial image in the list so we can compare side by side
        if return_including_init_image:
            images.insert(0, cfg.init_image)
        return images

    def inpainting(
        self,
        prompt,
        init_image,
        mask_image,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        num_inference_steps=50,
        **kwargs,
    ):
        """Inpainting"""
        width, height = init_image.size
        if height % 8 != 0 or width % 8 != 0:
            height = (height // 8) * 8
            width = (width // 8) * 8
            init_image = init_image.crop((0, 0, width, height))
            mask_image = mask_image.crop((0, 0, width, height))

        images = self._inpaint(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            generator=self._generator,
            **kwargs,
        ).images
        return images

    def stitch_images(
        self,
        image_filepaths,
        prompts,
        batch_dir,
        cfg,
        **kwargs,
    ):
        # Stitch images
        images = [
            eKonf.load_image(image_filepath) for image_filepath in image_filepaths
        ]

        with torch.autocast("cuda"):
            output = self.stitching(
                images,
                prompt=self._get_prompt(prompts, 0),
                width=cfg.width,
                height=cfg.height,
                inpaint_strength=cfg.inpaint_strength,
                num_images_per_prompt=cfg.num_images_per_prompt,
                num_inference_steps=cfg.num_inference_steps,
            )

        filename = f"{cfg.batch_name}({cfg.batch_num})_panorama.png"
        _img_path = os.path.join(batch_dir, filename)

        # add borders
        w, h = output.size
        panorama = Image.new("RGB", (w, 3 * h))
        panorama.paste(output, (0, h))
        panorama.save(_img_path)
        return _img_path

    def stitching(
        self,
        images,
        prompt,
        width=512,
        height=512,
        inpaint_strength=7.5,
        num_images_per_prompt=1,
        num_inference_steps=50,
        **kwargs,
    ):
        """Stitching images"""

        output = images[0]

        for img in tqdm(images[1:] + [images[0]]):
            w, h = output.size
            new = Image.new("RGB", (width, height))
            new.paste(output, (-w + width // 2, 0))
            new.paste(img, (width // 2, 0))

            msk = Image.new("L", (width, height))
            drw = ImageDraw.Draw(msk)
            drw.rectangle((width // 4, 0, 3 * width // 4, height), fill=255)
            _width, _height = new.size
            merged = self._inpaint(
                prompt=prompt,
                image=new,
                mask_image=msk,
                width=_width,
                height=_height,
                guidance_scale=inpaint_strength,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                generator=self._generator,
            ).images[0]

            new = Image.new("RGB", (w + width, height))
            new.paste(output, (0, 0))
            new.paste(merged, (w - width // 2, 0))
            img = img.crop((width // 2, 0, width, height))
            new.paste(img, (w + width // 2, 0))
            output = new

        w, h = output.size
        output = output.crop((width // 2, 0, w - width // 2, h))
        return output

    def load_models(self):
        torch.set_grad_enabled(False)

        cfg = self.model_config.generate
        self._generate = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            use_auth_token=self.hf_user_access_token,
            revision=cfg.revision,
            torch_dtype=torch.float16,
            cache_dir=cfg.cache_dir,
        ).to(cfg.device)

        cfg = self.model_config.inpaint
        self._inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            use_auth_token=self.hf_user_access_token,
            revision=cfg.revision,
            torch_dtype=torch.float16,
            cache_dir=cfg.cache_dir,
        ).to(cfg.device)

    def load_config(self, batch_name=None, batch_num=None, **args):
        """Load the settings"""
        args = super().load_config(batch_name=batch_name, batch_num=batch_num, **args)

        _mode = DiffuseMode.GENERATE.value
        for mode in DiffuseMode:
            if mode.value.lower() == args.mode.lower():
                _mode = mode.value
                break
        args.mode = _mode

        text_prompts = eKonf.to_dict(args.text_prompts)
        if isinstance(text_prompts, str):
            text_prompts = {0: [text_prompts]}
        elif isinstance(text_prompts, list):
            text_prompts = {0: text_prompts}
        args.text_prompts = text_prompts

        if args.height % 8 != 0 or args.width % 8 != 0:
            args.height = (args.height // 8) * 8
            args.width = (args.width // 8) * 8

        if args.set_seed == "random_seed":
            random.seed()
            args.seed = random.randint(0, 2 ** 32 - 1)
        else:
            args.seed = int(args.set_seed)
        self._generator = torch.Generator(device=self._device).manual_seed(args.seed)

        batch_arg_file = os.path.join(
            self._output.batch_dir, f"{args.batch_name}(*)_settings.yaml"
        )
        if args.resume_run:
            if args.run_to_resume == "latest":
                try:
                    args.batch_num
                except AttributeError:
                    args.batch_num = len(glob(batch_arg_file)) - 1
            else:
                args.batch_num = int(args.run_to_resume)
        else:
            args.batch_num = len(glob(batch_arg_file))

        return args

    def _get_prompt_series(self, args, max_samples=None):
        """Get the prompt series"""
        if max_samples is None:
            max_samples = args.num_samples
        text_prompts = eKonf.to_dict(args.text_prompts)
        text_series = split_prompts(text_prompts, max_samples) if text_prompts else None
        return text_series

    def _get_prompt(self, text_series, sample_num):
        if text_series is not None and sample_num >= len(text_series):
            text_prompt = text_series[-1]
        elif text_series is not None:
            text_prompt = text_series[sample_num]
        else:
            text_prompt = []

        return text_prompt

    def compare_images(self, images, rows=1, cols=None, resize_ratio=1.0):
        import PIL

        if cols is None:
            cols = len(images) // rows

        if resize_ratio != 1.0:
            images = [
                img.resize(
                    (int(img.width * resize_ratio), int(img.height * resize_ratio))
                )
                for img in images
            ]
        w, h = images[0].size
        grid = PIL.Image.new("RGB", size=(cols * w, rows * h))
        # grid_w, grid_h = grid.size

        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid
