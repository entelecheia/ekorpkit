import logging
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from tqdm.auto import tqdm
from PIL import Image, ImageDraw
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from .base import BaseModel
from .config import ImagineMode, StableImagineConfig, StableRunConfig, SchedulerType


log = logging.getLogger(__name__)


class StableDiffusion(BaseModel):
    def __init__(self, root_dir=None, config_name="default", **args):
        cfg = eKonf.compose(f"model/stable_diffusion={config_name}")
        cfg = eKonf.merge(cfg, args)
        super().__init__(root_dir=root_dir, **cfg)

        self.pipe = None
        self.inpaint_pipe = None
        self.generator = None

        if self.autoload:
            self.load()

    @property
    def hf_user_access_token(self):
        return self.config.hf_user_access_token or eKonf.osenv("HF_USER_ACCESS_TOKEN")

    def imagine(
        self,
        text_prompts=None,
        batch_name=None,
        batch_num=None,
        mode: ImagineMode = None,
        num_samples=None,
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
        log.info("> loading config...")
        config = self.load_config(
            batch_name=batch_name,
            batch_num=batch_num,
            imagine=args,
        )
        config_file = self.save_config(config)
        cfg = StableImagineConfig(**config.imagine)
        rc = StableRunConfig(
            batch_name=self.batch_name,
            batch_num=self.batch_num,
            batch_dir=self.batch_dir,
            imagine=cfg,
        )
        rc.set_seed(self.seed)

        results = {}
        sample_imagepaths = []
        with elapsed_timer(format_time=True) as elapsed:
            if cfg.mode == ImagineMode.GENERATE:
                sample_imagepaths = self.generate_images(rc)
            elif cfg.mode == ImagineMode.INPAINT:
                sample_imagepaths = self.inpaint_images(rc, return_including_init_image)
            elif cfg.mode == ImagineMode.STITCH:
                cfg.num_images_per_prompt = 1
                sample_imagepaths = self.generate_images(rc)
                stitched_image_path = self.stitch_images(sample_imagepaths, rc)
                results["stitched_image_path"] = stitched_image_path
                cfg.make_collage = False

            log.info(" >> elapsed time to imagine: {}".format(elapsed()))

        if len(sample_imagepaths) == 0:
            log.info(" >> no images generated")
            return results

        if cfg.make_collage and cfg.display_collage:
            eKonf.clear_output(wait=True)
            self.collage(image_filepaths=sample_imagepaths, resize_ratio=resize_ratio)

        results.update(
            {
                "image_filepaths": sample_imagepaths,
                "config_file": config_file,
            }
        )
        return results

    def generate_images(self, rc: StableRunConfig):
        # Generate images
        cfg = rc.imagine
        images = []
        image_num = 0
        with torch.autocast("cuda"):
            for i in tqdm(range(cfg.num_iterations)):
                log.info(f"> generating image {image_num+1}/{cfg.num_samples}")
                imgs = self.generating(
                    prompt=rc.get_prompt(i),
                    width=cfg.width,
                    height=cfg.height,
                    guidance_scale=cfg.guidance_scale,
                    num_images_per_prompt=cfg.num_images_per_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                    seed=cfg.seed,
                )
                for img in imgs:
                    img_path = rc.save(img, image_num, seed=cfg.seed)
                    if cfg.display_image:
                        eKonf.clear_output(wait=True)
                        eKonf.display(img)
                    images.append(img_path)
                    image_num += 1
                if cfg.increase_seed_by:
                    cfg.seed += cfg.increase_seed_by
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
        images = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images
        return images

    def inpaint_images(
        self,
        rc: StableRunConfig,
        return_including_init_image=False,
    ):
        # Inpaint images
        cfg = rc.imagine
        images = []
        image_num = 0

        init_image = eKonf.load_image(cfg.init_image)
        mask_image = eKonf.load_image(cfg.mask_image)

        with torch.autocast("cuda"):
            for i in tqdm(range(cfg.num_iterations)):
                log.info(f"> generating image {image_num+1}/{cfg.num_samples}")
                imgs = self.inpainting(
                    prompt=rc.get_prompt(i),
                    init_image=init_image,
                    mask_image=mask_image,
                    guidance_scale=cfg.inpaint_strength,
                    num_images_per_prompt=cfg.num_images_per_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                    seed=cfg.seed,
                )
                for img in imgs:
                    img_path = rc.save(img, image_num, seed=cfg.seed)
                    if cfg.display_image:
                        eKonf.clear_output(wait=True)
                        eKonf.display(img)
                    images.append(img_path)
                    image_num += 1
                if cfg.increase_seed_by:
                    cfg.seed += cfg.increase_seed_by
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

        images = self.inpaint_pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images
        return images

    def stitch_images(
        self,
        image_filepaths,
        rc: StableRunConfig,
        **kwargs,
    ):
        # Stitch images
        cfg = rc.imagine
        images = [
            eKonf.load_image(image_filepath) for image_filepath in image_filepaths
        ]

        with torch.autocast("cuda"):
            output = self.stitching(
                images,
                prompt=rc.get_prompt(0),
                width=cfg.width,
                height=cfg.height,
                inpaint_strength=cfg.inpaint_strength,
                num_images_per_prompt=cfg.num_images_per_prompt,
                num_inference_steps=cfg.num_inference_steps,
                seed=cfg.seed,
            )

        # add borders
        w, h = output.size
        panorama = Image.new("RGB", (w, 3 * h))
        panorama.paste(output, (0, h))
        img_path = rc.save(panorama, "panorama")
        if cfg.display_image:
            eKonf.clear_output(wait=True)
            eKonf.display(panorama)
        return img_path

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
            merged = self.inpaint_pipe(
                prompt=prompt,
                image=new,
                mask_image=msk,
                width=_width,
                height=_height,
                guidance_scale=inpaint_strength,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                **kwargs,
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
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            use_auth_token=self.hf_user_access_token,
            revision=cfg.revision,
            torch_dtype=torch.float16,
            cache_dir=self.path.cache_dir,
        ).to(cfg.device)

        cfg = self.model_config.inpaint
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
            use_auth_token=self.hf_user_access_token,
            revision=cfg.revision,
            torch_dtype=torch.float16,
            cache_dir=self.path.cache_dir,
        ).to(cfg.device)

    def load_config(self, batch_name=None, batch_num=None, **kwargs):
        """Load the settings"""
        config = super().load_config(
            batch_name=batch_name, batch_num=batch_num, **kwargs
        )

        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        return config

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


def set_scheduler(pipe, scheduler: SchedulerType):
    if scheduler == SchedulerType.DDIM:
        ddim = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe.scheduler = ddim
    elif scheduler == SchedulerType.K_LMS:
        lms = LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        pipe.scheduler = lms
    elif scheduler == SchedulerType.PNDM:
        pndm = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        pipe.scheduler = pndm
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")
