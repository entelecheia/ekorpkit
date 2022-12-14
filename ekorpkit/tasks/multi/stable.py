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
from ekorpkit.diffusers.base import BaseModel
from ekorpkit.diffusers.config import (
    ImagineMode,
    BatchConfig,
    StableImagineConfig,
    StableRunConfig,
    SchedulerType,
    CollageConfig,
    ImagineResult,
)


log = logging.getLogger(__name__)


class StableDiffusion(BaseModel):
    batch: BatchConfig = None
    imagine: StableImagineConfig = None
    collage: CollageConfig = None
    __pipes__ = {}

    def __init__(self, config_name: str = "stable.diffusion", **args):
        config_group = f"task={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)

        if self.autoload:
            self.load_diffusers()

    def generate(
        self,
        text_prompts=None,
        batch_name=None,
        batch_num=None,
        mode: ImagineMode = None,
        num_samples=None,
        return_including_init_image=False,
        **imagine_args,
    ):
        """Generate images"""

        if text_prompts is not None:
            imagine_args.update(dict(text_prompts=text_prompts))
        if mode is not None:
            imagine_args.update(dict(mode=mode))
        if num_samples is not None:
            imagine_args.update(dict(num_samples=num_samples))
        config = self.load_config(
            batch_name=batch_name,
            batch_num=batch_num,
            imagine=imagine_args,
        )
        config_file = self.save_config(config)
        rc = self.get_run_config(config)
        # run_cfg.set_seed()

        imagine_rst = ImagineResult(batch_num=rc.batch.batch_num)
        sample_imagepaths = []
        with elapsed_timer(format_time=True) as elapsed:
            if rc.imagine.mode == ImagineMode.GENERATE:
                sample_imagepaths = self.generate_images(rc)
            elif rc.imagine.mode == ImagineMode.INPAINT:
                sample_imagepaths = self.inpaint_images(rc, return_including_init_image)
            elif rc.imagine.mode == ImagineMode.STITCH:
                rc.imagine.num_images_per_prompt = 1
                sample_imagepaths = self.generate_images(rc)
                stitched_image_path = self.stitch_images(sample_imagepaths, rc)
                imagine_rst.stitched_image_path = stitched_image_path
                rc.imagine.save_collage = False
                rc.imagine.display_collage = False

            log.info(" >> elapsed time to imagine: {}".format(elapsed()))

        if len(sample_imagepaths) == 0:
            log.info(" >> no images generated")
            return imagine_rst

        if rc.imagine.save_collage or rc.imagine.display_collage:
            if rc.imagine.clear_output:
                eKonf.clear_output(wait=True)
            self.collage_images(
                images_or_uris=sample_imagepaths,
                save_collage=rc.imagine.save_collage,
                display_collage=rc.imagine.display_collage,
                **rc.collage.dict(),
            )

        imagine_rst.image_filepaths = sample_imagepaths
        imagine_rst.config_file = config_file
        return imagine_rst

    def get_run_config(self, config):
        self.batch = BatchConfig(**config.batch)
        self.imagine = StableImagineConfig(**config.imagine)
        self.collage = CollageConfig(**config.collage)
        rc = StableRunConfig(
            batch=self.batch, imagine=self.imagine, collage=self.collage
        )
        return rc

    def generate_images(self, rc: StableRunConfig):
        # Generate images
        cfg = rc.imagine
        images = []
        image_num = 0

        # if cfg.init_image is not None:
        #     init_image = eKonf.load_image(cfg.init_image)
        # else:
        #     init_image = None

        with torch.autocast("cuda"):
            for i in tqdm(range(cfg.num_iterations)):
                log.info(f"> generating image {image_num+1}/{cfg.num_samples}")
                imgs = self.generating(
                    prompt=cfg.get_prompt(i),
                    width=cfg.width,
                    height=cfg.height,
                    guidance_scale=cfg.guidance_scale,
                    num_images_per_prompt=cfg.num_images_per_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                    generator=self.get_generator(cfg.seed),
                )
                for img in imgs:
                    img_path = rc.save(img, image_num, seed=cfg.seed)
                    if cfg.display_image:
                        if cfg.clear_output:
                            eKonf.clear_output(wait=True)
                        if rc.batch.max_display_image_width is not None:
                            img = eKonf.scale_image(
                                img, max_width=rc.batch.max_display_image_width
                            )
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
        generator=None,
        **kwargs,
    ):
        """Generate images from a prompt"""
        pipe = self.get_pipe("generate")
        images = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
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
                    prompt=cfg.get_prompt(i),
                    init_image=init_image,
                    mask_image=mask_image,
                    guidance_scale=cfg.inpaint_strength,
                    num_images_per_prompt=cfg.num_images_per_prompt,
                    num_inference_steps=cfg.num_inference_steps,
                    generator=self.get_generator(cfg.seed),
                )
                for img in imgs:
                    img_path = rc.save(img, image_num, seed=cfg.seed)
                    if cfg.display_image:
                        if cfg.clear_output:
                            eKonf.clear_output(wait=True)
                        if rc.batch.max_display_image_width is not None:
                            img = eKonf.scale_image(
                                img, max_width=rc.batch.max_display_image_width
                            )
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
        generator=None,
        **kwargs,
    ):
        """Inpainting"""
        width, height = init_image.size
        if height % 8 != 0 or width % 8 != 0:
            height = (height // 8) * 8
            width = (width // 8) * 8
            init_image = init_image.crop((0, 0, width, height))
            mask_image = mask_image.crop((0, 0, width, height))

        pipe = self.get_pipe("inpaint")
        images = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
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
                prompt=cfg.get_prompt(0),
                width=cfg.width,
                height=cfg.height,
                inpaint_strength=cfg.inpaint_strength,
                num_images_per_prompt=cfg.num_images_per_prompt,
                num_inference_steps=cfg.num_inference_steps,
                seed=cfg.seed,
                max_display_image_width=rc.batch.max_display_image_width,
            )

        # add borders
        w, h = output.size
        panorama = Image.new("RGB", (w, 3 * h))
        panorama.paste(output, (0, h))
        img_path = rc.save(panorama, "panorama")
        if cfg.display_image:
            if cfg.clear_output:
                eKonf.clear_output(wait=True)
            if rc.batch.max_display_image_width is not None:
                panorama = eKonf.scale_image(
                    panorama, max_width=rc.batch.max_display_image_width
                )
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
        max_display_image_width=None,
        seed=None,
        **kwargs,
    ):
        """Stitching images"""

        output = images[0]
        pipe = self.get_pipe("inpaint")

        for img in tqdm(images[1:] + [images[0]]):
            w, h = output.size
            new = Image.new("RGB", (width, height))
            new.paste(output, (-w + width // 2, 0))
            new.paste(img, (width // 2, 0))

            msk = Image.new("L", (width, height))
            drw = ImageDraw.Draw(msk)
            drw.rectangle((width // 4, 0, 3 * width // 4, height), fill=255)
            _width, _height = new.size
            merged = pipe(
                prompt=prompt,
                image=new,
                mask_image=msk,
                width=_width,
                height=_height,
                guidance_scale=inpaint_strength,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                generator=self.get_generator(seed),
                **kwargs,
            ).images[0]
            eKonf.clear_output(wait=True)
            _img = eKonf.scale_image(merged, max_width=max_display_image_width)
            eKonf.display(_img)

            new = Image.new("RGB", (w + width, height))
            new.paste(output, (0, 0))
            new.paste(merged, (w - width // 2, 0))
            img = img.crop((width // 2, 0, width, height))
            new.paste(img, (w + width // 2, 0))
            output = new

        w, h = output.size
        output = output.crop((width // 2, 0, w - width // 2, h))
        return output

    def get_pipe(self, name: str):
        pipe = self.__pipes__.get(name)
        if pipe is None:
            self.load_diffusers(name)
            pipe = self.__pipes__.get(name)
        if pipe is None:
            raise ValueError(f"pipe {name} not found")
        return pipe

    def load_diffusers(self, models=None):
        torch.set_grad_enabled(False)

        if models is None:
            models = self.model.keys()
        if isinstance(models, str):
            models = [models]
        for model in models:
            cfg = self.model[model]
            if cfg.pipeline == "StableDiffusionPipeline":
                DiffusionPipeline = StableDiffusionPipeline
            elif cfg.pipeline == "StableDiffusionInpaintPipeline":
                DiffusionPipeline = StableDiffusionInpaintPipeline
            else:
                raise ValueError(f"pipeline {cfg.pipeline} not found")

            self.__pipes__[model] = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=cfg.name,
                use_auth_token=self.secrets.HUGGING_FACE_HUB_TOKEN.get_secret_value(),
                revision=cfg.revision,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
            ).to(cfg.device)

    def reset(self):
        self.__pipes__ = {}
        return super().reset()

    def get_generator(self, seed=None):
        if seed is None:
            seed = self.seed
        return torch.Generator(device=self.device).manual_seed(seed)


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
