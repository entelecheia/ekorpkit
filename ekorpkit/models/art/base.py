import os
import logging
from pathlib import Path
from ekorpkit import eKonf
from ekorpkit.batch import BaseConfig


log = logging.getLogger(__name__)


class BaseModel(BaseConfig):
    def __init__(self, root_dir=None, **args):
        super().__init__(root_dir=root_dir, **args)
        super().__init__(**args)

        self.sample_imagepaths = []

    @property
    def autoload(self):
        return self.config.get("autoload", False)

    @property
    def device(self):
        return self.config.get("device", "cpu")

    @property
    def num_devices(self):
        return self.config.get("num_devices")

    @num_devices.setter
    def num_devices(self, num_devices):
        self.config.num_devices = num_devices

    @property
    def version(self):
        return self.config.get("version", "0.0.0")

    @property
    def model_config(self):
        return self.config.model

    @property
    def module_config(self):
        return self.config.module

    def load(self):
        log.info("> downloading models...")
        self.download_models()
        log.info("> loading modules...")
        self.load_modules()
        log.info("> loading models...")
        self.load_models()

    def imagine(self, text_prompts=None, **args):
        """Imagine the text prompts"""
        raise NotImplementedError

    def load_models(self):
        raise NotImplementedError

    def load_modules(self):
        """Load the modules"""
        if self.module_config.get("modules") is None:
            log.info("No modules to load")
            return
        library_dir = self.path.library_dir
        for module in self.module_config.modules:
            name = module.name
            libname = module.libname
            liburi = module.liburi
            specname = module.specname
            libpath = os.path.join(library_dir, libname)
            syspath = module.get("syspath")
            if syspath is not None:
                syspath = os.path.join(library_dir, syspath)
            eKonf.ensure_import_module(name, libpath, liburi, specname, syspath)

    def download_models(self):
        """Download the models"""
        pass
        # download = self.args.download
        # for name, model in download.models.items():
        #     if not isinstance(model, str):
        #         log.info(f"Downloading model {name} from {model}")

    def get_text_prompt(self, prompts):
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
            return self.get_text_prompt(prompts)

    def get_image_path(
        self,
        sample_suffix,
        batch_name: str = None,
        batch_num: int = None,
        batch_dir: str = None,
        image_ext: str = "png",
    ) -> str:
        batch_dir = batch_dir or self.batch_dir
        batch_name = batch_name or self.batch_name
        batch_num = batch_num or self.batch_num
        if isinstance(sample_suffix, int):
            sample_suffix = f"{sample_suffix:04d}"
        else:
            sample_suffix = str(sample_suffix)
        filename = f"{batch_name}({batch_num})_{sample_suffix}.{image_ext}"
        return str(Path(batch_dir) / filename)

    def collage(
        self,
        image_filepaths=None,
        batch_name=None,
        batch_num=None,
        ncols=2,
        num_images=None,
        filename_patterns=None,
        show_prompt=True,
        prompt_fontsize=18,
        show_filename=False,
        filename_offset=(5, 5),
        fontname=None,
        fontsize=12,
        fontcolor=None,
        resize_ratio=1.0,
        **kwargs,
    ):
        self.load_config(batch_name, batch_num, **kwargs)
        batch_name = self.batch_name
        batch_num = self.batch_num
        cfg = self.config.imagine

        filename_patterns = filename_patterns or f"{batch_name}({batch_num})_*.png"
        # num_images = num_images or cfg.get("num_samples") or cfg.get("n_samples")
        prompt = None
        if show_prompt:
            prompt = self.get_text_prompt(cfg.text_prompts)
            log.info(f"Prompt: {prompt}")

        eKonf.collage(
            image_filepaths=image_filepaths,
            filename_patterns=filename_patterns,
            base_dir=self.batch_dir,
            num_images=num_images,
            ncols=ncols,
            title=prompt,
            title_fontsize=prompt_fontsize,
            show_filename=show_filename,
            filename_offset=filename_offset,
            fontname=fontname,
            fontsize=fontsize,
            fontcolor=fontcolor,
            resize_ratio=resize_ratio,
            **kwargs,
        )
