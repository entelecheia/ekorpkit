import os
import logging
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class BaseTTIModel:
    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.verbose = args.get("verbose", True)
        self.auto = args.auto
        self._path = self.args.path
        self._output = self.args.output
        self._module = self.args.module
        self._model = self.args.model
        self._config = self.args.config

        self.sample_imagepaths = []

    @property
    def path(self):
        return self._path

    @property
    def config(self):
        return self._config

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
        library_dir = self._module.library_dir
        for module in self._module.modules:
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

    def save_settings(self, args):
        """Save the settings"""
        _path = os.path.join(
            self._output.batch_dir, f"{args.batch_name}({args.batch_num})_settings.yaml"
        )
        log.info(f"Saving config to {_path}")
        eKonf.save(args, _path)

    def load_config(self, batch_name=None, batch_num=None, **args):
        """Load the settings"""
        _config = self._config
        if batch_name is None:
            batch_name = _config.batch_name
        else:
            _config.batch_name = batch_name
        self._prepare_folders(batch_name)
        if batch_num is not None:
            _path = os.path.join(
                self._output.batch_dir, f"{batch_name}({batch_num})_settings.yaml"
            )
            if os.path.exists(_path):
                log.info(f"Loading config from {_path}")
                batch_args = eKonf.load(_path)
                log.info(f"Merging config with diffuse defaults")
                _config = eKonf.merge(_config, batch_args)
                # return _config

        log.info(f"Merging config with args: {args}")
        args = eKonf.merge(_config, args)

        return args

    def show_config(self, batch_name=None, batch_num=None):
        args = self.load_config(batch_name, batch_num)
        eKonf.print(args)

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
        **kwargs,
    ):
        args = self.load_config(batch_name, batch_num, **kwargs)
        batch_name = batch_name or args.batch_name
        if batch_num is None:
            batch_num = args.batch_num

        filename_patterns = filename_patterns or f"{batch_name}({batch_num})_*.png"
        num_images = num_images or args.n_samples
        prompt = None
        if show_prompt:
            prompt = self.get_text_prompt(args.text_prompts)
            log.info(f"Prompt: {prompt}")

        eKonf.collage(
            image_filepaths=image_filepaths,
            filename_patterns=filename_patterns,
            base_dir=self._output.batch_dir,
            num_images=num_images,
            ncols=ncols,
            title=prompt,
            title_fontsize=prompt_fontsize,
            show_filename=show_filename,
            filename_offset=filename_offset,
            fontname=fontname,
            fontsize=fontsize,
            fontcolor=fontcolor,
        )

    def _prepare_folders(self, batch_name):
        self._output.batch_dir = os.path.join(self._output.root, batch_name)
        for _name, _path in self._output.items():
            if _name.endswith("_dir") and not os.path.exists(_path):
                os.makedirs(_path)
