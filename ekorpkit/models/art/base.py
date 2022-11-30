import logging
from ekorpkit import eKonf
from ekorpkit.config import BaseBatchModel
from ekorpkit.visualize.collage import collage, label_collage
from .config import (
    BatchImagineConfig,
    BatchConfig,
    ImagineConfig,
    RunConfig,
    CollageConfig,
    BatchRunConfig,
    BatchImagineResult,
)


log = logging.getLogger(__name__)


class BaseModel(BaseBatchModel):
    batch: BatchConfig = None
    imagine: ImagineConfig = None
    collage: CollageConfig = None
    config_to_save = ["batch", "imagine"]
    sample_imagepaths = []

    def __init__(self, **args):
        super().__init__(**args)

        self.load_modules()

    def load(self):
        log.info("> downloading models...")
        self.download_models()
        log.info("> loading models...")
        self.load_models()

    def generate(self, **args):
        """Imagine the text prompts"""
        raise NotImplementedError

    def load_models(self):
        raise NotImplementedError

    def download_models(self):
        """Download the models"""
        pass
        # download = self.args.download
        # for name, model in download.models.items():
        #     if not isinstance(model, str):
        #         log.info(f"Downloading model {name} from {model}")

    def get_run_config(self, config):
        batch = BatchConfig(output_dir=config.path.output_dir, **config.batch)
        imagine = ImagineConfig(**config.imagine)
        collage = CollageConfig(**config.collage)
        rc = RunConfig(batch=batch, imagine=imagine, collage=collage)
        return rc

    def collage_images(
        self,
        images_or_uris=None,
        batch_name=None,
        batch_num=None,
        display_collage=True,
        save_collage=True,
        clear_output=True,
        show_prompt=True,
        ncols=3,
        max_images=12,
        collage_width=1200,
        padding: int = 10,
        bg_color: str = "black",
        crop_to_min_size=False,
        show_filename=False,
        filename_offset=(5, 5),
        fontname=None,
        fontsize=12,
        fontcolor="#000",
        **kwargs,
    ):
        config = self.load_config(batch_name, batch_num, **kwargs)
        rc = self.get_run_config(config)

        if images_or_uris is None:
            images_or_uris = rc.image_filepaths
        collage_filepath = rc.batch.collage_filepath if save_collage else None

        collage_result = collage(
            images_or_uris=images_or_uris,
            collage_filepath=collage_filepath,
            ncols=ncols,
            max_images=max_images,
            collage_width=collage_width,
            padding=padding,
            bg_color=bg_color,
            crop_to_min_size=crop_to_min_size,
            show_filename=show_filename,
            filename_offset=filename_offset,
            fontname=fontname,
            fontsize=fontsize,
            fontcolor=fontcolor,
            **kwargs,
        )
        if display_collage and collage_result is not None:
            if clear_output:
                eKonf.clear_output(wait=True)
            if show_prompt:
                prompt = rc.imagine.get_str_prompt()
                if prompt:
                    print(f"Prompt: {prompt}")
            img = collage_result.image
            if rc.batch.max_display_image_width is not None:
                img = eKonf.scale_image(img, max_width=rc.batch.max_display_image_width)
            eKonf.display(img)

    def batch_imagine(
        self,
        batch_name,
        batch_run_params,
        batch_run_pairs=None,
        num_samples=1,
        max_display_image_width=800,
        **imagine_args,
    ):
        """Run a batch"""
        if num_samples is not None:
            imagine_args.update(dict(num_samples=num_samples))

        batch = BatchImagineConfig(
            output_dir=self.output_dir,
            batch_name=batch_name,
            batch_run_params=batch_run_params,
            batch_run_pairs=batch_run_pairs,
        )

        run_configs = {}
        collage_filepaths = {}
        batch_prompts = {}
        for batch_run_cfg in batch.batch_run_configs:
            batch_run_pair = batch_run_cfg.batch_run_pair
            batch_run_name = batch_run_cfg.batch_name
            for batch_args in batch.get_run_args(batch_run_pair):
                imgn_args = batch.get_imagine_config(batch_args, **imagine_args)
                batch_prompts[batch_run_name] = None
                if "text_prompts" in imgn_args:
                    batch_args["text_prompts"] = imgn_args["text_prompts"]
                    batch_prompts[batch_run_name] = imgn_args["text_prompts"]
                log.info(f"batch: {batch_run_name} with {batch_args}")
                imagine_rst = self.generate(
                    batch_name=batch_run_name,
                    **imgn_args,
                )
                batch_run_cfg.append(batch_args, imagine_rst)

            run_config_path = batch.save_run_config(batch_run_cfg)
            run_configs[batch_run_name] = run_config_path

        if batch.display_collage or batch.save_collage:
            eKonf.clear_output(wait=True)
            for batch_run_name, run_config_path in run_configs.items():
                collage_filepaths[batch_run_name] = self.batch_collage(
                    run_config_path, max_display_image_width=max_display_image_width
                )
        return BatchImagineResult(
            run_configs=run_configs,
            collage_filepaths=collage_filepaths,
            batch_prompts=batch_prompts,
        )

    def batch_collage(
        self,
        run_config_path,
        max_collages=10,
        display_collage=True,
        clear_output=False,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        ncols=3,
        max_images_per_collage=20,
        prompt_fontsize=10,
        show_filename=False,
        filename_offset=(5, 5),
        fontname=None,
        fontsize=18,
        fontcolor=None,
        max_display_image_width=800,
        dpi=100,
        **kwargs,
    ):
        cfg = BatchRunConfig(run_config_path=run_config_path)
        batch_name = cfg.batch_name
        batch_run_pair = cfg.batch_run_pair
        arg_names = list(batch_run_pair.keys())

        if "text_prompts" in arg_names:
            zlabel = "text_prompts"
            zvalues = batch_run_pair[zlabel]
            arg_names.remove(zlabel)
        else:
            zvalues = [None]

        if ylabel is None and len(arg_names) > 0:
            ylabel = arg_names[0]
        if ylabel in arg_names:
            yticklabels = batch_run_pair[ylabel]
            # reverse yticklabels so that the first image is the top left
            yticklabels = yticklabels[::-1]
            arg_names.remove(ylabel)
        else:
            yticklabels = None
            # raise ValueError(f"{ylabel} not in {arg_names}")

        if xlabel is None and len(arg_names) > 0:
            xlabel = arg_names[0]
        if xlabel in arg_names:
            xticklabels = batch_run_pair[xlabel]
            arg_names.remove(xlabel)
        else:
            xticklabels = None

        if zlabel is None and len(arg_names) > 0:
            zlabel = arg_names[0]
        if zlabel in arg_names:
            zvalues = batch_run_pair[zlabel]
            arg_names.remove(zlabel)

        if xticklabels is None and zlabel is not None and zlabel != "text_prompts":
            ncols = 1
        elif xticklabels:
            ncols = len(xticklabels)

        collage_filepaths = []
        prompt = ""
        if len(zvalues) > max_collages:
            zvalues = zvalues[:max_collages]
        for z, zvalue in enumerate(zvalues):

            title = f"Batch name: {batch_name}"
            if zvalue is not None:
                if zlabel == "text_prompts":
                    sfx = f"{zlabel}({z})"
                else:
                    sfx = f"{zlabel}({zvalue})"
                    title += f"\n{zlabel}: {zvalue}"
                output_filepath = cfg.collage_filepath(sfx)
            else:
                output_filepath = cfg.collage_filepath()
            collage_filepaths.append(output_filepath)
            image_filepaths = []
            for result in cfg.imagine_results:
                if zvalue is not None and zlabel == "text_prompts" and xlabel is None:
                    if result.batch_args[zlabel] == zvalue:
                        image_filepaths = result.image_filepaths
                        prompt = zvalue
                        break
                if zlabel is None or result.batch_args[zlabel] == zvalue:
                    image_filepaths.append(result.image_filepaths[0])
                    prompt = result.batch_args.get("text_prompts", None)

            if len(image_filepaths) == 0:
                continue
            if prompt:
                title += f"\nPrompt: {prompt}"

            collage_result = collage(
                images_or_uris=image_filepaths,
                collage_filepath=output_filepath,
                ncols=ncols,
                max_images=max_images_per_collage,
                show_filename=show_filename,
                filename_offset=filename_offset,
                fontname=fontname,
                fontsize=fontsize,
                fontcolor=fontcolor,
                **kwargs,
            )

            collage_result = label_collage(
                collage_result,
                collage_filepath=output_filepath,
                title=title,
                title_fontsize=prompt_fontsize,
                xlabel=xlabel,
                ylabel=ylabel,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                xlabel_fontsize=fontsize,
                ylabel_fontsize=fontsize,
                dpi=dpi,
                **kwargs,
            )
            if display_collage and collage_result is not None:
                if clear_output:
                    eKonf.clear_output(wait=True)
                if prompt:
                    print(f"Prompt[{z}]: {prompt}")
                img = collage_result.image
                if max_display_image_width is not None:
                    img = eKonf.scale_image(img, max_width=max_display_image_width)
                eKonf.display(img)

        return collage_filepaths

    def save_config(self, config=None):
        super().save_config(config=config, include=self.config_to_save)
