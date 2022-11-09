import os
import logging
from ekorpkit import eKonf
from .base import BaseModel
from .config import AnimMode


log = logging.getLogger(__name__)


class BatchDiffusion(BaseModel):
    def __init__(self, root_dir=None, config_name="default", **args):
        cfg = eKonf.compose(f"model/stable_diffusion={config_name}")
        cfg = eKonf.merge(cfg, args)
        super().__init__(root_dir=root_dir, **cfg)

    def batch_imagine(
        self,
        text_prompts=None,
        image_prompts=None,
        batch_name=None,
        batch_args=None,
        batch_pairs=None,
        show_collage=True,
        **args,
    ):
        """Run a batch"""
        animation_mode = AnimMode.NONE
        num_samples = 1
        args.update(dict(show_collage=False))

        if batch_args is None:
            batch_args = {}
        if not isinstance(batch_args, dict):
            raise ValueError("batch_args must be a dictionary")
        if len(batch_args) < 1:
            raise ValueError("batch_args must have at least 1 element")
        for k, v in batch_args.items():
            if not isinstance(v, (list, tuple)):
                v = [v]
        if batch_pairs is None:
            batch_pairs = [[arg_name] for arg_name in batch_args.keys()]

        _batch_results = []
        for batch_pair in batch_pairs:
            batch_pair_args = {}
            for arg_name in batch_pair:
                batch_pair_args[arg_name] = batch_args[arg_name]
            _batch_name = batch_name + "_" + "_".join(batch_pair)
            batch_config = dict(
                batch_name=_batch_name,
                batch_pair_args=batch_pair_args,
                text_prompts=text_prompts,
                image_prompts=image_prompts,
                results=[],
            )
            for pair_args in eKonf.dict_product(batch_pair_args):
                _args = args.copy()
                _args.update(pair_args)
                print(f"batch: {_batch_name} with {pair_args}")
                results = self.imagine(
                    text_prompts=text_prompts,
                    image_prompts=image_prompts,
                    batch_name=_batch_name,
                    animation_mode=animation_mode,
                    num_samples=num_samples,
                    **_args,
                )
                batch_config["results"].append(
                    dict(
                        args=pair_args,
                        config_file=results["config_file"],
                        image_filepaths=results["image_filepaths"],
                    )
                )

            _batch_config_path = self.save_batch_configs(batch_config)
            _batch_results.append(_batch_config_path)
            if show_collage:
                self.batch_collage(_batch_config_path)
        return _batch_results

    def save_batch_configs(self, args):
        """Save the settings"""
        _batch_name = args["batch_name"]
        _filename = f"{_batch_name}_batch_configs.yaml"
        _path = os.path.join(self._output.batch_configs_dir, _filename)
        log.info(f"Saving batch configs to {_path}")
        eKonf.save(args, _path)
        return _path

    def batch_collage(
        self,
        batch_config_path,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        prompt_fontsize=18,
        show_filename=False,
        filename_offset=(5, 5),
        fontname=None,
        fontsize=18,
        fontcolor=None,
        **kwargs,
    ):
        args = eKonf.load(batch_config_path)
        args = eKonf.to_dict(args)

        batch_name = args["batch_name"]
        batch_pair_args = args["batch_pair_args"]
        arg_names = list(batch_pair_args.keys())

        if ylabel is None:
            ylabel = arg_names[0]
        if ylabel in arg_names:
            yticklabels = batch_pair_args[ylabel]
            arg_names.remove(ylabel)
        else:
            raise ValueError(f"{ylabel} not in {arg_names}")
        # reverse yticklabels so that the first image is the top left
        yticklabels = yticklabels[::-1]

        if xlabel is None and len(arg_names) > 0:
            xlabel = arg_names[0]
        if xlabel in arg_names:
            xticklabels = batch_pair_args[xlabel]
            arg_names.remove(xlabel)
        else:
            xticklabels = None

        if zlabel is None and len(arg_names) > 0:
            zlabel = arg_names[0]
        if zlabel in arg_names:
            ztitles = batch_pair_args[zlabel]
            arg_names.remove(zlabel)
        else:
            ztitles = [None]

        results = args["results"]
        prompt = args["text_prompts"]
        ncols = 1 if xticklabels is None else len(xticklabels)

        log.info(f"Prompt: {prompt}")
        for ztitle in ztitles:
            title = f"batch name: {batch_name}\nprompt: {prompt}\n"
            if ztitle is not None:
                output_filepath = batch_config_path.replace(
                    ".yaml", f"_{zlabel}({ztitle}).png"
                )
                title += f"{zlabel}: {ztitle}\n"
            else:
                output_filepath = batch_config_path.replace(".yaml", ".png")

            image_filepaths = []
            for result in results:
                if zlabel is None or result["args"][zlabel] == ztitle:
                    image_filepaths.append(result["image_filepaths"][0])

            eKonf.collage(
                image_filepaths=image_filepaths,
                output_filepath=output_filepath,
                ncols=ncols,
                title=title,
                title_fontsize=prompt_fontsize,
                show_filename=show_filename,
                filename_offset=filename_offset,
                fontname=fontname,
                fontsize=fontsize,
                fontcolor=fontcolor,
                xlabel=xlabel,
                ylabel=ylabel,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                xlabel_fontsize=fontsize,
                ylabel_fontsize=fontsize,
                **kwargs,
            )
            print(f"Saved collage to {output_filepath}")
