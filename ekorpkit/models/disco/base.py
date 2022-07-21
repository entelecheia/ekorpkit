import os
import logging
import subprocess
import math
import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import shutil
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import PIL
from PIL import Image
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path
from ekorpkit import eKonf
from .utils import (
    _download_models,
    move_files,
    split_prompts,
    parse_key_frames,
    get_inbetweens,
)
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.visualize.motion import create_video as _create_video
from ekorpkit.visualize.motion import extract_frames
from ..art.base import BaseTTIModel
from enum import Enum

log = logging.getLogger(__name__)


class AnimMode(str, Enum):
    """Animation mode"""

    NONE = "None"
    ANIM_2D = "2D"
    ANIM_3D = "3D"
    VIDEO_INPUT = "Video Input"


class DiscoDiffusion(BaseTTIModel):

    TRANSLATION_SCALE = 1.0 / 200.0

    def __init__(self, **args):
        super().__init__(**args)

        self._midas = self.args.midas
        self.clip_models = []
        self.model = None
        self.diffusion = None
        self.angle_series = None
        self.zoom_series = None
        self.translation_x_series = None
        self.translation_y_series = None
        self.translation_z_series = None
        self.rotation_3d_x_series = None
        self.rotation_3d_y_series = None
        self.rotation_3d_z_series = None
        self.text_prompts_series = None
        self.image_prompts_series = None

        if self.auto.load:
            self.load()

    def imagine(
        self,
        text_prompts=None,
        image_prompts=None,
        batch_name=None,
        batch_num=None,
        animation_mode: AnimMode = None,
        run_to_resume=None,
        **args,
    ):

        if text_prompts is not None:
            args.update(dict(text_prompts=text_prompts))
        if image_prompts is not None:
            args.update(dict(image_prompts=image_prompts))
        if animation_mode is not None:
            args.update(dict(animation_mode=animation_mode))
        if run_to_resume is not None:
            args.update(dict(run_to_resume=run_to_resume))

        log.info("> loading settings...")
        args = self.load_config(batch_name=batch_name, batch_num=batch_num, **args)
        args = self._prepare_config(args)
        self._config = args

        if args.animation_mode == AnimMode.NONE:
            if args.start_sample >= args.n_samples:
                log.warning(
                    f"start_sample ({args.start_sample}) must be less than n_samples ({args.n_samples})"
                )
                return
        else:
            if args.start_frame >= args.max_frames:
                log.warning(
                    f"start frame {args.start_frame} is greater than max frames {args.max_frames}"
                )
                return

        log.info(
            f"Starting Run: {args.batch_name}({args.batch_num}) at frame {args.start_frame}"
        )
        self._prepare_models()
        self.sample_imagepaths = []
        with elapsed_timer(format_time=True) as elapsed:

            gc.collect()
            torch.cuda.empty_cache()
            try:
                if args.animation_mode == AnimMode.NONE:
                    self._run(args)
                else:
                    self._run_anim(args)
            except KeyboardInterrupt:
                pass
            finally:
                log.info(f"Seed used: {args.seed}")
                gc.collect()
                torch.cuda.empty_cache()

            log.info(" >> elapsed time to diffuse: {}".format(elapsed()))
            if args.animation_mode == AnimMode.NONE:
                print(f"{args.n_samples} samples generated to {self._output.batch_dir}")
                print(f"text prompts: {text_prompts}")
                print("sample image paths:")
                for p in self.sample_imagepaths:
                    print(p)

                if args.show_collage:
                    self.collage(image_filepaths=self.sample_imagepaths)

        self.save_settings(args)

    def _prepare_models(self):
        from guided_diffusion.script_util import create_model_and_diffusion

        log.info("Prepping model...")
        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        if self._model.diffusion_model == "custom":
            custom_path = self._path.model_dir + "/" + self._model.custom_model
            self.model.load_state_dict(torch.load(custom_path, map_location="cpu"))
        else:
            model_path = (
                self._path.model_dir + "/" + self._model.diffusion_model + ".pt"
            )
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.requires_grad_(False).eval().to(self.cuda_device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

    def init_flow_warp(self, args):
        """Init the optical flow"""
        import argparse
        from raft import RAFT
        from .flow import load_img, get_flow

        video_frames_dir = self._output.video_frames_dir
        flo_fwd_dir = self._output.flo_fwd_dir
        flo_dir = self._output.flo_dir
        raft_model_path = self.args.download.models.RAFT.path

        rsft_args = argparse.Namespace()
        rsft_args.small = False
        rsft_args.mixed_precision = True

        if not args.video_init_flow_warp:
            log.info("video_init_flow_warp not set, skipping")

        else:
            flows = glob(flo_dir + "/*.*")
            if (len(flows) > 0) and not args.force_flow_generation:
                log.info(
                    f"Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {self._output.flo_dir}.\nIf you wish to generate new flow files, set force_flow_generation=True and run again."
                )

            if (len(flows) == 0) or args.force_flow_generation:
                frames = sorted(glob(video_frames_dir + "/*.*"))
                if len(frames) < 2:
                    log.warning(
                        f"WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path."
                    )
                if len(frames) >= 2:

                    raft_model = torch.nn.DataParallel(RAFT(rsft_args))
                    raft_model.load_state_dict(torch.load(raft_model_path))
                    raft_model = raft_model.module.cuda(device=self.cuda_device).eval()

                    for f in Path(f"{flo_fwd_dir}").glob("*.*"):
                        f.unlink()

                    # TBD Call out to a consistency checker?
                    self.framecount = 0
                    for frame1, frame2 in tqdm(
                        zip(frames[:-1], frames[1:]), total=len(frames) - 1
                    ):

                        out_flow21_fn = f"{flo_fwd_dir}/{frame1.split('/')[-1]}"

                        frame1 = load_img(frame1, args.width_height)
                        frame2 = load_img(frame2, args.width_height)

                        flow21 = get_flow(frame2, frame1, raft_model)
                        np.save(out_flow21_fn, flow21)

                        if args.video_init_check_consistency:
                            # TBD
                            pass

                    del raft_model
                    gc.collect()

    def load_models(self):
        log.info("> loading diffusion models...")
        self.load_diffusion_models()
        log.info("> loading clip models...")
        self.load_clip_models()

    def load_diffusion_models(self):
        from guided_diffusion.script_util import model_and_diffusion_defaults
        from .secondary import SecondaryDiffusionImageNet2

        DEVICE = torch.device(
            f"cuda:{self.args.cuda_device}"
            if (torch.cuda.is_available() and not self.args.use_cpu)
            else "cpu"
        )
        log.info(f"Using device:{DEVICE}")
        self.cuda_device = DEVICE

        if not self.args.use_cpu:
            ## A100 fix thanks to Emad
            if torch.cuda.get_device_capability(self.cuda_device) == (8, 0):
                log.info("Disabling CUDNN for A100 gpu")
                torch.backends.cudnn.enabled = False

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(eKonf.to_dict(self.args.diffusion_model))
        self.model_default = self.model_config["image_size"]

        if self.args.model.use_secondary_model:
            secondary_model_path = self.args.download.models.model_secondary.path
            self.secondary_model = SecondaryDiffusionImageNet2()
            self.secondary_model.load_state_dict(
                torch.load(secondary_model_path, map_location="cpu")
            )
            self.secondary_model.eval().requires_grad_(False).to(self.cuda_device)

    def load_clip_models(self):
        """Load the clips"""
        import torchvision.transforms as T
        import lpips
        import clip

        _clips = self._model.clip_models
        clip_models = []
        for name, _use in _clips.items():
            if _use:
                clip_models.append(
                    clip.load(name, jit=False)[0]
                    .eval()
                    .requires_grad_(False)
                    .to(self.cuda_device)
                )
        self.clip_models = clip_models
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.cuda_device)

    def download_models(self):
        """Download the models"""
        download = self.args.download
        check_model_SHA = download.check_model_SHA
        for name, model in download.models.items():
            if not isinstance(model, str):
                log.info(f"Downloading model {name} from {model}")
                _download_models(name, **model, check_model_SHA=check_model_SHA)

        if not os.path.exists(download.pretrained_symlink):
            os.symlink(download.pretrained_dir, download.pretrained_symlink)
            log.info(
                f"Created symlink of {download.pretrained_dir} to {download.pretrained_symlink}"
            )

    def _3d_step(self, img_filepath, frame_num, midas_model, midas_transform, args):
        import py3d_tools as p3dT
        from . import disco_xform_utils as dxf

        if args.key_frames:
            translation_x = self.translation_x_series[frame_num]
            translation_y = self.translation_y_series[frame_num]
            translation_z = self.translation_z_series[frame_num]
            rotation_3d_x = self.rotation_3d_x_series[frame_num]
            rotation_3d_y = self.rotation_3d_y_series[frame_num]
            rotation_3d_z = self.rotation_3d_z_series[frame_num]
            log.info(f"translation_x: {translation_x}")
            log.info(f"translation_y: {translation_y}")
            log.info(f"translation_z: {translation_z}")
            log.info(f"rotation_3d_x: {rotation_3d_x}")
            log.info(f"rotation_3d_y: {rotation_3d_y}")
            log.info(f"rotation_3d_z: {rotation_3d_z}")

        translate_xyz = [
            -translation_x * self.TRANSLATION_SCALE,
            translation_y * self.TRANSLATION_SCALE,
            -translation_z * self.TRANSLATION_SCALE,
        ]
        rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
        log.info(f"translation: {translate_xyz}")
        log.info(f"rotation: {rotate_xyz_degrees}")
        rotate_xyz = [
            math.radians(rotate_xyz_degrees[0]),
            math.radians(rotate_xyz_degrees[1]),
            math.radians(rotate_xyz_degrees[2]),
        ]
        rot_mat = p3dT.euler_angles_to_matrix(
            torch.tensor(rotate_xyz, device=self.cuda_device), "XYZ"
        ).unsqueeze(0)
        log.info(f"rot_mat: {str(rot_mat)}")
        next_step_pil = dxf.transform_image_3d(
            img_filepath,
            midas_model,
            midas_transform,
            self.cuda_device,
            rot_mat,
            translate_xyz,
            args.near_plane,
            args.far_plane,
            args.fov,
            padding_mode=args.padding_mode,
            sampling_mode=args.sampling_mode,
            midas_weight=args.midas_weight,
        )
        return next_step_pil

    def _get_model_stats(self, text_prompt, image_prompt, args):
        """Update the model stats"""
        import clip
        from .diffuse import parse_prompt, MakeCutouts, fetch

        model_stats = []
        weights = []

        for clip_model in self.clip_models:
            cutn = 16
            model_stat = {
                "clip_model": None,
                "target_embeds": [],
                "make_cutouts": None,
                "weights": [],
            }
            model_stat["clip_model"] = clip_model

            for prompt in text_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(
                    clip.tokenize(prompt).to(self.cuda_device)
                ).float()

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append(
                            (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(
                                0, 1
                            )
                        )
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(
                    clip_model.visual.input_resolution,
                    cutn,
                    skip_augs=args.skip_augs,
                )
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert("RGB")
                    img = TF.resize(
                        img,
                        min(args.side_x, args.side_y, *img.size),
                        T.InterpolationMode.LANCZOS,
                    )
                    batch = model_stat["make_cutouts"](
                        TF.to_tensor(img)
                        .to(self.cuda_device)
                        .unsqueeze(0)
                        .mul(2)
                        .sub(1)
                    )
                    embed = clip_model.encode_image(self.normalize(batch)).float()
                    if args.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (
                                    embed
                                    + torch.randn(embed.shape).cuda() * args.rand_mag
                                ).clamp(0, 1)
                            )
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(
                model_stat["weights"], device=self.cuda_device
            )
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError("The weights must not sum to 0.")
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        return model_stats

    def _init_image(self, init_image, args):
        from .diffuse import create_perlin_noise, fetch

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert("RGB")
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(self.cuda_device).unsqueeze(0).mul(2).sub(1)

        if args.perlin_init:
            log.info(f"Perlin init with {args.perlin_mode} mode")
            if args.perlin_mode == "color":
                grayscale = False
                grayscale2 = False
            elif args.perlin_mode == "gray":
                grayscale = True
                grayscale2 = True
            else:
                grayscale = False
                grayscale2 = True
            init = create_perlin_noise(
                args.side_x,
                args.side_y,
                octaves=[1.5 ** -i * 0.5 for i in range(12)],
                width=1,
                height=1,
                grayscale=grayscale,
                device=self.cuda_device,
            )
            init2 = create_perlin_noise(
                args.side_x,
                args.side_y,
                octaves=[1.5 ** -i * 0.5 for i in range(8)],
                width=4,
                height=4,
                grayscale=grayscale2,
                device=self.cuda_device,
            )
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = (
                TF.to_tensor(init)
                .add(TF.to_tensor(init2))
                .div(2)
                .to(self.cuda_device)
                .unsqueeze(0)
                .mul(2)
                .sub(1)
            )
            del init2
        return init

    def _generate_progress_samples(
        self,
        cur_t,
        init,
        init_scale,
        skip_steps,
        loss_values,
        model_stats,
        frame_num,
        diffusion,
        midas_model,
        midas_transform,
        image_display,
        args,
    ):
        import cv2
        from .secondary import alpha_sigma_to_t
        from .diffuse import (
            MakeCutoutsDango,
            spherical_dist_loss,
            tv_loss,
            range_loss,
            regen_perlin,
            generate_eye_views,
        )

        batch_dir = self._output.batch_dir
        partial_dir = self._output.partial_dir
        prev_frame_path = self._output.prev_frame_path
        prev_frame_scaled_path = self._output.prev_frame_scaled_path
        progress_path = self._output.progress_path

        secondary_model = self.secondary_model
        model = self.model
        if isinstance(args.cut_overview, str):
            cut_overview = eval(args.cut_overview)
            cut_innercut = eval(args.cut_innercut)
            cut_icgray_p = eval(args.cut_icgray_p)
        else:
            cut_overview = args.cut_overview
            cut_innercut = args.cut_innercut
            cut_icgray_p = args.cut_icgray_p

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if args.use_secondary_model:
                    alpha = torch.tensor(
                        diffusion.sqrt_alphas_cumprod[cur_t],
                        device=self.cuda_device,
                        dtype=torch.float32,
                    )
                    sigma = torch.tensor(
                        diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                        device=self.cuda_device,
                        dtype=torch.float32,
                    )
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = (
                        torch.ones([n], device=self.cuda_device, dtype=torch.long)
                        * cur_t
                    )
                    out = diffusion.p_mean_variance(
                        model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                    )
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)

                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = (
                            int(t.item()) + 1
                        )  # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution = model_stat[
                                "clip_model"
                            ].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(
                            args.animation_mode,
                            input_resolution,
                            args.skip_augs,
                            Overview=cut_overview[1000 - t_int],
                            InnerCrop=cut_innercut[1000 - t_int],
                            IC_Size_Pow=args.cut_ic_pow,
                            IC_Grey_P=cut_icgray_p[1000 - t_int],
                        )
                        clip_in = self.normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = (
                            model_stat["clip_model"].encode_image(clip_in).float()
                        )
                        dists = spherical_dist_loss(
                            image_embeds.unsqueeze(1),
                            model_stat["target_embeds"].unsqueeze(0),
                        )
                        dists = dists.view(
                            [
                                cut_overview[1000 - t_int] + cut_innercut[1000 - t_int],
                                n,
                                -1,
                            ]
                        )
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(
                            losses.sum().item()
                        )  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += (
                            torch.autograd.grad(
                                losses.sum() * args.clip_guidance_scale, x_in
                            )[0]
                            / args.cutn_batches
                        )
                tv_losses = tv_loss(x_in)
                if args.use_secondary_model:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out["pred_xstart"])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = (
                    tv_losses.sum() * args.tv_scale
                    + range_losses.sum() * args.range_scale
                    + sat_losses.sum() * args.sat_scale
                )
                if init is not None and init_scale:
                    init_losses = self.lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any() == False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return (
                    grad * magnitude.clamp(max=args.clamp_max) / magnitude
                )  # min=-0.02, min=-clamp_max,
            return grad

        def symmetry_transformation_fn(x):
            if args.use_horizontal_symmetry:
                [n, c, h, w] = x.size()
                x = torch.concat(
                    (x[:, :, :, : w // 2], torch.flip(x[:, :, :, : w // 2], [-1])),
                    -1,
                )
                log.info("horizontal symmetry applied")
            if args.use_vertical_symmetry:
                [n, c, h, w] = x.size()
                x = torch.concat(
                    (x[:, :, : h // 2, :], torch.flip(x[:, :, : h // 2, :], [-2])),
                    -2,
                )
                log.info("vertical symmetry applied")
            return x

        if args.diffusion_sampling_mode == "ddim":
            sampling_fn = diffusion.ddim_sample_loop_progressive
        else:
            sampling_fn = diffusion.plms_sample_loop_progressive

        if args.perlin_init:
            init = regen_perlin(
                args.perlin_mode,
                args.batch_size,
                args.side_x,
                args.side_y,
                self.cuda_device,
            )

        if args.diffusion_sampling_mode == "ddim":
            progress_samples = sampling_fn(
                model,
                (args.batch_size, 3, args.side_y, args.side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                eta=args.eta,
                transformation_fn=symmetry_transformation_fn,
                transformation_percent=args.transformation_percent,
            )
        else:
            progress_samples = sampling_fn(
                model,
                (args.batch_size, 3, args.side_y, args.side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                order=2,
            )

        for prog_num, prog_sample in enumerate(progress_samples):
            cur_t -= 1
            intermediate_step = False
            if args.steps_per_checkpoint is not None:
                if prog_num % args.steps_per_checkpoint == 0 and prog_num > 0:
                    intermediate_step = True
            elif prog_num in args.intermediate_saves:
                intermediate_step = True

            with image_display:
                if (
                    prog_num % args.display_rate == 0
                    or cur_t == -1
                    or intermediate_step
                ):
                    for k, image in enumerate(prog_sample["pred_xstart"]):
                        # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                        if args.n_samples > 0:
                            # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                            if cur_t == -1 and args.intermediates_in_subfolder:
                                filename = f"{args.batch_name}({args.batch_num})_{frame_num:04}.png"
                            else:
                                filename = f"{args.batch_name}({args.batch_num})_{frame_num:04}-{prog_num:03}.png"
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        if prog_num % args.display_rate == 0 or cur_t == -1:
                            image.save(progress_path)
                            eKonf.clear_output(wait=True)
                            eKonf.display_image(progress_path)

                        if args.intermediates_in_subfolder:
                            _img_path = os.path.join(partial_dir, filename)
                        else:
                            _img_path = os.path.join(batch_dir, filename)

                        if args.steps_per_checkpoint is not None:
                            if (
                                prog_num % args.steps_per_checkpoint == 0
                                and prog_num > 0
                            ):
                                image.save(_img_path)
                        else:
                            if prog_num in args.intermediate_saves:
                                image.save(_img_path)
                        if cur_t == -1:
                            if args.animation_mode != AnimMode.NONE:
                                image.save(prev_frame_path)
                            _img_path = os.path.join(batch_dir, filename)
                            image.save(_img_path)
                            self.sample_imagepaths.append(_img_path)
                            log.info(f"Saved {filename}")
                            if args.animation_mode == AnimMode.ANIM_3D:
                                # If turbo, save a blended image
                                _img_path = os.path.join(batch_dir, filename)
                                if args.turbo_mode and frame_num > 0:
                                    # Mix new image with prevFrameScaled
                                    blend_factor = (1) / int(args.turbo_steps)
                                    # This is already updated..
                                    new_frame = cv2.imread(prev_frame_path)
                                    prev_frame_warped = cv2.imread(
                                        prev_frame_scaled_path
                                    )
                                    blended_image = cv2.addWeighted(
                                        new_frame,
                                        blend_factor,
                                        prev_frame_warped,
                                        (1 - blend_factor),
                                        0.0,
                                    )
                                    cv2.imwrite(_img_path, blended_image)
                                else:
                                    image.save(_img_path)

                                if args.vr_mode:
                                    generate_eye_views(
                                        self.TRANSLATION_SCALE,
                                        batch_dir,
                                        filename,
                                        frame_num,
                                        midas_model,
                                        midas_transform,
                                        self.cuda_device,
                                        args.vr_eye_angle,
                                        args.vr_ipd,
                                        args.near_plane,
                                        args.far_plane,
                                        args.fov,
                                        args.padding_mode,
                                        args.sampling_mode,
                                        args.midas_weight,
                                    )
        return loss_values

    def _init_2d(
        self,
        frame_num,
        seed,
        args,
    ):
        import cv2

        batch_dir = self._output.batch_dir
        prev_frame_path = self._output.prev_frame_path
        prev_frame_scaled_path = self._output.prev_frame_scaled_path

        if args.key_frames:
            angle = self.angle_series[frame_num]
            zoom = self.zoom_series[frame_num]
            translation_x = self.translation_x_series[frame_num]
            translation_y = self.translation_y_series[frame_num]
            log.info(
                f"angle: {angle}, zoom: {zoom}, translation_x: {translation_x}, translation_y: {translation_y}"
            )

        seed += 1
        if args.resume_run and frame_num == args.start_frame:
            _img_path = os.path.join(
                batch_dir,
                f"{args.batch_name}({args.batch_num})_{args.start_frame-1:04}.png",
            )
            img_0 = cv2.imread(_img_path)
        else:
            img_0 = cv2.imread(prev_frame_path)
        center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
        trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
        trans_mat = np.vstack([trans_mat, [0, 0, 1]])
        rot_mat = np.vstack([rot_mat, [0, 0, 1]])
        transformation_matrix = np.matmul(rot_mat, trans_mat)
        img_0 = cv2.warpPerspective(
            img_0,
            transformation_matrix,
            (img_0.shape[1], img_0.shape[0]),
            borderMode=cv2.BORDER_WRAP,
        )

        init_image = prev_frame_scaled_path
        cv2.imwrite(init_image, img_0)
        init_scale = args.frames_scale
        skip_steps = args.calc_frames_skip_steps
        return seed, init_image, init_scale, skip_steps

    def _init_3d(
        self,
        frame_num,
        seed,
        midas_model,
        midas_transform,
        args,
    ):
        import cv2
        from .diffuse import generate_eye_views

        batch_dir = self._output.batch_dir
        prev_frame_path = self._output.prev_frame_path
        prev_frame_scaled_path = self._output.prev_frame_scaled_path
        old_frame_scaled_path = self._output.old_frame_scaled_path
        skip_frame = False

        seed += 1
        if args.resume_run and frame_num == args.start_frame:
            img_filepath = os.path.join(
                batch_dir,
                f"{args.batch_name}({args.batch_num})_{args.start_frame-1:04}.png",
            )
            if args.turbo_mode and frame_num > args.turbo_preroll:
                shutil.copyfile(img_filepath, old_frame_scaled_path)
        else:
            img_filepath = prev_frame_path

        next_step_pil = self._3d_step(
            img_filepath, frame_num, midas_model, midas_transform, args
        )
        next_step_pil.save(prev_frame_scaled_path)

        ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
        if args.turbo_mode:
            if frame_num == args.turbo_preroll:  # start tracking oldframe
                # stash for later blending
                next_step_pil.save(old_frame_scaled_path)
            elif frame_num > args.turbo_preroll:
                # set up 2 warped image sequences, old & new, to blend toward new diff image
                old_frame = self._3d_step(
                    old_frame_scaled_path,
                    frame_num,
                    midas_model,
                    midas_transform,
                    args,
                )
                old_frame.save(old_frame_scaled_path)
                if frame_num % int(args.turbo_steps) != 0:
                    log.info("turbo skip this frame: skipping clip diffusion steps")
                    filename = f"{args.batch_name}({args.batch_num})_{frame_num:04}.png"
                    blend_factor = ((frame_num % int(args.turbo_steps)) + 1) / int(
                        args.turbo_steps
                    )
                    log.info(
                        "turbo skip this frame: skipping clip diffusion steps and saving blended frame"
                    )
                    # this is already updated..
                    newWarpedImg = cv2.imread(prev_frame_scaled_path)
                    oldWarpedImg = cv2.imread(old_frame_scaled_path)
                    blendedImage = cv2.addWeighted(
                        newWarpedImg,
                        blend_factor,
                        oldWarpedImg,
                        1 - blend_factor,
                        0.0,
                    )
                    _img_path = os.path.join(batch_dir, filename)
                    cv2.imwrite(_img_path, blendedImage)
                    # save it also as prev_frame to feed next iteration
                    next_step_pil.save(img_filepath)
                    if args.vr_mode:
                        generate_eye_views(
                            self.TRANSLATION_SCALE,
                            self._output.batch_dir,
                            filename,
                            frame_num,
                            midas_model,
                            midas_transform,
                            self.cuda_device,
                            args.vr_eye_angle,
                            args.vr_ipd,
                            args.near_plane,
                            args.far_plane,
                            args.fov,
                            args.padding_mode,
                            args.sampling_mode,
                            args.midas_weight,
                        )
                    skip_frame = True
                else:
                    # if not a skip frame, will run diffusion and need to blend.
                    oldWarpedImg = cv2.imread(prev_frame_scaled_path)
                    # swap in for blending later
                    cv2.imwrite(old_frame_scaled_path, oldWarpedImg)
                    log.info("clip/diff this frame - generate clip diff image")

        init_image = prev_frame_scaled_path
        init_scale = args.frames_scale
        skip_steps = args.calc_frames_skip_steps

        return seed, init_image, init_scale, skip_steps, skip_frame

    def _init_video(self, frame_num, seed, args):
        from .flow import warp

        batch_dir = self._output.batch_dir
        video_frames_dir = self._output.video_frames_dir
        warped_path = self._output.warped_path
        flo_dir = self._output.flo_dir

        init_scale = args.video_init_frames_scale
        skip_steps = args.calc_frames_skip_steps

        if not args.video_init_seed_continuity:
            seed += 1
        if args.video_init_flow_warp:
            if frame_num == 0:
                skip_steps = args.video_init_skip_steps
                init_image = os.path.join(video_frames_dir, f"{frame_num+1:04}.jpg")
            if frame_num > 0:
                _img_path = os.path.join(
                    batch_dir,
                    f"{args.batch_name}({args.batch_num})_{frame_num-1:04}.png",
                )
                prev = PIL.Image.open(_img_path)

                frame1_path = os.path.join(video_frames_dir, f"{frame_num:04}.jpg")
                frame2_path = os.path.join(video_frames_dir, f"{frame_num+1:04}.jpg")
                frame2 = PIL.Image.open(frame2_path)
                flo_path = os.path.join(
                    flo_dir,
                    f"{frame1_path.split('/')[-1]}.npy",
                )

                init_image = warped_path
                log.info(
                    f"warping frames with flow blend ratio: {args.video_init_flow_blend}"
                )
                weights_path = None
                if args.video_init_check_consistency:
                    # TBD
                    pass

                warp(
                    prev,
                    frame2,
                    flo_path,
                    blend=args.video_init_flow_blend,
                    weights_path=weights_path,
                ).save(init_image)

        else:
            init_image = os.path.join(video_frames_dir, f"{frame_num+1:04}.jpg")
        return seed, init_image, init_scale, skip_steps

    def _run_anim(self, args):
        """Run the simulation"""
        from .midas import Midas
        from ipywidgets import Output

        seed = args.seed
        diffusion = self.diffusion
        midas_model = None
        midas_transform = None
        cur_t = None
        batchBar = None

        text_series, image_series = self._get_prompt_series(args, args.max_frames)

        if (args.animation_mode == AnimMode.ANIM_3D) and (args.midas_weight > 0.0):
            midas = Midas(**self._midas)
            midas_model, midas_transform, _, _, _, _ = midas.init_midas_depth_model(
                args.midas_depth_model
            )

        log.info(f"looping over range({args.start_frame}, {args.max_frames})")
        for frame_num in range(args.start_frame, args.max_frames):
            # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
            if args.stop_on_next_loop:
                break

            eKonf.clear_output(wait=True)
            batchBar = tqdm(
                range(args.max_frames),
                desc=f"{args.batch_name}({args.batch_num}) frames",
            )
            batchBar.n = frame_num
            batchBar.refresh()

            init_image = None
            if os.path.exists(args.init_image):
                init_image = args.init_image
            init_scale = args.init_scale
            skip_steps = args.skip_steps

            if args.animation_mode == AnimMode.ANIM_2D:
                if frame_num > 0:
                    seed, init_image, init_scale, skip_steps = self._init_2d(
                        frame_num,
                        seed,
                        args,
                    )
            if args.animation_mode == AnimMode.ANIM_3D:
                if frame_num > 0:
                    (
                        seed,
                        init_image,
                        init_scale,
                        skip_steps,
                        skip_frame,
                    ) = self._init_3d(
                        frame_num,
                        seed,
                        midas_model,
                        midas_transform,
                        args,
                    )
                    if skip_frame:
                        continue
            if args.animation_mode == AnimMode.VIDEO_INPUT:
                seed, init_image, init_scale, skip_steps = self._init_video(
                    frame_num, seed, args
                )
            log.info(f"init_image: {init_image}")
            log.info(
                f"init_scale: {init_scale}, skip_steps: {skip_steps}, seed: {seed}"
            )

            loss_values = []

            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True

            text_prompt, image_prompt = self._get_prompt(
                text_series, image_series, frame_num
            )
            log.info(f"Image prompt: {image_prompt}")
            log.info(f"Frame {frame_num} Prompt: {text_prompt}")

            model_stats = self._get_model_stats(text_prompt, image_prompt, args)
            init = self._init_image(init_image, args)

            image_display = Output()

            eKonf.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1

            loss_values = self._generate_progress_samples(
                cur_t,
                init,
                init_scale,
                skip_steps,
                loss_values,
                model_stats,
                frame_num,
                diffusion,
                midas_model,
                midas_transform,
                image_display,
                args,
            )

            plt.plot(np.array(loss_values), "r")

        if batchBar is not None:
            batchBar.n = frame_num + 1
            batchBar.refresh()

    def _get_prompt_series(self, args, max_frames):
        text_prompts = eKonf.to_dict(args.text_prompts)
        image_prompts = eKonf.to_dict(args.image_prompts)
        text_series = split_prompts(text_prompts, max_frames) if text_prompts else None
        image_series = (
            split_prompts(image_prompts, max_frames) if image_prompts else None
        )
        return text_series, image_series

    def _get_prompt(self, text_series, image_series, frame_num):
        if text_series is not None and frame_num >= len(text_series):
            text_prompt = text_series[-1]
        elif text_series is not None:
            text_prompt = text_series[frame_num]
        else:
            text_prompt = []

        if image_series is not None and frame_num >= len(image_series):
            image_prompt = image_series[-1]
        elif image_series is not None:
            image_prompt = image_series[frame_num]
        else:
            image_prompt = []
        return text_prompt, image_prompt

    def _run(self, args):
        """Run the simulation"""
        from ipywidgets import Output

        seed = args.seed
        diffusion = self.diffusion

        midas_model = None
        midas_transform = None
        init_image = None
        cur_t = None
        batchBar = None
        loss_values = []

        text_series, image_series = self._get_prompt_series(args, args.n_samples)

        if os.path.exists(args.init_image):
            init_image = args.init_image
        init_scale = args.init_scale
        skip_steps = args.skip_steps
        log.info(f"init_image: {init_image}")
        log.info(f"init_scale: {init_scale}, skip_steps: {skip_steps}, seed: {seed}")

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        image_display = Output()
        log.info(f"looping over range({args.start_sample}, {args.n_samples})")
        for sample_num in range(args.start_sample, args.n_samples):
            # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
            if args.stop_on_next_loop:
                break

            eKonf.clear_output(wait=True)
            batchBar = tqdm(
                range(args.n_samples),
                desc=f"{args.batch_name}({args.batch_num}) samples",
            )
            batchBar.n = sample_num
            batchBar.refresh()

            text_prompt, image_prompt = self._get_prompt(
                text_series, image_series, sample_num
            )
            log.info(f"Image prompt: {image_prompt}")
            log.info(f"Sample {sample_num} Prompt: {text_prompt}")

            model_stats = self._get_model_stats(text_prompt, image_prompt, args)
            init = self._init_image(init_image, args)

            eKonf.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1

            loss_values = self._generate_progress_samples(
                cur_t,
                init,
                init_scale,
                skip_steps,
                loss_values,
                model_stats,
                sample_num,
                diffusion,
                midas_model,
                midas_transform,
                image_display,
                args,
            )

        if batchBar is not None:
            batchBar.n = sample_num + 1
            batchBar.refresh()
            plt.plot(np.array(loss_values), "r")

    def _prepare_config(self, args):
        batch_dir = self._output.batch_dir
        video_frames_dir = self._output.video_frames_dir
        flo_dir = self._output.flo_dir

        # Ensure prompts formatted correctly
        text_prompts = eKonf.to_dict(args.text_prompts)
        image_prompts = eKonf.to_dict(args.image_prompts)
        if isinstance(text_prompts, str):
            text_prompts = {0: [text_prompts]}
            log.info(f"converted string type text_prompts to {text_prompts}")
        elif isinstance(text_prompts, list):
            text_prompts = {0: text_prompts}
            log.info(f"converted list type text_prompts to {text_prompts}")
        if isinstance(image_prompts, str):
            image_prompts = {0: [image_prompts]}
            log.info(f"converted string type image_prompts to {image_prompts}")
        elif isinstance(image_prompts, list):
            image_prompts = {0: image_prompts}
            log.info(f"converted list type image_prompts to {image_prompts}")
        args.text_prompts = text_prompts
        args.image_prompts = image_prompts

        # Get corrected sizes
        args.side_x = (args.width_height[0] // 64) * 64
        args.side_y = (args.width_height[1] // 64) * 64
        if args.side_x != args.width_height[0] or args.side_y != args.width_height[1]:
            log.info(
                f"Changing output size to {args.side_x}x{args.side_y}. Dimensions must by multiples of 64."
            )

        _mode = AnimMode.NONE.value
        for mode in AnimMode:
            if mode.value.lower() == args.animation_mode.lower():
                _mode = mode.value
                break
        args.animation_mode = _mode

        if args.animation_mode == AnimMode.VIDEO_INPUT:
            args.steps = args.video_init_steps
            extract_frames(
                args.video_init_path,
                args.extract_nth_frame,
                video_frames_dir,
            )
            args.max_frames = len(glob(f"{video_frames_dir}/*.jpg"))

        # insist turbo be used only w 3d anim.
        if args.turbo_mode and args.animation_mode != AnimMode.ANIM_3D:
            log.info("Turbo mode only available with 3D animations. Disabling Turbo.")
            args.turbo_mode = False

        # insist VR be used only w 3d anim.
        if args.vr_mode and args.animation_mode != AnimMode.ANIM_3D:
            log.info("VR mode only available with 3D animations. Disabling VR.")
            args.vr_mode = False

        if args.key_frames:
            try:
                self.angle_series = get_inbetweens(
                    parse_key_frames(args.angle), args.max_frames, args.interp_spline
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `angle` correctly for key frames.\n"
                    "Attempting to interpret `angle` as "
                    f'"0: ({args.angle})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.angle = f"0: ({args.angle})"
                self.angle_series = get_inbetweens(
                    parse_key_frames(args.angle), args.max_frames, args.interp_spline
                )

            try:
                self.zoom_series = get_inbetweens(
                    parse_key_frames(args.zoom), args.max_frames, args.interp_spline
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `zoom` correctly for key frames.\n"
                    "Attempting to interpret `zoom` as "
                    f'"0: ({args.zoom})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.zoom = f"0: ({args.zoom})"
                self.zoom_series = get_inbetweens(
                    parse_key_frames(args.zoom), args.max_frames, args.interp_spline
                )

            try:
                self.translation_x_series = get_inbetweens(
                    parse_key_frames(args.translation_x),
                    args.max_frames,
                    args.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_x` correctly for key frames.\n"
                    "Attempting to interpret `translation_x` as "
                    f'"0: ({args.translation_x})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.translation_x = f"0: ({args.translation_x})"
                self.translation_x_series = get_inbetweens(
                    parse_key_frames(args.translation_x),
                    args.max_frames,
                    args.interp_spline,
                )

            try:
                self.translation_y_series = get_inbetweens(
                    parse_key_frames(args.translation_y),
                    args.max_frames,
                    args.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_y` correctly for key frames.\n"
                    "Attempting to interpret `translation_y` as "
                    f'"0: ({args.translation_y})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.translation_y = f"0: ({args.translation_y})"
                self.translation_y_series = get_inbetweens(
                    parse_key_frames(args.translation_y),
                    args.max_frames,
                    args.interp_spline,
                )

            try:
                self.translation_z_series = get_inbetweens(
                    parse_key_frames(args.translation_z),
                    args.max_frames,
                    args.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_z` correctly for key frames.\n"
                    "Attempting to interpret `translation_z` as "
                    f'"0: ({args.translation_z})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.translation_z = f"0: ({args.translation_z})"
                self.translation_z_series = get_inbetweens(
                    parse_key_frames(args.translation_z),
                    args.max_frames,
                    args.interp_spline,
                )

            try:
                self.rotation_3d_x_series = get_inbetweens(
                    parse_key_frames(args.rotation_3d_x),
                    args.max_frames,
                    args.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_x` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_x` as "
                    f'"0: ({args.rotation_3d_x})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.rotation_3d_x = f"0: ({args.rotation_3d_x})"
                self.rotation_3d_x_series = get_inbetweens(
                    parse_key_frames(args.rotation_3d_x),
                    args.max_frames,
                    args.interp_spline,
                )

            try:
                self.rotation_3d_y_series = get_inbetweens(
                    parse_key_frames(args.rotation_3d_y),
                    args.max_frames,
                    args.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_y` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_y` as "
                    f'"0: ({args.rotation_3d_y})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.rotation_3d_y = f"0: ({args.rotation_3d_y})"
                self.rotation_3d_y_series = get_inbetweens(
                    parse_key_frames(args.rotation_3d_y),
                    args.max_frames,
                    args.interp_spline,
                )

            try:
                self.rotation_3d_z_series = get_inbetweens(
                    parse_key_frames(args.rotation_3d_z),
                    args.max_frames,
                    args.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_z` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_z` as "
                    f'"0: ({args.rotation_3d_z})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                args.rotation_3d_z = f"0: ({args.rotation_3d_z})"
                self.rotation_3d_z_series = get_inbetweens(
                    parse_key_frames(args.rotation_3d_z),
                    args.max_frames,
                    args.interp_spline,
                )

        else:
            args.angle = float(args.angle)
            args.zoom = float(args.zoom)
            args.translation_x = float(args.translation_x)
            args.translation_y = float(args.translation_y)
            args.translation_z = float(args.translation_z)
            args.rotation_3d_x = float(args.rotation_3d_x)
            args.rotation_3d_y = float(args.rotation_3d_y)
            args.rotation_3d_z = float(args.rotation_3d_z)

        if eKonf.is_list(args.intermediate_saves):
            args.steps_per_checkpoint = None
        else:
            if (
                not isinstance(args.steps_per_checkpoint, int)
                or args.steps_per_checkpoint < 1
            ):
                if args.intermediate_saves and args.intermediate_saves > 0:
                    args.steps_per_checkpoint = math.floor(
                        (args.steps - args.skip_steps - 1)
                        // (args.intermediate_saves + 1)
                    )
                    args.steps_per_checkpoint = (
                        args.steps_per_checkpoint
                        if args.steps_per_checkpoint > 0
                        else 1
                    )
                else:
                    args.steps_per_checkpoint = None
                    # args.steps_per_checkpoint = args.steps + 10
            log.info(f"Will save every {args.steps_per_checkpoint} steps")
            args.intermediate_saves = []

        timestep_respacing = f"ddim{args.steps}"
        diffusion_steps = (
            (1000 // args.steps) * args.steps if args.steps < 1000 else args.steps
        )
        self.model_config.update(
            {
                "timestep_respacing": timestep_respacing,
                "diffusion_steps": diffusion_steps,
            }
        )
        skip_step_ratio = int(args.frames_skip_steps.rstrip("%")) / 100
        args.calc_frames_skip_steps = math.floor(args.steps * skip_step_ratio)

        if args.animation_mode == AnimMode.VIDEO_INPUT:
            self.init_flow_warp(args)

            frames = sorted(glob(video_frames_dir + "/*.*"))
            if len(frames) == 0:
                raise Exception(f"ERROR: 0 frames found in {video_frames_dir}.")
            flows = glob(flo_dir + "/*.*")
            if (len(flows) == 0) and args.video_init_flow_warp:
                raise Exception(f"ERROR: 0 flow files found in {flo_dir}.")

        if args.steps <= args.calc_frames_skip_steps:
            raise Exception("ERROR: You can't skip more steps than your total steps")

        batch_config_file = os.path.join(
            batch_dir, f"{args.batch_name}(*)_settings.yaml"
        )
        if args.resume_run:
            if args.batch_num is None:
                if args.run_to_resume == "latest":
                    args.batch_num = len(glob(batch_config_file)) - 1
                    log.info("Resuming latest batch")
                else:
                    args.batch_num = int(args.run_to_resume)
                    log.info("Resuming batch with run_to_resume as batch_num")
            log.info(f"Resuming batch_num: {args.batch_num}")

            _frame_file = os.path.join(
                batch_dir, f"{args.batch_name}({args.batch_num})_*.png"
            )
            if args.resume_from_frame == "latest":
                start_frame = len(glob(_frame_file))
                if (
                    args.animation_mode != AnimMode.ANIM_3D
                    and args.turbo_mode == True
                    and start_frame > args.turbo_preroll
                    and start_frame % int(args.turbo_steps) != 0
                ):
                    start_frame = start_frame - (start_frame % int(args.turbo_steps))
            else:
                start_frame = int(args.resume_from_frame) + 1
                if (
                    args.animation_mode != AnimMode.ANIM_3D
                    and args.turbo_mode == True
                    and start_frame > args.turbo_preroll
                    and start_frame % int(args.turbo_steps) != 0
                ):
                    start_frame = start_frame - (start_frame % int(args.turbo_steps))
                if args.retain_overwritten_frames:
                    existing_frames = len(glob(_frame_file))
                    frames_to_save = existing_frames - start_frame
                    log.info(f"Moving {frames_to_save} frames to the Retained folder")
                    move_files(
                        start_frame,
                        existing_frames,
                        self._output.batch_dir,
                        self._output.retain_dir,
                        args.batch_name,
                        args.batch_num,
                    )
        else:
            start_frame = 0
            args.batch_num = len(glob(batch_config_file))
        args.start_frame = start_frame
        args.start_sample = start_frame

        if args.set_seed == "random_seed":
            random.seed()
            args.seed = random.randint(0, 2 ** 32)
        else:
            args.seed = int(args.set_seed)
        log.info(f"Using seed: {args.seed}")

        args.n_samples = args.n_samples if args.animation_mode == AnimMode.NONE else 1
        args.max_frames = args.max_frames if args.animation_mode != AnimMode.NONE else 1

        if args.animation_mode == AnimMode.VIDEO_INPUT:
            # This isn't great in terms of what will get saved to the settings.. but it should work.
            args.clip_guidance_scale = args.video_init_clip_guidance_scale
            args.tv_scale = args.video_init_tv_scale
            args.range_scale = args.video_init_range_scale
            args.sat_scale = args.video_init_sat_scale
            args.cutn_batches = args.video_init_cutn_batches
            args.skip_steps = args.video_init_skip_steps
            args.frames_scale = args.video_init_frames_scale
            args.frames_skip_steps = args.video_init_frames_skip_steps

        return args

    def make_gif(
        self,
        batch_name=None,
        batch_num=None,
        sample_num=0,
        output_file=None,
        filename_patterns=None,
        duration=100,
        loop=0,
        width=None,
        optimize=True,
        quality=50,
        show=False,
        force=False,
        **kwargs,
    ):
        args = self.load_config(batch_name, batch_num, **kwargs)
        batch_name = batch_name or args.batch_name
        if batch_num is None:
            batch_num = args.batch_num
        base_dir = self._output.partial_dir

        filename_patterns = (
            filename_patterns or f"{batch_name}({batch_num})_{sample_num:04}-*.png"
        )
        output_file = output_file or f"{batch_name}({batch_num})_{sample_num:04}.gif"
        output_path = os.path.join(self._output.batch_dir, output_file)

        eKonf.make_gif(
            filename_patterns=filename_patterns,
            base_dir=base_dir,
            output_filepath=output_path,
            duration=duration,
            loop=loop,
            width=width,
            optimize=optimize,
            quality=quality,
            show=show,
            force=force,
            **kwargs,
        )

    def _prepare_folders(self, batch_name):
        super()._prepare_folders(batch_name)
        o = self._output
        batch_dir = o.batch_dir
        o.retain_dir = os.path.join(batch_dir, "retained")
        o.partial_dir = os.path.join(batch_dir, "partials")
        o.video_frames_dir = os.path.join(batch_dir, "video_frames")
        o.flo_dir = os.path.join(o.video_frames_dir, "out_flo_fwd")
        o.temp_flo_dir = os.path.join(o.video_frames_dir, "temp_flo")
        o.flo_fwd_dir = os.path.join(o.video_frames_dir, "out_flo_fwd")
        o.flo_out_dir = os.path.join(batch_dir, "flow")
        o.blend_out_dir = os.path.join(batch_dir, "blend")
        for _name, _path in o.items():
            if _name.endswith("_dir") and not os.path.exists(_path):
                os.makedirs(_path)
        o.prev_frame_path = os.path.join(batch_dir, "prev_frame.png")
        o.prev_frame_scaled_path = os.path.join(batch_dir, "prev_frame_scaled.png")
        o.progress_path = os.path.join(batch_dir, "progress.png")
        o.warped_path = os.path.join(batch_dir, "warped.png")
        o.old_frame_scaled_path = os.path.join(batch_dir, "old_frame_scaled.png")

    def create_video(self, batch_name=None, batch_num=None, force=False, **kwargs):
        import PIL
        from tqdm.notebook import trange
        from .flow import warp

        args = self.load_config(batch_name, batch_num)
        batch_name = batch_name or args.batch_name
        if batch_num is None:
            batch_num = args.batch_num
        # @title ### **Create video**
        # @markdown Video file will save in the same folder as your images.

        _video = args.video_output
        _video = eKonf.merge(_video, kwargs)

        flo_dir = self._output.flo_dir
        batch_dir = self._output.batch_dir
        flo_out_dir = self._output.flo_out_dir
        blend_out_dir = self._output.blend_out_dir

        if _video.skip_video_for_run_all == True:
            print(
                "Skipping video creation, set skip_video_for_run_all=False if you want to run it"
            )

        else:
            frame_files = f"{batch_name}({batch_num})_*.png"
            frames_path = os.path.join(batch_dir, frame_files)
            frames_in = glob(frames_path)

            if _video.last_frame == "final_frame":
                _video.last_frame = len(frames_in)
                log.info(f"Total frames: {_video.last_frame}")

            frame_filename = f"{batch_name}({batch_num})_%04d.png"
            frames_in_path = os.path.join(batch_dir, frame_filename)
            mp4_filename = f"{batch_name}({batch_num}).mp4"

            if (args.video_init_blend_mode == "optical flow") and (
                args.animation_mode == AnimMode.VIDEO_INPUT
            ):
                frames_path = os.path.join(flo_out_dir, frame_files)
                if _video.last_frame == "final_frame":
                    _video.last_frame = len(glob(frames_path))

                frames_in_path = os.path.join(flo_out_dir, frame_filename)
                mp4_filename = f"{batch_name}({batch_num})_flow.mp4"
                shutil.copy(frames_in[0], flo_out_dir)
                log.info(f"Copying first frame to {flo_out_dir}")

                for i in trange(
                    _video.init_frame, min(len(frames_in), _video.last_frame)
                ):
                    frame1_path = frames_in[i - 1]
                    frame2_path = frames_in[i]

                    frame1 = PIL.Image.open(frame1_path)
                    frame2 = PIL.Image.open(frame2_path)
                    frame1_stem = f"{(int(frame1_path.split('/')[-1].split('_')[-1][:-4])+1):04}.jpg"

                    _flo_path = os.path.join(flo_dir, f"{frame1_stem}.npy")
                    log.info(f"Loading flow from {_flo_path}")
                    weights_path = None
                    if _video.video_init_check_consistency:
                        # TBD
                        pass

                    _filename = f"{batch_name}({batch_num})_{i:04}.png"
                    _frame_path = os.path.join(flo_out_dir, _filename)
                    warp(
                        frame1,
                        frame2,
                        _flo_path,
                        blend=_video.blend,
                        weights_path=weights_path,
                    ).save(_frame_path)
                    log.info(f"Saving warped frame {_frame_path}")

            if args.video_init_blend_mode == "linear":
                frames_path = os.path.join(blend_out_dir, frame_files)
                if _video.last_frame == "final_frame":
                    _video.last_frame = len(glob(frames_path))

                frames_in_path = os.path.join(blend_out_dir, frame_filename)
                mp4_filename = f"{batch_name}({batch_num})_blend.mp4"
                shutil.copy(frames_in[0], blend_out_dir)
                log.info(f"Copying first frame to {blend_out_dir}")

                for i in trange(1, len(frames_in)):
                    frame1_path = frames_in[i - 1]
                    frame2_path = frames_in[i]

                    frame1 = PIL.Image.open(frame1_path)
                    frame2 = PIL.Image.open(frame2_path)

                    _filename = f"{batch_name}({batch_num})_{i:04}.png"
                    _frame_path = os.path.join(blend_out_dir, _filename)
                    frame = PIL.Image.fromarray(
                        (
                            np.array(frame1) * (1 - _video.blend)
                            + np.array(frame2) * (_video.blend)
                        ).astype("uint8")
                    ).save(_frame_path)
                    log.info(f"Saving blended frame {_frame_path}")

            mp4_path = os.path.join(batch_dir, mp4_filename)

            return _create_video(
                base_dir=batch_dir,
                video_path=mp4_path,
                input_url=frames_in_path,
                fps=_video.fps,
                start_number=_video.init_frame,
                vframes=(_video.last_frame + 1),
                force=force,
            )
