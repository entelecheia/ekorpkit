import os
import logging
import subprocess
import math
import torch
import random
import numpy as np
import gc
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
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


log = logging.getLogger(__name__)


class DiscoDiffusion:

    TRANSLATION_SCALE = 1.0 / 200.0

    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.verbose = args.get("verbose", True)
        self.auto = args.auto
        self._path = self.args.path
        self._output = self.args.output
        self._module = self.args.module
        self._midas = self.args.midas
        self._model = self.args.model
        self._diffuse = self.args.diffuse

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
        self.prompts_series = None
        self.image_prompts_series = None

        self.is_notebook = eKonf.is_notebook()
        self.is_colab = eKonf.is_colab()

        if self.auto.load:
            self.load()

    @property
    def path(self):
        return self._path

    def load(self):
        log.info("> downloading models...")
        self.download_models()
        log.info("> loading modules...")
        self.load_modules()
        log.info("> loading diffusion models...")
        self.load_diffusion_models()
        log.info("> loading clip models...")
        self.load_clip_models()

    def diffuse(self, text_prompts=None, image_prompts=None, **args):
        """Diffuse the model"""

        log.info("> loading settings...")
        args = self.load_config(**args)
        log.info("> loading optical flow...")
        self.load_optical_flow(args)

        log.info(
            f"Starting Run: {args.batch_name}({args.batch_num}) at frame {args.start_frame}"
        )

        text_prompts = text_prompts or eKonf.to_dict(args.text_prompts)
        image_prompts = image_prompts or eKonf.to_dict(args.image_prompts)
        if isinstance(text_prompts, str):
            text_prompts = {0: [text_prompts]}
        elif isinstance(text_prompts, list):
            text_prompts = {0: text_prompts}
        if isinstance(image_prompts, str):
            image_prompts = {0: [image_prompts]}
        elif isinstance(image_prompts, list):
            image_prompts = {0: image_prompts}
        self.prompts_series = (
            split_prompts(text_prompts, args.max_frames) if text_prompts else None
        )
        self.image_prompts_series = (
            split_prompts(image_prompts, args.max_frames) if image_prompts else None
        )
        args.text_prompts = text_prompts
        args.image_prompts = image_prompts
        self._diffuse = args

        self._prepare_models()

        gc.collect()
        torch.cuda.empty_cache()
        try:
            self._run(args)
        except KeyboardInterrupt:
            pass
        finally:
            log.info(f"Seed used: {args.seed}")
            gc.collect()
            torch.cuda.empty_cache()

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

    def load_optical_flow(self, args):
        """Load the optical flow"""
        import argparse
        from raft import RAFT
        from .flow import load_img, get_flow

        if args.animation_mode == "Video Input":
            rsft_args = argparse.Namespace()
            rsft_args.small = False
            rsft_args.mixed_precision = True

            if not args.video_init_flow_warp:
                log.info("video_init_flow_warp not set, skipping")

            else:
                flows = glob(self._output.flo_dir + "/*.*")
                if (len(flows) > 0) and not args.force_flow_generation:
                    log.info(
                        f"Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {self._output.flo_dir}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again."
                    )

                if (len(flows) == 0) or args.force_flow_generation:
                    frames = sorted(glob(self._output.in_dir + "/*.*"))
                    if len(frames) < 2:
                        log.warning(
                            f"WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path."
                        )
                    if len(frames) >= 2:

                        raft_model = torch.nn.DataParallel(RAFT(rsft_args))
                        raft_model.load_state_dict(
                            torch.load(self._model.raft_model_path)
                        )
                        raft_model = raft_model.module.cuda().eval()

                        for f in Path(f"{self._output.flo_fwd_dir}").glob("*.*"):
                            f.unlink()

                        # TBD Call out to a consistency checker?
                        self.framecount = 0
                        for frame1, frame2 in tqdm(
                            zip(frames[:-1], frames[1:]), total=len(frames) - 1
                        ):

                            out_flow21_fn = (
                                f"{self._output.flo_fwd_dir}/{frame1.split('/')[-1]}"
                            )

                            frame1 = load_img(frame1, args.width_height)
                            frame2 = load_img(frame2, args.width_height)

                            flow21 = get_flow(frame2, frame1, raft_model)
                            np.save(out_flow21_fn, flow21)

                            if args.video_init_check_consistency:
                                # TBD
                                pass

                        del raft_model
                        gc.collect()

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
        download = self.args.download
        check_model_SHA = download.check_model_SHA
        for name, model in download.models.items():
            if not isinstance(model, str):
                log.info(f"Downloading model {name} from {model}")
                _download_models(name, **model, check_model_SHA=check_model_SHA)

    def save_settings(self, args):
        """Save the settings"""
        _path = os.path.join(
            self._output.batch_dir, f"{args.batch_name}({args.batch_num})_settings.yaml"
        )
        eKonf.save(args, _path)

    def _3d_step(self, img_filepath, frame_num, midas_model, midas_transform):
        import py3d_tools as p3dT
        from . import disco_xform_utils as dxf

        args = self._diffuse
        if args.key_frames:
            translation_x = self.translation_x_series[frame_num]
            translation_y = self.translation_y_series[frame_num]
            translation_z = self.translation_z_series[frame_num]
            rotation_3d_x = self.rotation_3d_x_series[frame_num]
            rotation_3d_y = self.rotation_3d_y_series[frame_num]
            rotation_3d_z = self.rotation_3d_z_series[frame_num]
            print(
                f"translation_x: {translation_x}",
                f"translation_y: {translation_y}",
                f"translation_z: {translation_z}",
                f"rotation_3d_x: {rotation_3d_x}",
                f"rotation_3d_y: {rotation_3d_y}",
                f"rotation_3d_z: {rotation_3d_z}",
            )

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

    def _run(self, args):
        """Run the simulation"""
        import cv2
        import clip
        import torchvision.transforms as T
        import torchvision.transforms.functional as TF
        import PIL
        from PIL import Image
        from .midas import Midas
        from .diffuse import (
            generate_eye_views,
            parse_prompt,
            MakeCutouts,
            fetch,
            create_perlin_noise,
            MakeCutoutsDango,
            spherical_dist_loss,
            tv_loss,
            range_loss,
            regen_perlin,
        )
        from .secondary import alpha_sigma_to_t
        from .flow import warp
        from IPython import display
        from ipywidgets import Output

        seed = args.seed
        batch_dir = self._output.batch_dir
        old_frame_scaled_path = os.path.join(batch_dir, "old_frame_scaled.png")
        prev_frame_path = os.path.join(batch_dir, "prev_frame.png")
        prev_frame_scaled_path = os.path.join(batch_dir, "prev_frame_scaled.png")
        warped_path = os.path.join(batch_dir, "warped.png")
        progress_path = os.path.join(batch_dir, "progress.png")

        if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
            midas = Midas(**self._midas)
            (
                midas_model,
                midas_transform,
                midas_net_w,
                midas_net_h,
                midas_resize_mode,
                midas_normalization,
            ) = midas.init_midas_depth_model(args.midas_depth_model)

        if isinstance(args.cut_overview, str):
            cut_overview = eval(args.cut_overview)
            cut_innercut = eval(args.cut_innercut)
            cut_icgray_p = eval(args.cut_icgray_p)

        log.info(f"looping over range({args.start_frame}, {args.max_frames})")
        for frame_num in range(args.start_frame, args.max_frames):
            # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
            if args.stop_on_next_loop:
                break

            if self.is_notebook:
                display.clear_output(wait=True)

            # Print Frame progress if animation mode is on
            if args.animation_mode != "None":
                batchBar = tqdm(range(args.max_frames), desc="Frames")
                batchBar.n = frame_num
                batchBar.refresh()

            # Inits if not video frames
            if args.animation_mode != "Video Input":
                if args.init_image in ["", "none", "None", "NONE"]:
                    init_image = None
                else:
                    init_image = args.init_image
                init_scale = args.init_scale
                skip_steps = args.skip_steps

            if args.animation_mode == "2D":
                if args.key_frames:
                    angle = self.angle_series[frame_num]
                    zoom = self.zoom_series[frame_num]
                    translation_x = self.translation_x_series[frame_num]
                    translation_y = self.translation_y_series[frame_num]
                    log.info(
                        f"angle: {angle}, zoom: {zoom}, translation_x: {translation_x}, translation_y: {translation_y}"
                    )

                if frame_num > 0:
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
                    trans_mat = np.float32(
                        [[1, 0, translation_x], [0, 1, translation_y]]
                    )
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

            if args.animation_mode == "3D":
                if frame_num > 0:
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
                        img_filepath, frame_num, midas_model, midas_transform
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
                            )
                            old_frame.save(old_frame_scaled_path)
                            if frame_num % int(args.turbo_steps) != 0:
                                log.info(
                                    "turbo skip this frame: skipping clip diffusion steps"
                                )
                                filename = f"{args.batch_name}({args.batch_num})_{frame_num:04}.png"
                                blend_factor = (
                                    (frame_num % int(args.turbo_steps)) + 1
                                ) / int(args.turbo_steps)
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
                                continue
                            else:
                                # if not a skip frame, will run diffusion and need to blend.
                                oldWarpedImg = cv2.imread(prev_frame_scaled_path)
                                # swap in for blending later
                                cv2.imwrite(old_frame_scaled_path, oldWarpedImg)
                                log.info(
                                    "clip/diff this frame - generate clip diff image"
                                )

                    init_image = prev_frame_scaled_path
                    init_scale = args.frames_scale
                    skip_steps = args.calc_frames_skip_steps

            if args.animation_mode == "Video Input":
                init_scale = args.video_init_frames_scale
                skip_steps = args.calc_frames_skip_steps
                if not args.video_init_seed_continuity:
                    seed += 1
                if args.video_init_flow_warp:
                    if frame_num == 0:
                        skip_steps = args.video_init_skip_steps
                        init_image = os.path.join(
                            args.video_frames_dir, f"{frame_num+1:04}.jpg"
                        )
                    if frame_num > 0:
                        _img_path = os.path.join(
                            batch_dir,
                            f"/{args.batch_name}({args.batch_num})_{frame_num-1:04}.png",
                        )
                        prev = PIL.Image.open(_img_path)

                        frame1_path = os.path.join(
                            args.video_frames_dir, f"{frame_num:04}.jpg"
                        )
                        frame2_path = os.path.join(
                            args.video_frames_dir, f"{frame_num+1:04}.jpg"
                        )
                        frame2 = PIL.Image.open(frame2_path)
                        flo_path = os.path.join(
                            args.self._video.flo_dir,
                            f"{frame1_path.split('/')[-1]}.npy",
                        )

                        init_image = warped_path
                        log(args.video_init_flow_blend)
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
                    init_image = os.path.join(
                        args.video_frames_dir, f"{frame_num+1:04}.jpg"
                    )

            loss_values = []

            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True

            target_embeds, weights = [], []

            if self.prompts_series is not None and frame_num >= len(
                self.prompts_series
            ):
                frame_prompt = self.prompts_series[-1]
            elif self.prompts_series is not None:
                frame_prompt = self.prompts_series[frame_num]
            else:
                frame_prompt = []

            if self.image_prompts_series is not None and frame_num >= len(
                self.image_prompts_series
            ):
                image_prompt = self.image_prompts_series[-1]
            elif self.image_prompts_series is not None:
                image_prompt = self.image_prompts_series[frame_num]
            else:
                image_prompt = []
            log.info(f"Image prompt: {image_prompt}")
            log.info(f"Frame {frame_num} Prompt: {frame_prompt}")

            model_stats = []
            for clip_model in self.clip_models:
                cutn = 16
                model_stat = {
                    "clip_model": None,
                    "target_embeds": [],
                    "make_cutouts": None,
                    "weights": [],
                }
                model_stat["clip_model"] = clip_model

                for prompt in frame_prompt:
                    txt, weight = parse_prompt(prompt)
                    txt = clip_model.encode_text(
                        clip.tokenize(prompt).to(self.cuda_device)
                    ).float()

                    if args.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (
                                    txt + torch.randn(txt.shape).cuda() * args.rand_mag
                                ).clamp(0, 1)
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
                                        + torch.randn(embed.shape).cuda()
                                        * args.rand_mag
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

            init = None
            if init_image is not None:
                init = Image.open(fetch(init_image)).convert("RGB")
                init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
                init = (
                    TF.to_tensor(init).to(self.cuda_device).unsqueeze(0).mul(2).sub(1)
                )

            if args.perlin_init:
                if args.perlin_mode == "color":
                    init = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
                    )
                    init2 = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(8)], 4, 4, False
                    )
                elif args.perlin_mode == "gray":
                    init = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(12)], 1, 1, True
                    )
                    init2 = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(8)], 4, 4, True
                    )
                else:
                    init = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
                    )
                    init2 = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(8)], 4, 4, True
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

            cur_t = None

            cuda_device = self.cuda_device
            model = self.model
            diffusion = self.diffusion
            secondary_model = self.secondary_model

            def cond_fn(x, t, y=None):
                with torch.enable_grad():
                    x_is_NaN = False
                    x = x.detach().requires_grad_()
                    n = x.shape[0]
                    if self._model.use_secondary_model:
                        alpha = torch.tensor(
                            diffusion.sqrt_alphas_cumprod[cur_t],
                            device=cuda_device,
                            dtype=torch.float32,
                        )
                        sigma = torch.tensor(
                            diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                            device=cuda_device,
                            dtype=torch.float32,
                        )
                        cosine_t = alpha_sigma_to_t(alpha, sigma)
                        out = secondary_model(x, cosine_t[None].repeat([n])).pred
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)
                    else:
                        my_t = (
                            torch.ones([n], device=cuda_device, dtype=torch.long)
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
                                    cut_overview[1000 - t_int]
                                    + cut_innercut[1000 - t_int],
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
                    if self._model.use_secondary_model:
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

            if self._model.diffusion_sampling_mode == "ddim":
                sample_fn = diffusion.ddim_sample_loop_progressive
            else:
                sample_fn = diffusion.plms_sample_loop_progressive

            image_display = Output()
            for i in range(args.n_samples):
                if args.animation_mode == "None":
                    display.clear_output(wait=True)
                    batchBar = tqdm(
                        range(args.n_samples),
                        desc=f"{args.batch_name}({args.batch_num})",
                    )
                    batchBar.n = i
                    batchBar.refresh()
                print("")
                display.display(image_display)
                gc.collect()
                torch.cuda.empty_cache()
                cur_t = diffusion.num_timesteps - skip_steps - 1
                total_steps = cur_t

                if args.perlin_init:
                    init = regen_perlin(
                        args.perlin_mode,
                        args.batch_size,
                        args.side_x,
                        args.side_y,
                        self.cuda_device,
                    )

                if self._model.diffusion_sampling_mode == "ddim":
                    samples = sample_fn(
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
                        # transformation_fn=symmetry_transformation_fn,
                        # transformation_percent=_extra.transformation_percent,
                    )
                else:
                    samples = sample_fn(
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

                # with run_display:
                # display.clear_output(wait=True)
                for j, sample in enumerate(samples):
                    cur_t -= 1
                    intermediate_step = False
                    if args.steps_per_checkpoint is not None:
                        if j % args.steps_per_checkpoint == 0 and j > 0:
                            intermediate_step = True
                    elif j in args.intermediate_saves:
                        intermediate_step = True
                    with image_display:
                        if (
                            j % args.display_rate == 0
                            or cur_t == -1
                            or intermediate_step == True
                        ):
                            for k, image in enumerate(sample["pred_xstart"]):
                                # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                                current_time = datetime.now().strftime(
                                    "%y%m%d-%H%M%S_%f"
                                )
                                percent = math.ceil(j / total_steps * 100)
                                if args.n_samples > 0:
                                    # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                    if cur_t == -1 and args.intermediates_in_subfolder:
                                        save_num = (
                                            frame_num
                                            if args.animation_mode != "None"
                                            else i
                                        )
                                        filename = f"{args.batch_name}({args.batch_num})_{save_num:04}.png"
                                    else:
                                        # If we're working with percentages, append it
                                        if args.steps_per_checkpoint is not None:
                                            # filename = f"{args.batch_name}({args.batch_num})_{i:04}-{percent:03}%.png"
                                            filename = f"{args.batch_name}({args.batch_num})_{i:04}-{j:03}.png"
                                        # Or else, iIf we're working with specific steps, append those
                                        else:
                                            filename = f"{args.batch_name}({args.batch_num})_{i:04}-{j:03}.png"
                                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                                if j % args.display_rate == 0 or cur_t == -1:
                                    image.save(progress_path)
                                    display.clear_output(wait=True)
                                    display.display(display.Image(progress_path))

                                if args.intermediates_in_subfolder:
                                    _img_path = os.path.join(
                                        self._output.partial_dir, filename
                                    )
                                else:
                                    _img_path = os.path.join(batch_dir, filename)

                                if args.steps_per_checkpoint is not None:
                                    if j % args.steps_per_checkpoint == 0 and j > 0:
                                        image.save(_img_path)
                                else:
                                    if j in args.intermediate_saves:
                                        image.save(_img_path)
                                if cur_t == -1:
                                    if frame_num == 0:
                                        self.save_settings(args)
                                    if args.animation_mode != "None":
                                        image.save(prev_frame_path)
                                    image.save(os.path.join(batch_dir, filename))
                                    log.info(f"Saved {filename}")
                                    if args.animation_mode == "3D":
                                        # If turbo, save a blended image
                                        _img_path = os.path.join(batch_dir, filename)
                                        if args.turbo_mode and frame_num > 0:
                                            # Mix new image with prevFrameScaled
                                            blend_factor = (1) / int(args.turbo_steps)
                                            # This is already updated..
                                            newFrame = cv2.imread(prev_frame_path)
                                            prev_frame_warped = cv2.imread(
                                                prev_frame_scaled_path
                                            )
                                            blendedImage = cv2.addWeighted(
                                                newFrame,
                                                blend_factor,
                                                prev_frame_warped,
                                                (1 - blend_factor),
                                                0.0,
                                            )
                                            cv2.imwrite(_img_path, blendedImage)
                                        else:
                                            image.save(_img_path)

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

                                    # if frame_num != _diffuse.max_frames-1:
                                    #   display.clear_output()

                if args.animation_mode == "None":
                    batchBar.n = i + 1
                    batchBar.refresh()

                plt.plot(np.array(loss_values), "r")

    def load_config(self, **args):
        """Load the settings"""
        log.info(f"Merging config with args: {args}")
        args = eKonf.merge(self._diffuse, args)
        self._prepare_folders(args.batch_name)

        # Get corrected sizes
        args.side_x = (args.width_height[0] // 64) * 64
        args.side_y = (args.width_height[1] // 64) * 64
        if args.side_x != args.width_height[0] or args.side_y != args.width_height[1]:
            log.info(
                f"Changing output size to {args.side_x}x{args.side_y}. Dimensions must by multiples of 64."
            )

        if args.animation_mode == "Video Input":
            args.steps = args.video_init_steps

            log.info(f"Exporting Video Frames (1 every {args.extract_nth_frame})...")
            try:
                for f in Path(f"{self._output.video_frames_dir}").glob("*.jpg"):
                    f.unlink()
            except:
                log.info(f"No video frames found in {self._output.video_frames_dir}")
            vf = f"select=not(mod(n\,{args.extract_nth_frame}))"
            if os.path.exists(args.video_init_path):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        f"{args.video_init_path}",
                        "-vf",
                        f"{vf}",
                        "-vsync",
                        "vfr",
                        "-q:v",
                        "2",
                        "-loglevel",
                        "error",
                        "-stats",
                        f"{self._output.video_frames_dir}/%04d.jpg",
                    ],
                    stdout=subprocess.PIPE,
                ).stdout.decode("utf-8")
            else:
                log.warning(
                    f"WARNING!\n\nVideo not found: {args.video_init_path}.\nPlease check your video path."
                )
            # !ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {video.video_frames_dir}/%04d.jpg

        if args.animation_mode == "Video Input":
            args.max_frames = len(glob(f"{self._output.video_frames_dir}/*.jpg"))

        # insist turbo be used only w 3d anim.
        if args.turbo_mode and args.animation_mode != "3D":
            log.info("Turbo mode only available with 3D animations. Disabling Turbo.")
            args.turbo_mode = False

        # insist VR be used only w 3d anim.
        if args.vr_mode and args.animation_mode != "3D":
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
                if args.intermediate_saves is not None:
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
                    args.steps_per_checkpoint = args.steps + 10
            log.info(f"Will save every {args.steps_per_checkpoint} steps")

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

        if args.animation_mode == "Video Input":
            frames = sorted(glob(self._output.in_dir + "/*.*"))
            if len(frames) == 0:
                raise Exception(
                    "ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell."
                )
            flows = glob(self._output.flo_dir + "/*.*")
            if (len(flows) == 0) and args.video_init_flow_warp:
                raise Exception(
                    "ERROR: 0 flow files found.\nPlease rerun the flow generation cell."
                )

        if args.steps <= args.calc_frames_skip_steps:
            raise Exception("ERROR: You can't skip more steps than your total steps")

        if args.resume_run:
            if args.run_to_resume == "latest":
                try:
                    args.batch_num
                except:
                    args.batch_num = (
                        len(
                            glob(
                                f"{self._output.batch_dir}/{args.batch_name}(*)_settings.yaml"
                            )
                        )
                        - 1
                    )
            else:
                args.batch_num = int(args.run_to_resum)

            if args.resume_from_frame == "latest":
                start_frame = len(
                    glob(
                        self._output.batch_dir
                        + f"/{args.batch_name}({args.batch_num})_*.png"
                    )
                )
                if (
                    args.animation_mode != "3D"
                    and args.turbo_mode == True
                    and start_frame > args.turbo_preroll
                    and start_frame % int(args.turbo_steps) != 0
                ):
                    start_frame = start_frame - (start_frame % int(args.turbo_steps))
            else:
                start_frame = int(args.resume_from_frame) + 1
                if (
                    args.animation_mode != "3D"
                    and args.turbo_mode == True
                    and start_frame > args.turbo_preroll
                    and start_frame % int(args.turbo_steps) != 0
                ):
                    start_frame = start_frame - (start_frame % int(args.turbo_steps))
                if args.retain_overwritten_frames:
                    existing_frames = len(
                        glob(
                            self._output.batch_dir
                            + f"/{args.batch_name}({args.batch_num})_*.png"
                        )
                    )
                    frames_to_save = existing_frames - start_frame
                    print(f"Moving {frames_to_save} frames to the Retained folder")
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
            args.batch_num = len(glob(self._output.batch_dir + "/*.txt"))
            while os.path.isfile(
                f"{self._output.batch_dir}/{args.batch_name}({args.batch_num})_settings.yaml"
            ) or os.path.isfile(
                f"{self._output.batch_dir}/{args.batch_name}-{args.batch_num}_settings.yaml"
            ):
                args.batch_num += 1
        args.start_frame = start_frame

        if args.set_seed == "random_seed":
            random.seed()
            args.seed = random.randint(0, 2 ** 32)
        else:
            args.seed = int(args.set_seed)
        log.info(f"Using seed: {args.seed}")

        args.n_samples = args.n_samples if args.animation_mode == "None" else 1
        args.max_frames = args.max_frames if args.animation_mode != "None" else 1

        if args.animation_mode == "Video Input":
            # This isn't great in terms of what will get saved to the settings.. but it should work.
            args.steps = args.video_init_steps
            args.clip_guidance_scale = args.video_init_clip_guidance_scale
            args.tv_scale = args.video_init_tv_scale
            args.range_scale = args.video_init_range_scale
            args.sat_scale = args.video_init_sat_scale
            args.cutn_batches = args.video_init_cutn_batches
            args.skip_steps = args.video_init_skip_steps
            args.frames_scale = args.video_init_frames_scale
            args.frames_skip_steps = args.video_init_frames_skip_steps

        return args

    def collage(
        self,
        batch_name=None,
        batch_num=0,
        ncols=7,
        num_images=None,
        filename_patterns=None,
        **kwargs,
    ):
        args = self.load_config(**kwargs)
        batch_name = batch_name or args.batch_name
        batch_num = batch_num or args.batch_num
        self._prepare_folders(batch_name)

        filename_patterns = filename_patterns or f"{batch_name}({batch_num})_*.png"
        num_images = num_images or args.n_samples
        eKonf.collage(
            filename_patterns=filename_patterns,
            base_dir=self._output.batch_dir,
            num_images=num_images,
            ncols=ncols,
        )

    def make_gif(
        self,
        batch_name=None,
        batch_num=0,
        sample_num=0,
        show=False,
        force_remake=False,
        output_file=None,
        filename_patterns=None,
        duration=100,
        loop=0,
        width=None,
        optimize=True,
        quality=50,
        **kwargs,
    ):
        from PIL import Image
        from IPython.display import Image as Img
        from IPython.display import display

        args = self.load_config(**kwargs)
        batch_name = batch_name or args.batch_name
        batch_num = batch_num or args.batch_num
        self._prepare_folders(batch_name)
        base_dir = self._output.partial_dir

        filename_patterns = (
            filename_patterns or f"{batch_name}({batch_num})_{sample_num:04}-*.png"
        )
        log.info(f"Making GIF from {filename_patterns}")
        output_file = output_file or f"{batch_name}({batch_num})_{sample_num:04}.gif"
        output_path = os.path.join(self._output.batch_dir, output_file)
        if os.path.exists(output_path) and not force_remake:
            log.info(f"Skipping GIF creation, already exists: {output_path}")
        else:
            frames = [
                Image.open(image) for image in glob(f"{base_dir}/{filename_patterns}")
            ]
            if len(frames) > 0:
                frame_one = frames[0]
                frame_one.save(
                    output_path,
                    format="GIF",
                    append_images=frames,
                    save_all=True,
                    duration=duration,
                    loop=loop,
                    optimize=optimize,
                    quality=quality,
                )
                log.info(f"Saved GIF to {output_path}")
            else:
                log.warning(f"No frames found for {filename_patterns}")

        if show and os.path.exists(output_path):
            display(Img(data=open(output_path, "rb").read(), width=width))

    def _prepare_folders(self, batch_name):
        self._output.batch_dir = os.path.join(self._path.output_dir, batch_name)
        self._output.retain_dir = os.path.join(self._output.batch_dir, "retained")
        self._output.partial_dir = os.path.join(self._output.batch_dir, "partials")
        self._output.video_frames_dir = os.path.join(
            self._output.batch_dir, "video_frames"
        )
        self._output.in_dir = self._output.video_frames_dir
        self._output.flo_dir = os.path.join(
            self._output.video_frames_dir, "out_flo_fwd"
        )
        self._output.temp_flo_dir = os.path.join(
            self._output.video_frames_dir, "temp_flo"
        )
        self._output.flo_fwd_dir = os.path.join(
            self._output.video_frames_dir, "out_flo_fwd"
        )
        for _name, _path in self._output.items():
            if not os.path.exists(_path):
                os.makedirs(_path)
