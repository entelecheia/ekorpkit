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

# from tqdm.auto import tqdm

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
        self._module = self.args.module
        self._midas = self.args.midas

        self.clip_models = []
        self.batch_num = None
        self._basic = self.args.basic
        self._diffuse = self.args.diffuse
        self._video = self.args.video
        self._anim = self.args.animnation
        self._extra = self.args.extra
        self._model = self.args.model
        self.text_prompts = self.args.text_prompts
        self.image_prompts = self.args.image_prompts
        self.model = None
        self.diffusion = None
        # Make sure GPU memory doesn't get corrupted from cancelling the run mid-way through, allow a full frame to complete
        self.stop_on_next_loop = False

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
        log.info("> loading settings...")
        self.load_settings()
        log.info("> loading optical flow...")
        self.load_optical_flow()
        log.info("> loading diffusion models...")
        self.load_diffusion_models()
        log.info("> loading clip models...")
        self.load_clip_models()

    def diffuse(self):
        """Diffuse the model"""

        self._diffuse = self.args.diffuse
        steps = self._basic.steps
        if self.animation_mode == "Video Input":
            steps = self._video.video_init_steps
            self._basic.steps = steps
        timestep_respacing = f"ddim{steps}"
        diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
        self.model_config.update(
            {
                "timestep_respacing": timestep_respacing,
                "diffusion_steps": diffusion_steps,
            }
        )
        skip_step_ratio = int(self._basic.frames_skip_steps.rstrip("%")) / 100
        self._diffuse.skip_step_ratio = skip_step_ratio
        self._diffuse.calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

        if self.animation_mode == "Video Input":
            frames = sorted(glob(self._video.in_dir + "/*.*"))
            if len(frames) == 0:
                raise Exception(
                    "ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell."
                )
            flows = glob(self._video.flo_dir + "/*.*")
            if (len(flows) == 0) and self._video.video_init_flow_warp:
                raise Exception(
                    "ERROR: 0 flow files found.\nPlease rerun the flow generation cell."
                )

        if self._basic.steps <= self._diffuse.calc_frames_skip_steps:
            raise Exception("ERROR: You can't skip more steps than your total steps")

        if self._diffuse.resume_run:
            if self._diffuse.run_to_resume == "latest":
                try:
                    self.batch_num
                except:
                    self.batch_num = (
                        len(
                            glob(
                                f"{self._basic.batch_dir}/{self._basic.batch_name}(*)_settings.txt"
                            )
                        )
                        - 1
                    )
            else:
                self.batch_num = int(self._diffuse.run_to_resum)

            resume_from_frame = self._diffuse.resume_from_frame
            retain_overwritten_frames = self._diffuse.retain_overwritten_frames

            if self._diffuse.resume_from_frame == "latest":
                start_frame = len(
                    glob(
                        self._basic.batch_dir
                        + f"/{self._basic.batch_name}({self.batch_num})_*.png"
                    )
                )
                if (
                    self.animation_mode != "3D"
                    and self._anim.turbo_mode == True
                    and start_frame > self._anim.turbo_preroll
                    and start_frame % int(self._anim.turbo_steps) != 0
                ):
                    start_frame = start_frame - (
                        start_frame % int(self._anim.turbo_steps)
                    )
            else:
                start_frame = int(resume_from_frame) + 1
                if (
                    self.animation_mode != "3D"
                    and self._anim.turbo_mode == True
                    and start_frame > self._anim.turbo_preroll
                    and start_frame % int(self._anim.turbo_steps) != 0
                ):
                    start_frame = start_frame - (
                        start_frame % int(self._anim.turbo_steps)
                    )
                if retain_overwritten_frames is True:
                    existing_frames = len(
                        glob(
                            self._basic.batch_dir
                            + f"/{self._basic.batch_name}({self.batch_num})_*.png"
                        )
                    )
                    frames_to_save = existing_frames - start_frame
                    print(f"Moving {frames_to_save} frames to the Retained folder")
                    move_files(
                        start_frame,
                        existing_frames,
                        self._basic.batch_dir,
                        self._basic.retain_dir,
                        self._basic.batch_name,
                        self.batch_num,
                    )
        else:
            start_frame = 0
            self.batch_num = len(glob(self._basic.batch_dir + "/*.txt"))
            while os.path.isfile(
                f"{self._basic.batch_dir}/{self._basic.batch_name}({self.batch_num})_settings.yaml"
            ) or os.path.isfile(
                f"{self._basic.batch_dir}/{self._basic.batch_name}-{self.batch_num}_settings.yaml"
            ):
                self.batch_num += 1

        log.info(
            f"Starting Run: {self._basic.batch_name}({self.batch_num}) at frame {start_frame}"
        )
        self._diffuse.batch_num = self.batch_num
        self._diffuse.start_frame = start_frame

        if self._extra.set_seed == "random_seed":
            random.seed()
            self.seed = random.randint(0, 2 ** 32)
        else:
            self.seed = int(self._extra.set_seed)
        self._diffuse.seed = self.seed
        log.info(f"Using seed: {self.seed}")

        self.prompts_series = (
            split_prompts(self.text_prompts, self._anim.max_frames)
            if self.text_prompts
            else None
        )
        self.image_prompts_series = (
            split_prompts(self.image_prompts, self._anim.max_frames)
            if self.image_prompts
            else None
        )

        self._diffuse.n_batches = (
            self._diffuse.n_batches if self.animation_mode == "None" else 1
        )
        self._diffuse.max_frames = (
            self._anim.max_frames if self.animation_mode != "None" else 1
        )
        if isinstance(self._extra.cut_overview, str):
            self._extra.cut_overview = eval(self._extra.cut_overview)
            self._extra.cut_innercut = eval(self._extra.cut_innercut)
            self._extra.cut_icgray_p = eval(self._extra.cut_icgray_p)

        if self.animation_mode == "Video Input":
            # This isn't great in terms of what will get saved to the settings.. but it should work.
            self._diffuse.steps = self._video.video_init_steps
            self._diffuse.clip_guidance_scale = (
                self._video.video_init_clip_guidance_scale
            )
            self._diffuse.tv_scale = self._video.video_init_tv_scale
            self._diffuse.range_scale = self._video.video_init_range_scale
            self._diffuse.sat_scale = self._video.video_init_sat_scale
            self._diffuse.cutn_batches = self._video.video_init_cutn_batches
            self._diffuse.skip_steps = self._video.video_init_skip_steps
            self._diffuse.frames_scale = self._video.video_init_frames_scale
            self._diffuse.frames_skip_steps = self._video.video_init_frames_skip_steps

        self._prepare_models()

        gc.collect()
        torch.cuda.empty_cache()
        try:
            self._run()
        except KeyboardInterrupt:
            pass
        finally:
            log.info(f"Seed used: {self.seed}")
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

    def load_optical_flow(self):
        """Load the optical flow"""
        import argparse
        from raft import RAFT
        from .flow import load_img, get_flow

        if self.animation_mode == "Video Input":
            rsft_args = argparse.Namespace()
            rsft_args.small = False
            rsft_args.mixed_precision = True

            if not self._video.video_init_flow_warp:
                log.info("video_init_flow_warp not set, skipping")

            else:
                flows = glob(self._video.flo_dir + "/*.*")
                if (len(flows) > 0) and not self._video.force_flow_generation:
                    log.info(
                        f"Skipping flow generation:\nFound {len(flows)} existing flow files in current working folder: {self._video.flo_dir}.\nIf you wish to generate new flow files, check force_flow_generation and run this cell again."
                    )

                if (len(flows) == 0) or self._video.force_flow_generation:
                    frames = sorted(glob(self._video.in_dir + "/*.*"))
                    if len(frames) < 2:
                        log.warning(
                            f"WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path."
                        )
                    if len(frames) >= 2:

                        raft_model = torch.nn.DataParallel(RAFT(rsft_args))
                        raft_model.load_state_dict(
                            torch.load(self._video.raft_model_path)
                        )
                        raft_model = raft_model.module.cuda().eval()

                        for f in Path(f"{self._video.flo_fwd_dir}").glob("*.*"):
                            f.unlink()

                        # TBD Call out to a consistency checker?
                        self.framecount = 0
                        for frame1, frame2 in tqdm(
                            zip(frames[:-1], frames[1:]), total=len(frames) - 1
                        ):

                            out_flow21_fn = (
                                f"{self._video.flo_fwd_dir}/{frame1.split('/')[-1]}"
                            )

                            frame1 = load_img(frame1, self._basic.width_height)
                            frame2 = load_img(frame2, self._basic.width_height)

                            flow21 = get_flow(frame2, frame1, raft_model)
                            np.save(out_flow21_fn, flow21)

                            if self._video.video_init_check_consistency:
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

        _clip = self.args.clip
        clip_models = []
        if _clip.ViTB32:
            clip_models.append(
                clip.load("ViT-B/32", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.ViTB16:
            clip_models.append(
                clip.load("ViT-B/16", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.ViTL14:
            clip_models.append(
                clip.load("ViT-L/14", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.ViTL14_336px:
            clip_models.append(
                clip.load("ViT-L/14@336px", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.RN50:
            clip_models.append(
                clip.load("RN50", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.RN50x4:
            clip_models.append(
                clip.load("RN50x4", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.RN50x16:
            clip_models.append(
                clip.load("RN50x16", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.RN50x64:
            clip_models.append(
                clip.load("RN50x64", jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.cuda_device)
            )
        if _clip.RN101:
            clip_models.append(
                clip.load("RN101", jit=False)[0]
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

    def load_settings(self):
        """Load the settings"""
        basic = self.args.basic
        # Get corrected sizes
        basic.side_x = (basic.width_height[0] // 64) * 64
        basic.side_y = (basic.width_height[1] // 64) * 64
        if (
            basic.side_x != basic.width_height[0]
            or basic.side_y != basic.width_height[1]
        ):
            log.info(
                f"Changing output size to {basic.side_x}x{basic.side_y}. Dimensions must by multiples of 64."
            )
        self._basic = basic
        self.animation_mode = basic.animation_mode

        video = self.args.video
        if basic.animation_mode == "Video Input":
            basic.steps = video.video_init_steps

            log.info(f"Exporting Video Frames (1 every {video.extract_nth_frame})...")
            try:
                for f in Path(f"{video.video_frames_dir}").glob("*.jpg"):
                    f.unlink()
            except:
                log.info(f"No video frames found in {video.video_frames_dir}")
            vf = f"select=not(mod(n\,{video.extract_nth_frame}))"
            if os.path.exists(video.video_init_path):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        f"{video.video_init_path}",
                        "-vf",
                        f"{vf}",
                        "-vsync",
                        "vfr",
                        "-q:v",
                        "2",
                        "-loglevel",
                        "error",
                        "-stats",
                        f"{video.video_frames_dir}/%04d.jpg",
                    ],
                    stdout=subprocess.PIPE,
                ).stdout.decode("utf-8")
            else:
                log.warning(
                    f"WARNING!\n\nVideo not found: {video.video_init_path}.\nPlease check your video path."
                )
            # !ffmpeg -i {video_init_path} -vf {vf} -vsync vfr -q:v 2 -loglevel error -stats {video.video_frames_dir}/%04d.jpg
        self._video = video

        anim = self.args.animnation
        if self.animation_mode == "Video Input":
            anim.max_frames = len(glob(f"{video.video_frames_dir}/*.jpg"))

        # insist turbo be used only w 3d anim.
        if anim.turbo_mode and basic.animation_mode != "3D":
            log.info("Turbo mode only available with 3D animations. Disabling Turbo.")
            anim.turbo_mode = False

        # insist VR be used only w 3d anim.
        if anim.vr_mode and self.animation_mode != "3D":
            log.info("VR mode only available with 3D animations. Disabling VR.")
            anim.vr_mode = False

        if anim.key_frames:
            try:
                self.angle_series = get_inbetweens(
                    parse_key_frames(anim.angle), anim.max_frames, anim.interp_spline
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `angle` correctly for key frames.\n"
                    "Attempting to interpret `angle` as "
                    f'"0: ({anim.angle})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.angle = f"0: ({anim.angle})"
                self.angle_series = get_inbetweens(
                    parse_key_frames(anim.angle), anim.max_frames, anim.interp_spline
                )

            try:
                self.zoom_series = get_inbetweens(
                    parse_key_frames(anim.zoom), anim.max_frames, anim.interp_spline
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `zoom` correctly for key frames.\n"
                    "Attempting to interpret `zoom` as "
                    f'"0: ({anim.zoom})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.zoom = f"0: ({anim.zoom})"
                self.zoom_series = get_inbetweens(
                    parse_key_frames(anim.zoom), anim.max_frames, anim.interp_spline
                )

            try:
                self.translation_x_series = get_inbetweens(
                    parse_key_frames(anim.translation_x),
                    anim.max_frames,
                    anim.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_x` correctly for key frames.\n"
                    "Attempting to interpret `translation_x` as "
                    f'"0: ({anim.translation_x})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.translation_x = f"0: ({anim.translation_x})"
                self.translation_x_series = get_inbetweens(
                    parse_key_frames(anim.translation_x),
                    anim.max_frames,
                    anim.interp_spline,
                )

            try:
                self.translation_y_series = get_inbetweens(
                    parse_key_frames(anim.translation_y),
                    anim.max_frames,
                    anim.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_y` correctly for key frames.\n"
                    "Attempting to interpret `translation_y` as "
                    f'"0: ({anim.translation_y})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.translation_y = f"0: ({anim.translation_y})"
                self.translation_y_series = get_inbetweens(
                    parse_key_frames(anim.translation_y),
                    anim.max_frames,
                    anim.interp_spline,
                )

            try:
                self.translation_z_series = get_inbetweens(
                    parse_key_frames(anim.translation_z),
                    anim.max_frames,
                    anim.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `translation_z` correctly for key frames.\n"
                    "Attempting to interpret `translation_z` as "
                    f'"0: ({anim.translation_z})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.translation_z = f"0: ({anim.translation_z})"
                self.translation_z_series = get_inbetweens(
                    parse_key_frames(anim.translation_z),
                    anim.max_frames,
                    anim.interp_spline,
                )

            try:
                self.rotation_3d_x_series = get_inbetweens(
                    parse_key_frames(anim.rotation_3d_x),
                    anim.max_frames,
                    anim.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_x` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_x` as "
                    f'"0: ({anim.rotation_3d_x})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.rotation_3d_x = f"0: ({anim.rotation_3d_x})"
                self.rotation_3d_x_series = get_inbetweens(
                    parse_key_frames(anim.rotation_3d_x),
                    anim.max_frames,
                    anim.interp_spline,
                )

            try:
                self.rotation_3d_y_series = get_inbetweens(
                    parse_key_frames(anim.rotation_3d_y),
                    anim.max_frames,
                    anim.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_y` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_y` as "
                    f'"0: ({anim.rotation_3d_y})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.rotation_3d_y = f"0: ({anim.rotation_3d_y})"
                self.rotation_3d_y_series = get_inbetweens(
                    parse_key_frames(anim.rotation_3d_y),
                    anim.max_frames,
                    anim.interp_spline,
                )

            try:
                self.rotation_3d_z_series = get_inbetweens(
                    parse_key_frames(anim.rotation_3d_z),
                    anim.max_frames,
                    anim.interp_spline,
                )
            except RuntimeError as e:
                log.warning(
                    "WARNING: You have selected to use key frames, but you have not "
                    "formatted `rotation_3d_z` correctly for key frames.\n"
                    "Attempting to interpret `rotation_3d_z` as "
                    f'"0: ({anim.rotation_3d_z})"\n'
                    "Please read the instructions to find out how to use key frames "
                    "correctly.\n"
                )
                anim.rotation_3d_z = f"0: ({anim.rotation_3d_z})"
                self.rotation_3d_z_series = get_inbetweens(
                    parse_key_frames(anim.rotation_3d_z),
                    anim.max_frames,
                    anim.interp_spline,
                )

        else:
            anim.angle = float(anim.angle)
            anim.zoom = float(anim.zoom)
            anim.translation_x = float(anim.translation_x)
            anim.translation_y = float(anim.translation_y)
            anim.translation_z = float(anim.translation_z)
            anim.rotation_3d_x = float(anim.rotation_3d_x)
            anim.rotation_3d_y = float(anim.rotation_3d_y)
            anim.rotation_3d_z = float(anim.rotation_3d_z)

        self.animnation = anim

        extra = self.args.extra
        if not eKonf.is_list(extra.intermediate_saves):
            if extra.intermediate_saves:
                extra.steps_per_checkpoint = math.floor(
                    (basic.steps - basic.skip_steps - 1)
                    // (extra.intermediate_saves + 1)
                )
                extra.steps_per_checkpoint = (
                    extra.steps_per_checkpoint if extra.steps_per_checkpoint > 0 else 1
                )
                log.info(f"Will save every {extra.steps_per_checkpoint} steps")
            else:
                extra.steps_per_checkpoint = basic.steps + 10
        else:
            extra.steps_per_checkpoint = None
        self._extra = extra

    def download_models(self):
        """Download the models"""
        download = self.args.download
        check_model_SHA = download.check_model_SHA
        for name, model in download.models.items():
            if not isinstance(model, str):
                log.info(f"Downloading model {name} from {model}")
                _download_models(name, **model, check_model_SHA=check_model_SHA)

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

    def _run(self):
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

        _diffuse = self._diffuse
        _anim = self._anim
        _basic = self._basic
        _video = self._video
        _extra = self._extra
        seed = self.seed

        if (self.animation_mode == "3D") and (_anim.midas_weight > 0.0):
            midas = Midas(**self._midas)
            (
                midas_model,
                midas_transform,
                midas_net_w,
                midas_net_h,
                midas_resize_mode,
                midas_normalization,
            ) = midas.init_midas_depth_model(_anim.midas_depth_model)

        log.info(f"looping over range({_diffuse.start_frame}, {_diffuse.max_frames})")
        for frame_num in range(_diffuse.start_frame, _diffuse.max_frames):
            if self.stop_on_next_loop:
                break

            if self.is_notebook:
                display.clear_output(wait=True)

            # Print Frame progress if animation mode is on
            if self.animation_mode != "None":
                batchBar = tqdm(range(_diffuse.max_frames), desc="Frames")
                batchBar.n = frame_num
                batchBar.refresh()

            # Inits if not video frames
            if self.animation_mode != "Video Input":
                if _basic.init_image in ["", "none", "None", "NONE"]:
                    init_image = None
                else:
                    init_image = _basic.init_image
                init_scale = _basic.init_scale
                skip_steps = _basic.skip_steps

            if self.animation_mode == "2D":
                if _anim.key_frames:
                    angle = self.angle_series[frame_num]
                    zoom = self.zoom_series[frame_num]
                    translation_x = self.translation_x_series[frame_num]
                    translation_y = self.translation_y_series[frame_num]
                    log.info(
                        f"angle: {angle}, zoom: {zoom}, translation_x: {translation_x}, translation_y: {translation_y}"
                    )

                if frame_num > 0:
                    seed += 1
                    if _diffuse.resume_run and frame_num == _diffuse.start_frame:
                        img_0 = cv2.imread(
                            _basic.batch_dir
                            + f"/{_diffuse.batch_name}({_diffuse.batch_num})_{_diffuse.start_frame-1:04}.png"
                        )
                    else:
                        img_0 = cv2.imread("prevFrame.png")
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

                    cv2.imwrite("prevFrameScaled.png", img_0)
                    init_image = "prevFrameScaled.png"
                    init_scale = _diffuse.frames_scale
                    skip_steps = _diffuse.calc_frames_skip_steps

            if self.animation_mode == "3D":
                if frame_num > 0:
                    seed += 1
                    if _diffuse.resume_run and frame_num == _diffuse.start_frame:
                        img_filepath = (
                            _basic.batch_dir
                            + f"/{_diffuse.batch_name}({_diffuse.batch_num})_{_diffuse.start_frame-1:04}.png"
                        )
                        if _diffuse.turbo_mode and frame_num > _diffuse.turbo_preroll:
                            shutil.copyfile(img_filepath, "oldFrameScaled.png")
                    else:
                        img_filepath = "prevFrame.png"

                    next_step_pil = self._3d_step(
                        img_filepath, frame_num, midas_model, midas_transform
                    )
                    next_step_pil.save("prevFrameScaled.png")

                    ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                    if _anim.turbo_mode:
                        if frame_num == _anim.turbo_preroll:  # start tracking oldframe
                            next_step_pil.save(
                                "oldFrameScaled.png"
                            )  # stash for later blending
                        elif frame_num > _anim.turbo_preroll:
                            # set up 2 warped image sequences, old & new, to blend toward new diff image
                            old_frame = self._3d_step(
                                "oldFrameScaled.png",
                                frame_num,
                                midas_model,
                                midas_transform,
                            )
                            old_frame.save("oldFrameScaled.png")
                            if frame_num % int(_anim.turbo_steps) != 0:
                                log.info(
                                    "turbo skip this frame: skipping clip diffusion steps"
                                )
                                filename = f"{_diffuse.batch_name}({_diffuse.batch_num})_{frame_num:04}.png"
                                blend_factor = (
                                    (frame_num % int(_anim.turbo_steps)) + 1
                                ) / int(_anim.turbo_steps)
                                log.info(
                                    "turbo skip this frame: skipping clip diffusion steps and saving blended frame"
                                )
                                newWarpedImg = cv2.imread(
                                    "prevFrameScaled.png"
                                )  # this is already updated..
                                oldWarpedImg = cv2.imread("oldFrameScaled.png")
                                blendedImage = cv2.addWeighted(
                                    newWarpedImg,
                                    blend_factor,
                                    oldWarpedImg,
                                    1 - blend_factor,
                                    0.0,
                                )
                                cv2.imwrite(
                                    f"{_basic.batch_dir}/{filename}", blendedImage
                                )
                                next_step_pil.save(
                                    f"{img_filepath}"
                                )  # save it also as prev_frame to feed next iteration
                                if _diffuse.vr_mode:
                                    generate_eye_views(
                                        self.TRANSLATION_SCALE,
                                        _basic.batch_dir,
                                        filename,
                                        frame_num,
                                        midas_model,
                                        midas_transform,
                                        self.cuda_device,
                                        _anim.vr_eye_angle,
                                        _anim.vr_ipd,
                                        _anim.near_plane,
                                        _anim.far_plane,
                                        _anim.fov,
                                        _anim.padding_mode,
                                        _anim.sampling_mode,
                                        _anim.midas_weight,
                                    )
                                continue
                            else:
                                # if not a skip frame, will run diffusion and need to blend.
                                oldWarpedImg = cv2.imread("prevFrameScaled.png")
                                cv2.imwrite(
                                    f"oldFrameScaled.png", oldWarpedImg
                                )  # swap in for blending later
                                log.info(
                                    "clip/diff this frame - generate clip diff image"
                                )

                    init_image = "prevFrameScaled.png"
                    init_scale = _diffuse.frames_scale
                    skip_steps = _diffuse.calc_frames_skip_steps

            if self.animation_mode == "Video Input":
                init_scale = _video.video_init_frames_scale
                skip_steps = _video.calc_frames_skip_steps
                if not _video.video_init_seed_continuity:
                    seed += 1
                if _video.video_init_flow_warp:
                    if frame_num == 0:
                        skip_steps = _video.video_init_skip_steps
                        init_image = f"{_video.video_frames_dir}/{frame_num+1:04}.jpg"
                    if frame_num > 0:
                        prev = PIL.Image.open(
                            _basic.batch_dir
                            + f"/{_diffuse.batch_name}({_diffuse.batch_num})_{frame_num-1:04}.png"
                        )

                        frame1_path = f"{_video.video_frames_dir}/{frame_num:04}.jpg"
                        frame2 = PIL.Image.open(
                            f"{_video.video_frames_dir}/{frame_num+1:04}.jpg"
                        )
                        flo_path = f"/{_diffuse.self._video.flo_dir}/{frame1_path.split('/')[-1]}.npy"

                        init_image = "warped.png"
                        log(_video.video_init_flow_blend)
                        weights_path = None
                        if _video.video_init_check_consistency:
                            # TBD
                            pass

                        warp(
                            prev,
                            frame2,
                            flo_path,
                            blend=_diffuse.video_init_flow_blend,
                            weights_path=weights_path,
                        ).save(init_image)

                else:
                    init_image = f"{_video.video_frames_dir}/{frame_num+1:04}.jpg"

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

                    if _extra.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (
                                    txt
                                    + torch.randn(txt.shape).cuda() * _diffuse.rand_mag
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
                        skip_augs=_diffuse.skip_augs,
                    )
                    for prompt in image_prompt:
                        path, weight = parse_prompt(prompt)
                        img = Image.open(fetch(path)).convert("RGB")
                        img = TF.resize(
                            img,
                            min(_diffuse.side_x, _diffuse.side_y, *img.size),
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
                        if _diffuse.fuzzy_prompt:
                            for i in range(25):
                                model_stat["target_embeds"].append(
                                    (
                                        embed
                                        + torch.randn(embed.shape).cuda()
                                        * _diffuse.rand_mag
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
                init = init.resize((_diffuse.side_x, _diffuse.side_y), Image.LANCZOS)
                init = (
                    TF.to_tensor(init).to(self.cuda_device).unsqueeze(0).mul(2).sub(1)
                )

            if _extra.perlin_init:
                if _diffuse.perlin_mode == "color":
                    init = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
                    )
                    init2 = create_perlin_noise(
                        [1.5 ** -i * 0.5 for i in range(8)], 4, 4, False
                    )
                elif _diffuse.perlin_mode == "gray":
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

            device = self.cuda_device
            model = self.model
            diffusion = self.diffusion
            secondary_model = self.secondary_model

            def cond_fn(x, t, y=None):
                with torch.enable_grad():
                    x_is_NaN = False
                    x = x.detach().requires_grad_()
                    n = x.shape[0]
                    if self._model.use_secondary_model is True:
                        alpha = torch.tensor(
                            diffusion.sqrt_alphas_cumprod[cur_t],
                            device=device,
                            dtype=torch.float32,
                        )
                        sigma = torch.tensor(
                            diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                            device=device,
                            dtype=torch.float32,
                        )
                        cosine_t = alpha_sigma_to_t(alpha, sigma)
                        out = secondary_model(x, cosine_t[None].repeat([n])).pred
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)
                    else:
                        my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                        out = diffusion.p_mean_variance(
                            model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                        )
                        fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                        x_in = out["pred_xstart"] * fac + x * (1 - fac)
                        x_in_grad = torch.zeros_like(x_in)
                    for model_stat in model_stats:
                        for i in range(_diffuse.cutn_batches):
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
                                self.animation_mode,
                                input_resolution,
                                _basic.skip_augs,
                                Overview=_extra.cut_overview[1000 - t_int],
                                InnerCrop=_extra.cut_innercut[1000 - t_int],
                                IC_Size_Pow=_extra.cut_ic_pow,
                                IC_Grey_P=_extra.cut_icgray_p[1000 - t_int],
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
                                    _extra.cut_overview[1000 - t_int]
                                    + _extra.cut_innercut[1000 - t_int],
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
                                    losses.sum() * _diffuse.clip_guidance_scale, x_in
                                )[0]
                                / _diffuse.cutn_batches
                            )
                    tv_losses = tv_loss(x_in)
                    if self._model.use_secondary_model is True:
                        range_losses = range_loss(out)
                    else:
                        range_losses = range_loss(out["pred_xstart"])
                    sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                    loss = (
                        tv_losses.sum() * _diffuse.tv_scale
                        + range_losses.sum() * _diffuse.range_scale
                        + sat_losses.sum() * _diffuse.sat_scale
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
                if _extra.clamp_grad and x_is_NaN == False:
                    magnitude = grad.square().mean().sqrt()
                    return (
                        grad * magnitude.clamp(max=_extra.clamp_max) / magnitude
                    )  # min=-0.02, min=-clamp_max,
                return grad

            if self._model.diffusion_sampling_mode == "ddim":
                sample_fn = diffusion.ddim_sample_loop_progressive
            else:
                sample_fn = diffusion.plms_sample_loop_progressive

            image_display = Output()
            for i in range(_diffuse.n_batches):
                if _diffuse.animation_mode == "None":
                    display.clear_output(wait=True)
                    batchBar = tqdm(range(_diffuse.n_batches), desc="Batches")
                    batchBar.n = i
                    batchBar.refresh()
                print("")
                display.display(image_display)
                gc.collect()
                torch.cuda.empty_cache()
                cur_t = diffusion.num_timesteps - skip_steps - 1
                total_steps = cur_t

                if _extra.perlin_init:
                    init = regen_perlin(
                        _extra.perlin_mode,
                        _diffuse.batch_size,
                        _basic.side_x,
                        _basic.side_y,
                        self.cuda_device,
                    )

                if self._model.diffusion_sampling_mode == "ddim":
                    samples = sample_fn(
                        model,
                        (_diffuse.batch_size, 3, _basic.side_y, _basic.side_x),
                        clip_denoised=_extra.clip_denoised,
                        model_kwargs={},
                        cond_fn=cond_fn,
                        progress=True,
                        skip_timesteps=skip_steps,
                        init_image=init,
                        randomize_class=_extra.randomize_class,
                        eta=_extra.eta,
                        # transformation_fn=symmetry_transformation_fn,
                        # transformation_percent=_extra.transformation_percent,
                    )
                else:
                    samples = sample_fn(
                        model,
                        (_diffuse.batch_size, 3, _basic.side_y, _basic.side_x),
                        clip_denoised=_extra.clip_denoised,
                        model_kwargs={},
                        cond_fn=cond_fn,
                        progress=True,
                        skip_timesteps=skip_steps,
                        init_image=init,
                        randomize_class=_extra.randomize_class,
                        order=2,
                    )

                # with run_display:
                # display.clear_output(wait=True)
                for j, sample in enumerate(samples):
                    cur_t -= 1
                    intermediate_step = False
                    if _extra.steps_per_checkpoint is not None:
                        if j % _extra.steps_per_checkpoint == 0 and j > 0:
                            intermediate_step = True
                    elif j in _extra.intermediate_saves:
                        intermediate_step = True
                    with image_display:
                        if (
                            j % _diffuse.display_rate == 0
                            or cur_t == -1
                            or intermediate_step == True
                        ):
                            for k, image in enumerate(sample["pred_xstart"]):
                                # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                                current_time = datetime.now().strftime(
                                    "%y%m%d-%H%M%S_%f"
                                )
                                percent = math.ceil(j / total_steps * 100)
                                if _diffuse.n_batches > 0:
                                    # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                    if (
                                        cur_t == -1
                                        and _extra.intermediates_in_subfolder is True
                                    ):
                                        save_num = (
                                            f"{frame_num:04}"
                                            if self.animation_mode != "None"
                                            else i
                                        )
                                        filename = f"{_basic.batch_name}({_diffuse.batch_num})_{save_num}.png"
                                    else:
                                        # If we're working with percentages, append it
                                        if _extra.steps_per_checkpoint is not None:
                                            filename = f"{_basic.batch_name}({_diffuse.batch_num})_{i:04}-{percent:02}%.png"
                                        # Or else, iIf we're working with specific steps, append those
                                        else:
                                            filename = f"{_basic.batch_name}({_diffuse.batch_num})_{i:04}-{j:03}.png"
                                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                                if j % _diffuse.display_rate == 0 or cur_t == -1:
                                    image.save("progress.png")
                                    display.clear_output(wait=True)
                                    display.display(display.Image("progress.png"))
                                if _extra.steps_per_checkpoint is not None:
                                    if j % _extra.steps_per_checkpoint == 0 and j > 0:
                                        if _extra.intermediates_in_subfolder is True:
                                            image.save(
                                                f"{_extra.partial_dir}/{filename}"
                                            )
                                        else:
                                            image.save(f"{_basic.batch_dir}/{filename}")
                                else:
                                    if j in _extra.intermediate_saves:
                                        if _extra.intermediates_in_subfolder is True:
                                            image.save(
                                                f"{_extra.partial_dir}/{filename}"
                                            )
                                        else:
                                            image.save(f"{_basic.batch_dir}/{filename}")
                                if cur_t == -1:
                                    # if frame_num == 0:
                                    #     save_settings()
                                    if self.animation_mode != "None":
                                        image.save("prevFrame.png")
                                    image.save(f"{_basic.batch_dir}/{filename}")
                                    log.info(f"Saved {filename}")
                                    if self.animation_mode == "3D":
                                        # If turbo, save a blended image
                                        if _anim.turbo_mode and frame_num > 0:
                                            # Mix new image with prevFrameScaled
                                            blend_factor = (1) / int(_anim.turbo_steps)
                                            newFrame = cv2.imread(
                                                "prevFrame.png"
                                            )  # This is already updated..
                                            prev_frame_warped = cv2.imread(
                                                "prevFrameScaled.png"
                                            )
                                            blendedImage = cv2.addWeighted(
                                                newFrame,
                                                blend_factor,
                                                prev_frame_warped,
                                                (1 - blend_factor),
                                                0.0,
                                            )
                                            cv2.imwrite(
                                                f"{_basic.batch_dir}/{filename}",
                                                blendedImage,
                                            )
                                        else:
                                            image.save(f"{_basic.batch_dir}/{filename}")

                                        if _anim.vr_mode:
                                            generate_eye_views(
                                                self.TRANSLATION_SCALE,
                                                _basic.batch_dir,
                                                filename,
                                                frame_num,
                                                midas_model,
                                                midas_transform,
                                                self.cuda_device,
                                                _anim.vr_eye_angle,
                                                _anim.vr_ipd,
                                                _anim.near_plane,
                                                _anim.far_plane,
                                                _anim.fov,
                                                _anim.padding_mode,
                                                _anim.sampling_mode,
                                                _anim.midas_weight,
                                            )

                                    # if frame_num != _diffuse.max_frames-1:
                                    #   display.clear_output()

                plt.plot(np.array(loss_values), "r")
