import torch
import logging
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class Midas:
    def __init__(self, **args):
        args = eKonf.to_config(args)

        self.cuda_device = torch.device(
            f"cuda:{args.cuda_device}"
            if (torch.cuda.is_available() and not args.use_cpu)
            else "cpu"
        )

        # Initialize MiDaS depth model.
        # It remains resident in VRAM and likely takes around 2GB VRAM.
        # You could instead initialize it for each frame (and free it after each frame) to save VRAM.. but initializing it is slow.
        self.default_models = args.default_models

    def init_midas_depth_model(self, midas_model_type="dpt_large", optimize=True):
        import torchvision.transforms as T
        import cv2
        from midas.dpt_depth import DPTDepthModel
        from midas.midas_net import MidasNet
        from midas.midas_net_custom import MidasNet_small
        from midas.transforms import Resize, NormalizeImage, PrepareForNet

        midas_model = None
        net_w = None
        net_h = None
        resize_mode = None
        normalization = None

        log.info(f"Initializing MiDaS '{midas_model_type}' depth model...")
        # load network
        midas_model_path = self.default_models[midas_model_type]

        if midas_model_type == "dpt_large":  # DPT-Large
            midas_model = DPTDepthModel(
                path=midas_model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model_type == "dpt_hybrid":  # DPT-Hybrid
            midas_model = DPTDepthModel(
                path=midas_model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model_type == "dpt_hybrid_nyu":  # DPT-Hybrid-NYU
            midas_model = DPTDepthModel(
                path=midas_model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif midas_model_type == "midas_v21":
            midas_model = MidasNet(midas_model_path, non_negative=True)
            net_w, net_h = 384, 384
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif midas_model_type == "midas_v21_small":
            midas_model = MidasNet_small(
                midas_model_path,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={"expand": True},
            )
            net_w, net_h = 256, 256
            resize_mode = "upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            print(f"midas_model_type '{midas_model_type}' not implemented")
            assert False

        midas_transform = T.Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        midas_model.eval()

        if optimize == True:
            if self.cuda_device == torch.device("cuda"):
                midas_model = midas_model.to(memory_format=torch.channels_last)
                midas_model = midas_model.half()

        midas_model.to(self.cuda_device)

        log.info(f"MiDaS '{midas_model_type}' depth model initialized.")
        return midas_model, midas_transform, net_w, net_h, resize_mode, normalization
