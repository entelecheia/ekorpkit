from utils.utils import InputPadder
import numpy as np
import PIL, cv2
from PIL import Image
import torch


TAG_CHAR = np.array([202021.25], np.float32)


def writeFlow(filename, uv, v=None):
    """
    https://github.com/NVIDIA/flownet2-pytorch/blob/master/utils/flow_utils.py
    Copyright 2017 NVIDIA CORPORATION

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    f = open(filename, "wb")
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def load_img(img, size):
    img = Image.open(img).convert("RGB").resize(size)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


def get_flow(frame1, frame2, model, iters=20):
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    _, flow12 = model(frame1, frame2, iters=iters, test_mode=True)
    flow12 = flow12[0].permute(1, 2, 0).detach().cpu().numpy()

    return flow12


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def makeEven(_x):
    return _x if (_x % 2 == 0) else _x + 1


def fit(img, maxsize=512):
    maxdim = max(*img.size)
    if maxdim > maxsize:
        # if True:
        ratio = maxsize / maxdim
        x, y = img.size
        size = (makeEven(int(x * ratio)), makeEven(int(y * ratio)))
        img = img.resize(size)
    return img


def warp(frame1, frame2, flo_path, blend=0.5, weights_path=None):
    flow21 = np.load(flo_path)
    frame1pil = np.array(
        frame1.convert("RGB").resize((flow21.shape[1], flow21.shape[0]))
    )
    frame1_warped21 = warp_flow(frame1pil, flow21)
    # frame2pil = frame1pil
    frame2pil = np.array(
        frame2.convert("RGB").resize((flow21.shape[1], flow21.shape[0]))
    )

    if weights_path:
        # TBD
        pass
    else:
        blended_w = frame2pil * (1 - blend) + frame1_warped21 * (blend)

    return PIL.Image.fromarray(blended_w.astype("uint8"))
