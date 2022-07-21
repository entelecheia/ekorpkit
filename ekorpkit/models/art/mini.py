import os
import logging
import random
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from glob import glob
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key, shard
from PIL import Image
from tqdm.auto import trange
from .base import BaseTTIModel


log = logging.getLogger(__name__)


class DalleMini(BaseTTIModel):
    def __init__(self, **args):
        super().__init__(**args)

        self._num_devices = self.args.num_devices
        self.model = None
        self.model_params = None
        self.processor = None
        self.vqgan = None
        self.vqgan_params = None
        self.clip = None
        self.clip_params = None
        self.clip_processor = None
        self.devices = None

        if self.auto.load:
            self.load()

    def imagine(self, text_prompts=None, **args):

        """Diffuse the model"""

        log.info("> loading settings...")
        args = self.load_config(**args)

        text_prompts = text_prompts or eKonf.to_dict(args.text_prompts)
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        args.text_prompts = text_prompts

        self.sample_imagepaths = []

        tokenized_prompts = self.processor(args.text_prompts)
        print(f"Prompts: {args.text_prompts}\n")

        _batch_dir = self._output.batch_dir
        # _num_devices = min(self._num_devices, args.n_samples)
        _num_devices = self._num_devices
        _devices = self.devices[:_num_devices]
        log.info(f"Using {_num_devices} devices")
        log.info(f"Devices: {_devices}")
        args.num_samples = max(args.n_samples // _num_devices, 1) * _num_devices
        self._config = args
        self.save_settings(args)

        # Keys are passed to the model on each device to generate unique inference per device.
        # create a random key
        log.info(f"Seed used: {args.seed}")
        key = jax.random.PRNGKey(args.seed)
        # Model parameters are replicated on each device for faster inference.
        _model_params = replicate(self.model_params, devices=_devices)
        _vqgan_params = replicate(self.vqgan_params, devices=_devices)
        _tokenized_prompts = replicate(tokenized_prompts, devices=_devices)

        # Model functions are compiled and parallelized to take advantage of multiple devices.
        # model inference
        @partial(
            jax.pmap,
            axis_name="batch",
            static_broadcasted_argnums=(3, 4, 5, 6),
            devices=_devices,
        )
        def p_generate(
            tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.model.generate(
                **tokenized_prompt,
                prng_key=key,
                params=params,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                condition_scale=condition_scale,
            )

        # decode image
        @partial(jax.pmap, axis_name="batch", devices=_devices)
        def p_decode(indices, params):
            return self.vqgan.decode_code(indices, params=params)

        with elapsed_timer(format_time=True) as elapsed:

            # generate images
            # sample_images = []
            sample_num = 0
            for i in trange(max(args.n_samples // _num_devices, 1)):
                eKonf.clear_output(wait=True)
                # get a new key
                key, subkey = jax.random.split(key)
                # generate images
                encoded_images = p_generate(
                    _tokenized_prompts,
                    shard_prng_key(subkey),
                    _model_params,
                    args.gen_top_k,
                    args.gen_top_p,
                    args.temperature,
                    args.cond_scale,
                )
                # remove BOS
                encoded_images = encoded_images.sequences[..., 1:]
                # decode images
                decoded_images = p_decode(encoded_images, _vqgan_params)
                decoded_images = decoded_images.clip(0.0, 1.0).reshape(
                    (-1, 256, 256, 3)
                )
                for decoded_img in decoded_images:
                    img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                    eKonf.display(img)
                    # sample_images.append(img)
                    filename = (
                        f"{args.batch_name}({args.batch_num})_{sample_num:04}.png"
                    )
                    _img_path = os.path.join(_batch_dir, filename)
                    img.save(_img_path)
                    self.sample_imagepaths.append(_img_path)
                    log.info(f"Saved {filename}")
                    sample_num += 1

            eKonf.clear_output(wait=True)
            log.info(" >> elapsed time to diffuse: {}".format(elapsed()))
            print(f"{args.n_samples} samples generated to {_batch_dir}")
            print(f"text prompts: {text_prompts}")
            print("sample image paths:")
            for p in self.sample_imagepaths:
                print(p)

            if args.show_collage:
                self.collage(image_filepaths=self.sample_imagepaths)

    def load_models(self):
        from dalle_mini import DalleBart, DalleBartProcessor
        from vqgan_jax.modeling_flax_vqgan import VQModel

        # check how many devices are available
        log.info(f"Available devices: {jax.local_device_count()}")
        if self._num_devices:
            self._num_devices = min(self._num_devices, jax.local_device_count())
        else:
            self._num_devices = jax.local_device_count()
        self.devices = jax.local_devices()
        log.info(f"Using {self._num_devices} devices")
        log.info(f"Devices: {self.devices}")

        # Load dalle-mini
        self.model, self.model_params = DalleBart.from_pretrained(
            self._model.DALLE_MODEL,
            revision=self._model.DALLE_COMMIT_ID,
            dtype=jnp.float16,
            _do_init=self._model.DALLE_INIT,
        )
        self.processor = DalleBartProcessor.from_pretrained(
            self._model.DALLE_MODEL,
            revision=self._model.DALLE_COMMIT_ID,
        )
        # Load VQGAN
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            self._model.VQGAN_REPO,
            revision=self._model.VQGAN_COMMIT_ID,
            _do_init=self._model.VQGAN_INIT,
        )

    def load_clip_models(self):
        self.clip, self.clip_params = FlaxCLIPModel.from_pretrained(
            self._model.CLIP_REPO,
            revision=self._model.CLIP_COMMIT_ID,
            dtype=jnp.float16,
            _do_init=self._model.CLIP_INIT,
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            self._model.CLIP_REPO,
            revision=self._model.CLIP_COMMIT_ID,
        )

    def rank_image_by_clip_score(self, prompts, images):
        """
        Rank an image by its clip score.
        """

        if self.clip is None:
            self.load_clip_models()

        _num_devices = self._num_devices
        _devices = self.devices[:_num_devices]
        log.info(f"Using {_num_devices} devices")
        log.info(f"Devices: {_devices}")

        _clip_params = replicate(self.clip_params, devices=_devices)

        # score images
        @partial(jax.pmap, axis_name="batch", devices=_devices)
        def p_clip(inputs, params):
            logits = self.clip(params=params, **inputs).logits_per_image
            return logits

        # from flax.training.common_utils import shard

        # get clip scores
        clip_inputs = self.clip_processor(
            text=prompts * _num_devices,
            images=images,
            return_tensors="np",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).data
        logits = p_clip(shard(clip_inputs), _clip_params)

        # organize scores per prompt
        p = len(prompts)
        logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()

        for i, prompt in enumerate(prompts):
            print(f"Prompt: {prompt}\n")
            for idx in logits[i].argsort()[::-1]:
                eKonf.display(images[idx * p + i])
                print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
            print()

    def load_config(self, batch_name=None, batch_num=None, **args):
        """Load the settings"""
        args = super().load_config(batch_name=batch_name, batch_num=batch_num, **args)

        if args.set_seed == "random_seed":
            random.seed()
            args.seed = random.randint(0, 2 ** 32 - 1)
        else:
            args.seed = int(args.set_seed)

        batch_arg_file = os.path.join(
            self._output.batch_dir, f"{args.batch_name}(*)_settings.yaml"
        )
        if args.resume_run:
            if args.run_to_resume == "latest":
                try:
                    args.batch_num
                except:
                    args.batch_num = len(glob(batch_arg_file)) - 1
            else:
                args.batch_num = int(args.run_to_resume)
        else:
            args.batch_num = len(glob(batch_arg_file))

        return args
