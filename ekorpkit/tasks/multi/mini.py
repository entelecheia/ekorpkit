import logging
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key, shard
from PIL import Image
from tqdm.auto import trange
from ekorpkit.diffusers.base import BaseModel


log = logging.getLogger(__name__)


class DalleMini(BaseModel):
    def __init__(self, config_name: str = "default", **args):
        config_group = f"task/multi/disco={config_name}"
        super().__init__(config_group=config_group, **args)

        self.model = None
        self.model_params = None
        self.processor = None
        self.vqgan = None
        self.vqgan_params = None
        self.clip = None
        self.clip_params = None
        self.clip_processor = None
        self.devices = None

        if self.autoload:
            self.load()

    def generate(
        self,
        text_prompts=None,
        batch_name=None,
        batch_num=None,
        show_collage=False,
        **args,
    ):

        """Diffuse the model"""

        log.info("> loading settings...")
        config = self.load_config(
            batch_name=batch_name,
            batch_num=batch_num,
            imagine=args,
        )

        cfg = config.imagine
        text_prompts = text_prompts or eKonf.to_dict(cfg.text_prompts)
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        cfg.text_prompts = text_prompts

        self.sample_imagepaths = []

        tokenized_prompts = self.processor(cfg.text_prompts)
        print(f"Prompts: {cfg.text_prompts}\n")

        batch_dir = self.batch_dir
        num_devices = self.num_devices
        # _num_devices = min(_num_devices, args.num_samples)
        devices = self.devices[:num_devices]
        log.info(f"Using {num_devices} devices")
        log.info(f"Devices: {devices}")
        cfg.num_samples = max(cfg.num_samples // num_devices, 1) * num_devices

        # Keys are passed to the model on each device to generate unique inference per device.
        # create a random key
        log.info(f"Seed used: {self.seed}")
        key = jax.random.PRNGKey(self.seed)
        # Model parameters are replicated on each device for faster inference.
        model_params = replicate(self.model_params, devices=devices)
        vqgan_params = replicate(self.vqgan_params, devices=devices)
        tokenized_prompts = replicate(tokenized_prompts, devices=devices)

        # Model functions are compiled and parallelized to take advantage of multiple devices.
        # model inference
        @partial(
            jax.pmap,
            axis_name="batch",
            static_broadcasted_argnums=(3, 4, 5, 6),
            devices=devices,
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
        @partial(jax.pmap, axis_name="batch", devices=devices)
        def p_decode(indices, params):
            return self.vqgan.decode_code(indices, params=params)

        with elapsed_timer(format_time=True) as elapsed:

            # generate images
            # sample_images = []
            sample_num = 0
            for i in trange(max(cfg.num_samples // num_devices, 1)):
                eKonf.clear_output(wait=True)
                # get a new key
                key, subkey = jax.random.split(key)
                # generate images
                encoded_images = p_generate(
                    tokenized_prompts,
                    shard_prng_key(subkey),
                    model_params,
                    cfg.gen_top_k,
                    cfg.gen_top_p,
                    cfg.temperature,
                    cfg.cond_scale,
                )
                # remove BOS
                encoded_images = encoded_images.sequences[..., 1:]
                # decode images
                decoded_images = p_decode(encoded_images, vqgan_params)
                decoded_images = decoded_images.clip(0.0, 1.0).reshape(
                    (-1, 256, 256, 3)
                )
                for decoded_img in decoded_images:
                    img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                    eKonf.display(img)
                    # sample_images.append(img)
                    filename = (
                        f"{self.batch_name}({self.batch_num})_{sample_num:04}.png"
                    )
                    img_path = str(batch_dir / filename)
                    img.save(img_path)
                    self.sample_imagepaths.append(img_path)
                    log.info(f"Saved {filename}")
                    sample_num += 1

            eKonf.clear_output(wait=True)
            cfg.num_samples = sample_num
            log.info(" >> elapsed time to diffuse: {}".format(elapsed()))
            print(f"{cfg.num_samples} samples generated to {batch_dir}")
            print(f"text prompts: {text_prompts}")

            if show_collage:
                self.collage_images(images_or_uris=self.sample_imagepaths)

        config.imagine = cfg
        self.config = config
        results = {
            "image_filepaths": self.sample_imagepaths,
            "config_file": self.save_config(),
            "config": eKonf.to_dict(config),
        }
        return results

    def load_models(self):
        from dalle_mini import DalleBart, DalleBartProcessor
        from vqgan_jax.modeling_flax_vqgan import VQModel

        # check how many devices are available
        log.info(f"Available devices: {jax.local_device_count()}")
        if self.num_devices:
            self.num_devices = min(self.num_devices, jax.local_device_count())
        else:
            self.num_devices = jax.local_device_count()
        self.devices = jax.local_devices()
        log.info(f"Using {self.num_devices} devices")
        log.info(f"Devices: {self.devices}")

        # Load dalle-mini
        self.model, self.model_params = DalleBart.from_pretrained(
            self.model.DALLE_MODEL,
            revision=self.model.DALLE_COMMIT_ID,
            dtype=jnp.float16,
            _do_init=self.model.DALLE_INIT,
        )
        self.processor = DalleBartProcessor.from_pretrained(
            self.model.DALLE_MODEL,
            revision=self.model.DALLE_COMMIT_ID,
        )
        # Load VQGAN
        self.vqgan, self.vqgan_params = VQModel.from_pretrained(
            self.model.VQGAN_REPO,
            revision=self.model.VQGAN_COMMIT_ID,
            _do_init=self.model.VQGAN_INIT,
        )

    def load_clip_models(self):
        self.clip, self.clip_params = FlaxCLIPModel.from_pretrained(
            self.model.CLIP_REPO,
            revision=self.model.CLIP_COMMIT_ID,
            dtype=jnp.float16,
            _do_init=self.model.CLIP_INIT,
        )
        self.clip_processor = CLIPProcessor.from_pretrained(
            self.model.CLIP_REPO,
            revision=self.model.CLIP_COMMIT_ID,
        )

    def rank_image_by_clip_score(self, prompts, images):
        """
        Rank an image by its clip score.
        """

        if self.clip is None:
            self.load_clip_models()

        _num_devices = self.num_devices
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
