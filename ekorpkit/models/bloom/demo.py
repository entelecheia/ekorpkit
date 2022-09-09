import logging
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from enum import Enum
from ekorpkit.utils.func import elapsed_timer
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class DecodingMethods(Enum):
    greedy_search = "greedy_search"
    beam_search = "beam_search"
    top_k_sampling = "top_k_sampling"
    top_p_sampling = "top_p_sampling"


class BloomDemo:
    def __init__(
        self,
        model_uri="bigscience/bloom",
        **kwargs,
    ):
        if kwargs:
            self.args = eKonf.to_config(kwargs)
        else:
            self.args = {}
        self.model_uri = model_uri
        self.auto_load = self.args.get("auto", {}).get("load", False)

        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.TRANSFORMERS_CACHE = eKonf.osenv("TRANSFORMERS_CACHE")

        if self.auto_load:
            self.init_model()

    def init_model(self):
        with elapsed_timer(format_time=True) as elapsed:
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_uri, cache_dir=self.TRANSFORMERS_CACHE)
            # self.model = AutoModel.from_pretrained(self.model_uri, cache_dir=self.TRANSFORMERS_CACHE)
            self.pipe = pipeline(
                model=self.model_uri,
                torch_dtype=torch.bfloat16,
                cache_dir=self.TRANSFORMERS_CACHE
            )
            print(" >> elapsed time to load model: {}".format(elapsed()))

    def query(
        self,
        prompt,
        temperature=0.7,
        top_p=None,
        max_new_tokens=32,
        repetition_penalty=None,
        do_sample=False,
        num_return_sequences=1,
        length_penalty=0.0,
        eos_token_id=None,
        seed=42,
    ):
        response = self.pipe(
            f"{prompt}",
            temperature=temperature,  # 0 to 1
            top_p=top_p,  # None, 0-1
            max_new_tokens=max_new_tokens,  # up to 2047 theoretically
            return_full_text=False,  # include prompt or not.
            repetition_penalty=repetition_penalty,  # None, 0-100 (penalty for repeat tokens.
            do_sample=do_sample,  # True: use sampling, False: Greedy decoding.
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            eos_token_id=eos_token_id,
            seed=seed,
        )
        return response

    def inference(
        self,
        prompt,
        max_length,
        decoding_method: DecodingMethods = DecodingMethods.greedy_search,
        length_penalty=0.0,
        eos_token_id=None,
        seed=42,
    ):
        if decoding_method == DecodingMethods.top_p_sampling:
            parameters = {
                "max_new_tokens": max_length,
                "top_p": 0.9,
                "do_sample": True,
                "seed": seed,
                "early_stopping": False,
                "length_penalty": length_penalty,
                "eos_token_id": None,
            }
        else:
            parameters = {
                "max_new_tokens": max_length,
                "do_sample": False,
                "seed": seed,
                "early_stopping": False,
                "length_penalty": length_penalty,
                "eos_token_id": None,
            }

        payload = {
            "prompt": prompt,
            "parameters": parameters,
            "options": {"use_cache": False},
        }

        data = self.query(payload)

        generation = data[0]["generated_text"].split(prompt, 1)[1]
        return (prompt, generation)
