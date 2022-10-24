import logging
import torch
from transformers import pipeline
from huggingface_hub import InferenceApi
from ekorpkit.utils.func import elapsed_timer
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class BloomDemo:
    def __init__(
        self,
        model_uri="bigscience/bloom",
        device=0,
        hf_user_access_token=None,
        **kwargs,
    ):
        if kwargs:
            self.args = eKonf.to_config(kwargs)
        else:
            self.args = {}
        self.model_uri = model_uri
        self.device = device
        self.hf_user_access_token = hf_user_access_token
        self.auto_load = self.args.get("auto", {}).get("load", False)

        self.inference_api = None
        self.pipe = None
        self.TRANSFORMERS_CACHE = eKonf.osenv("TRANSFORMERS_CACHE")

    def init_pipe(self, model_uri=None, task="text-generation", device=None):
        if model_uri is None:
            model_uri = self.model_uri
        if device is None:
            device = self.device if torch.cuda.is_available() else -1
        with elapsed_timer(format_time=True) as elapsed:
            self.pipe = pipeline(
                model=model_uri,
                task=task,
                device=device,
                torch_dtype=torch.bfloat16,
                cache_dir=self.TRANSFORMERS_CACHE,
            )
            print(" >> elapsed time to load model: {}".format(elapsed()))

    def init_api(self, model_uri=None, token=None):
        if model_uri is None:
            model_uri = self.model_uri
        if token is None:
            token = self.hf_user_access_token or eKonf.osenv("HF_USER_ACCESS_TOKEN")
        self.inference_api = InferenceApi(model_uri, token=token)
        print(" >> Inference API initialized")

    def infer(
        self,
        prompt,
        inference_method="api",
        temperature=0.7,
        top_k=None,
        top_p=None,
        max_new_tokens=50,
        repetition_penalty=None,
        do_sample=False,
        num_return_sequences=1,
        num_beams=None,
        no_repeat_ngram_size=None,
        early_stopping=False,
        return_full_text=True,
        **kwargs,
    ):
        if inference_method == "api":
            response = self.infer_api(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                return_full_text=return_full_text,
                **kwargs,
            )
        else:
            response = self.infer_local(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                return_full_text=return_full_text,
                **kwargs,
            )

        generated_text = response[0]["generated_text"]
        return generated_text

    def infer_api(
        self,
        prompt,
        temperature=0.7,
        top_k=None,
        top_p=None,
        max_new_tokens=50,
        repetition_penalty=None,
        do_sample=False,
        num_return_sequences=1,
        num_beams=None,
        no_repeat_ngram_size=None,
        early_stopping=False,
        return_full_text=True,
        seed=123,
        **kwargs,
    ):
        if self.inference_api is None:
            self.init_api()

        top_k = None if top_k == 0 else top_k
        top_p = None if num_beams else top_p
        num_beams = None if num_beams == 0 else num_beams
        no_repeat_ngram_size = None if num_beams is None else no_repeat_ngram_size
        early_stopping = None if num_beams is None else num_beams > 0
        max_new_tokens = min(int(max_new_tokens), 2047)

        params = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "do_sample": do_sample,
            "early_stopping": early_stopping,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "return_full_text": return_full_text,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        }

        response = self.inference_api(prompt, params=params)
        return response

    def infer_local(
        self,
        prompt,
        temperature=0.7,
        top_k=None,
        top_p=None,
        max_new_tokens=50,
        repetition_penalty=None,
        do_sample=False,
        num_return_sequences=1,
        num_beams=None,
        no_repeat_ngram_size=None,
        early_stopping=False,
        return_full_text=True,
        **kwargs,
    ):
        if self.pipe is None:
            self.init_pipe()
        max_new_tokens = min(int(max_new_tokens), 2047)

        response = self.pipe(
            prompt,
            temperature=temperature,  # 0 to 1
            top_k=top_k,
            top_p=top_p,  # None, 0-1
            max_new_tokens=max_new_tokens,  # up to 2047 theoretically
            return_full_text=return_full_text,  # include prompt or not.
            repetition_penalty=repetition_penalty,  # None, 0-100 (penalty for repeat tokens.
            do_sample=do_sample,  # True: use sampling, False: Greedy decoding.
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )
        return response
