import logging
from omegaconf import DictConfig
from ekorpkit import eKonf
from ekorpkit.models.transformer.trainers.base import BaseTrainer
from .stable import StableDiffusion


logger = logging.getLogger(__name__)


class PromptGenerator(BaseTrainer):
    _generate_: DictConfig = None
    generated_prompts: list = None
    __diffuser_obj__ = None

    class Config:
        underscore_attrs_are_private = False

    def __init__(self, config_group: str = "app/prompt", **args):
        super().__init__(config_group=config_group, **args)

    def _generate_text(
        self,
        prompt,
        num_return_sequences=5,
        max_length=50,
        min_length=30,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        **kwargs,
    ):
        tokenizer = self.tokenizer_obj
        model = self.model_obj.to(self.device)

        encoded_prompt = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(model.device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,  # gets rid of warning
        )

        # tokenized_start_token = tokenizer.encode(self.bos_token)

        generated_texts = []
        for i, generated_sequence in enumerate(output_sequences):
            tokens = list(generated_sequence)
            # else:
            #     tokens = []
            #     for i, s in enumerate(generated_sequence):
            #         if s in tokenized_start_token and i != 0:
            #             if len(tokens) >= cfg.min_prompt_length:
            #                 break
            #         tokens.append(s)

            text = tokenizer.decode(
                tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            text = (
                text.strip().replace("\n", " ").replace("/", ",")
            )  # / remove slash. It causes problems in namings
            generated_texts.append(text)
            if self.verbose:
                logger.info(f"Prompt {i}: {text}")
        return generated_texts

    def generate_prompts(
        self,
        prompt=None,
        num_prompts_to_generate=5,
        batch_name=None,
        generate_images=True,
        num_samples=3,
        **kwargs,
    ):
        self.load_config()
        cfg = self._generate_
        cfg.num_prompts_to_generate = num_prompts_to_generate
        cfg = eKonf.merge(cfg, kwargs)

        if prompt is None:
            prompt = cfg.prompt
        if prompt is None:
            prompt = self.bos_token
        cfg.prompt = prompt

        generated_prompts = self._generate_text(**cfg)
        self.generated_prompts = generated_prompts
        self._generate_ = cfg
        self.save_config()

        if generate_images:
            self.generate_images(
                generated_prompts, batch_name=batch_name, num_samples=num_samples
            )
        return generated_prompts

    def generate_images(
        self,
        prompts=None,
        batch_name=None,
        num_samples=3,
        max_display_image_width=800,
        **kwargs,
    ):
        if prompts is None:
            prompts = list(self.generated_prompts)
        if prompts is None:
            logger.warning("No prompts provided")
            return None
        if not isinstance(prompts, list):
            prompts = [prompts]

        if batch_name is None:
            batch_name = f"{self.name}_batch"
        batch_run_params = {"text_prompts": prompts}
        batch_run_pairs = [["text_prompts"]]

        batch_results = self.diffuser_obj.batch_imagine(
            batch_name=batch_name,
            batch_run_params=batch_run_params,
            batch_run_pairs=batch_run_pairs,
            num_samples=num_samples,
            max_display_image_width=max_display_image_width,
            **kwargs,
        )

        return batch_results

    @property
    def diffuser_obj(self):
        if self.__diffuser_obj__ is None:
            self.__diffuser_obj__ = StableDiffusion()
        return self.__diffuser_obj__

    def imagine(
        self,
        text_prompts=None,
        batch_name=None,
        num_samples=1,
        **imagine_args,
    ):
        if batch_name is None:
            batch_name = self.name
        return self.diffuser_obj.generate(
            text_prompts=text_prompts,
            batch_name=batch_name,
            num_samples=num_samples,
            **imagine_args,
        )
