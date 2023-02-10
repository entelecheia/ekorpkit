import logging
from pydantic import BaseModel
from ekorpkit.models.transformer.trainers.base import BaseLMTrainer
from ekorpkit.tasks.multi import StableDiffusion


logger = logging.getLogger(__name__)


class GenerateConfig(BaseModel):
    """Generate config for prompt task"""

    prompt: str = None
    num_prompts_to_generate: int = 5
    # Maximum and min length of the generated prompts. Will cut off mid word. This is expected behavior
    max_prompt_length: int = 100
    min_prompt_length: int = 10
    # temperature: Default: 1.2. Turning up will inject more chaos.
    temperature: float = 1.2
    # top_k: Default 70. The number of top tokens returned by the AI. Will be randomly selected for generation.
    top_k: int = 70
    # top_p: Default 0.9. The total percent to consider from the `top_k` returned tokens.
    # For more information refer to [this guide!]( https://docs.cohere.ai/token-picking/)
    top_p: float = 0.9


class MethodConfig(BaseModel):
    generate: GenerateConfig = None


class PromptGenerator(BaseLMTrainer):
    method: MethodConfig = None
    generated_prompts: list = None
    __diffuser_obj__ = None

    class Config:
        underscore_attrs_are_private = False

    def __init__(self, config_name: str = "stable.prompt", **args):
        config_group = f"task={config_name}"
        super().__init__(config_name=config_name, config_group=config_group, **args)

    def initialize_configs(self, **args):
        super().initialize_configs(**args)
        if not isinstance(self.method, MethodConfig):
            self.method = MethodConfig(**self.method)

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
        model = self.model_obj

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

        generated_texts = []
        for i, generated_sequence in enumerate(output_sequences):
            tokens = list(generated_sequence)

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
        self.load_config(batch_name=batch_name)
        args = self.method.generate
        args.num_prompts_to_generate = num_prompts_to_generate
        args = args.copy(update=kwargs)

        if prompt is None:
            prompt = args.prompt
        if prompt is None:
            prompt = self.tokenizer.bos_token
        args.prompt = prompt

        generated_prompts = self._generate_text(
            prompt=prompt,
            num_return_sequences=args.num_prompts_to_generate,
            max_length=args.max_prompt_length,
            min_length=args.min_prompt_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        self.generated_prompts = generated_prompts
        self._generate_ = args
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
            logger.info("Initializing diffuser")
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
