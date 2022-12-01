import logging
import random
import wandb
from pathlib import Path
from omegaconf import DictConfig
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from ekorpkit import eKonf
from ekorpkit.config import BaseBatchModel
from ekorpkit.datasets.config import DatasetConfig
from .stable import StableDiffusion


log = logging.getLogger(__name__)


class PromptGenerator(BaseBatchModel):
    model: DictConfig = None
    dataset: DatasetConfig = None
    tokenizer: DictConfig = None
    _generate_: DictConfig = None
    _train_: DictConfig = None
    generated_prompts: list = None
    __model_obj__ = None
    __dataset__ = None
    __tokenizer_obj__ = None
    __diffuser_obj__ = None

    class Config:
        underscore_attrs_are_private = False

    def __init__(self, config_group: str = "app/prompt", **args):
        super().__init__(config_group=config_group, **args)
        self._init_configs()

    def _init_configs(self):
        self.model.cache_dir = str(self.cache_dir)

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

        tokenizer = self.tokenizer_obj
        model = self.model_obj

        if prompt is None:
            prompt = cfg.prompt
        if prompt is None:
            prompt = self.bos_token
        cfg.prompt = prompt

        encoded_prompt = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(model.device)

        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=cfg.max_prompt_length,
            min_length=cfg.min_prompt_length,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            do_sample=True,
            num_return_sequences=cfg.num_prompts_to_generate,
            pad_token_id=tokenizer.pad_token_id,  # gets rid of warning
        )

        tokenized_start_token = tokenizer.encode(self.bos_token)

        generated_prompts = []
        for i, generated_sequence in enumerate(output_sequences):
            if not cfg.check_start_token:
                tokens = list(generated_sequence)
            else:
                tokens = []
                for i, s in enumerate(generated_sequence):
                    if s in tokenized_start_token and i != 0:
                        if len(tokens) >= cfg.min_prompt_length:
                            break
                    tokens.append(s)

            text = tokenizer.decode(
                tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            text = (
                text.strip().replace("\n", " ").replace("/", ",")
            )  # / remove slash. It causes problems in namings
            generated_prompts.append(text)
            if self.verbose:
                log.info(f"Prompt {i}: {text}")

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
            log.warning("No prompts provided")
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

    @property
    def model_obj(self):
        if self.__model_obj__ is None:
            self.load_model()
        return self.__model_obj__

    @property
    def model_name(self):
        return self.config.model.model_name

    @model_name.setter
    def model_name(self, model_name):
        self.config.model.model_name = model_name

    def load_model(self, model_name: str = None, model_dir: str = None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        if model_dir is None:
            model_dir = self.model_dir
        if model_name is None:
            log.warning("No model provided")
            return None

        if self.__model_obj__ is None or model_name != self.model_name:

            model_path = Path(model_dir) / model_name
            log.info(f"Loading model from {model_path}")
            self.__model_obj__ = AutoModelForCausalLM.from_pretrained(
                model_path, **kwargs
            )
            self.__model_obj__.to(self.device)
            if self.verbose:
                log.info(f"Model: {self.model_obj}")
            self.model_name = model_name

        else:
            log.info("Model already loaded")

    def tokenize(
        self,
        dataset,
        prompt_column,
        shuffle_prompts=False,
        num_shuffles=4,
        batch_size=10000,
        split="train",
        padding="max_length",
    ):
        tokenizer = self.tokenizer_obj
        if tokenizer is None:
            log.warning("No tokenizer provided")
            return None
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]

        def _tokenize(dataset):
            # Remove empty lines
            dataset[prompt_column] = [
                self.bos_token + prompt.strip() + self.eos_token
                for prompt in dataset[prompt_column]
                if len(prompt.strip()) > 0
            ]

            outputs = tokenizer(
                dataset[prompt_column],
                padding=padding,
                truncation=True,
                # return_special_tokens_mask=True,
                max_length=tokenizer.model_max_length,
            )
            return {"input_ids": outputs["input_ids"]}

        def _tokenize_shuffled(dataset):
            block_size = tokenizer.model_max_length
            prompts = dataset[prompt_column]

            shuffled_prompts = ""
            for _ in range(num_shuffles):
                random.shuffle(prompts)
                concat = "".join(
                    [
                        self.bos_token + prompt.strip() + " " + self.eos_token
                        for prompt in prompts
                        if len(prompt.strip()) > 0
                    ]
                )
                shuffled_prompts += concat

            input_ids = tokenizer(shuffled_prompts)["input_ids"]

            result = []
            for i in range(
                0, len(input_ids) - block_size + 1, block_size
            ):  # Truncate in block of block_size
                result.append(
                    tokenizer.build_inputs_with_special_tokens(
                        input_ids[i : i + block_size]
                    )
                )
            return {"input_ids": result}

        if shuffle_prompts:
            _tokenize_function = _tokenize_shuffled
        else:
            _tokenize_function = _tokenize

        tokenized_dataset = dataset.map(
            _tokenize_function,
            batched=True,
            batch_size=batch_size,
            drop_last_batch=False,
            remove_columns=[prompt_column],
        )
        return tokenized_dataset

    @property
    def training_dir(self):
        return self.output_dir / "training"

    @property
    def trainer_config(self):
        return self.config.trainer

    @trainer_config.setter
    def trainer_config(self, value):
        self.config.trainer = value

    def train(
        self,
        model_name=None,
        pretrained_model_name_or_path=None,
        dataset=None,
        prompt_column=None,
        split=None,
    ):
        if model_name is None:
            model_name = self.model_name

        if model_name is None:
            log.warning("No model provided")
            return None

        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = self._train_.pretrained_model_name_or_path
        else:
            self._train_.pretrained_model_name_or_path = pretrained_model_name_or_path
        if prompt_column is None:
            prompt_column = self._train_.prompt_column
        else:
            self._train_.prompt_column = prompt_column
        if split is None:
            split = self._train_.split
        else:
            self._train_.split = split

        # load up the dataset
        if dataset is None:
            dataset = self.raw_datasets

        model_train_dir = self.training_dir / model_name
        cache_dir = self.cache_dir

        shuffle_prompts = self._train_.shuffle_prompts
        num_shuffles = self._train_.num_shuffles
        batch_size = self._train_.batch_size
        padding = self._train_.padding

        tokenizer = self.tokenizer_obj
        tokenized_dataset = self.tokenize(
            dataset=dataset,
            prompt_column=prompt_column,
            shuffle_prompts=shuffle_prompts,
            num_shuffles=num_shuffles,
            batch_size=batch_size,
            split=split,
            padding=padding,
        )

        # train the model
        self.trainer_config.seed = self.seed
        self.trainer_config.run_name = model_name
        training_args = TrainingArguments(
            output_dir=model_train_dir,
            **self.trainer_config,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, cache_dir=cache_dir
        )
        model.resize_token_embeddings(len(tokenizer))
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )
        trainer.train()

        wandb.finish()
        trainer = None

        self.generated_prompts = None
        self.save_model(model, model_name=model_name)

    def save_model(self, model, model_name=None, model_dir=None, **kwargs):
        if model is None:
            log.warning("No model provided")
            return None

        if model_name is None:
            model_name = self.model_name
        if model_dir is None:
            model_dir = self.model_dir

        if model_name is None:
            log.warning("No model provided")
            return None

        model_path = Path(model_dir) / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving model to {model_path}")
        model.save_pretrained(model_path, **kwargs)
        self.__model_obj__ = model
        self.__model_obj__.to(self.config.device)
        self.model_name = model_name
        self.save_config()

    @property
    def raw_datasets(self):
        return self.dataset.datasets

    @property
    def tokenizer_obj(self):
        if self.__tokenizer_obj__ is None:
            self.load_tokenizer()
        return self.__tokenizer_obj__

    def load_tokenizer(self):
        tokenizer = self.tokenizer

        tokenizer_kwargs = {
            "cache_dir": tokenizer.cache_dir,
            "use_fast": tokenizer.use_fast_tokenizer,
            "revision": tokenizer.model_revision,
            "use_auth_token": tokenizer.use_auth_token,
        }
        if tokenizer.tokenizer_name:
            tokenizer_obj = AutoTokenizer.from_pretrained(
                tokenizer.tokenizer_name, **tokenizer_kwargs
            )
        elif tokenizer.model_name_or_path:
            tokenizer_obj = AutoTokenizer.from_pretrained(
                tokenizer.model_name_or_path, **tokenizer_kwargs
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        special_tokens = {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
        }
        tokenizer_obj.add_special_tokens(special_tokens)
        self.__tokenizer_obj__ = tokenizer_obj
        if self.verbose:
            log.info(f"Tokenizer: {self.__tokenizer_obj__}")
