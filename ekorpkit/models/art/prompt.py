import logging
import random
import wandb
import datasets
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from ekorpkit import eKonf
from tqdm.auto import tqdm
from functools import partialmethod
from ekorpkit.batch import BaseConfig


def disable_tqdm():
    # THIS IS A DIRTY HACK TO SILENCE THE PROGRESS BAR
    # THE TQDM PROGRESS BAR BOTTLENECKS CELL OUTPUT AND SLOWS THE NOTEBOOK
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


log = logging.getLogger(__name__)


class PromptGenerator(BaseConfig):
    start_token = "<BOP>"
    pad_token = "<PAD>"
    end_token = "<EOP>"
    special_tokens = {
        "bos_token": start_token,
        "eos_token": end_token,
        "pad_token": pad_token,
    }

    def __init__(self, root_dir=None, config_name="default", **args):
        cfg = eKonf.compose(f"model/prompt={config_name}")
        cfg = eKonf.merge(cfg, args)
        super().__init__(root_dir=root_dir, **cfg)
        eKonf.env_set("WANDB_PROJECT", cfg.wandb_project)

        self.model = None
        self.dataset = None
        self.tokenizer = None

    def generate_prompts(self, prompt=None, **kwargs):
        self.load_config()
        cfg = self.config.generate
        cfg = eKonf.merge(cfg, kwargs)

        if self.tokenizer is None:
            self.load_tokenizer()
        tokenizer = self.tokenizer
        if self.model is None:
            self.load_model()
        model = self.model

        if prompt is None:
            prompt = cfg.prompt
        if prompt is None:
            prompt = self.start_token
        cfg.prompt = prompt

        encoded_prompt = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        encoded_prompt = encoded_prompt.to(model.device)

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

        tokenized_start_token = tokenizer.encode(self.start_token)

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

        self.config.generated_prompts = generated_prompts
        self.config.generate = cfg
        self.save_config()
        return generated_prompts

    def load_dataset(self, dset=None, **kwargs):
        if dset is None:
            dset = self.config.dataset
        if dset is None:
            log.warning("No dataset provided")
            return None

        if isinstance(dset, str):
            self.dataset = datasets.load_dataset(dset, **kwargs)
        elif eKonf.is_config(dset):
            self.dataset = datasets.load_dataset(
                dset.path,
                cache_dir=self.path.cache_dir,
                download_mode=dset.download_mode,
                **kwargs,
            )
        else:
            self.dataset = dset
        if self.verbose:
            log.info(f"Dataset: {self.dataset}")

    def load_tokenizer(self, tokenizer=None, **kwargs):
        if tokenizer is None:
            tokenizer = self.config.tokenizer
        if tokenizer is None:
            log.warning("No tokenizer provided")
            return None

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **kwargs)
        elif eKonf.is_config(tokenizer):
            self.start_token = tokenizer.start_token
            self.end_token = tokenizer.end_token
            self.pad_token = tokenizer.pad_token
            self.special_tokens = {
                "bos_token": self.start_token,
                "eos_token": self.end_token,
                "pad_token": self.pad_token,
            }
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer.pretrained_model_name_or_path,
                cache_dir=self.path.cache_dir,
                **kwargs,
            )
        else:
            self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens(self.special_tokens)
        if self.verbose:
            log.info(f"Tokenizer: {self.tokenizer}")

    def load_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.config.model.model_name
        if model_name is None:
            log.warning("No model provided")
            return None
        self.config.model.model_name = model_name

        model_path = self.model_dir / model_name
        log.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.to(self.config.device)
        if self.verbose:
            log.info(f"Model: {self.model}")

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
        if self.tokenizer is None:
            self.load_tokenizer()
        tokenizer = self.tokenizer
        if tokenizer is None:
            log.warning("No tokenizer provided")
            return None
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]

        def _tokenize(dataset):
            # Remove empty lines
            dataset[prompt_column] = [
                self.start_token + prompt.strip() + self.end_token
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
                        self.start_token + prompt.strip() + " " + self.end_token
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

    def train(self, model_name=None):
        if model_name is None:
            model_name = self.config.model.model_name
        if model_name is None:
            log.warning("No model provided")
            return None
        self.config.model.model_name = model_name

        model_train_dir = self.path.training_dir / model_name
        cache_dir = self.path.cache_dir
        pretrained_model_name_or_path = self.config.model.pretrained_model_name_or_path
        prompt_column = self.config.dataset.prompt_column
        split = self.config.dataset.split

        shuffle_prompts = self.config.tokenize.shuffle_prompts
        num_shuffles = self.config.tokenize.num_shuffles
        batch_size = self.config.tokenize.batch_size
        padding = self.config.tokenize.padding

        tokenizer = self.tokenizer
        # load up the dataset
        if self.dataset is None:
            self.load_dataset()
        tokenized_dataset = self.tokenize(
            dataset=self.dataset,
            prompt_column=prompt_column,
            shuffle_prompts=shuffle_prompts,
            num_shuffles=num_shuffles,
            batch_size=batch_size,
            split=split,
            padding=padding,
        )

        # train the model
        self.config.trainer.seed = self.seed
        training_args = TrainingArguments(
            output_dir=model_train_dir,
            **self.config.trainer,
        )
        self.model = None
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

        self.model = model
        self.model.to(self.config.device)
        self.config.generated_prompts = None
        self.save_model(model_name=model_name)

    def save_model(self, model_name=None, **kwargs):
        if self.model is None:
            log.warning("Train model first")
            return None

        if model_name is None:
            model_name = self.config.model.model_name
        if model_name is None:
            log.warning("No model provided")
            return None
        self.config.model.model_name = model_name

        model_path = self.path.model_dir / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving model to {model_path}")
        self.model.save_pretrained(model_path, **kwargs)
        self.save_config()
