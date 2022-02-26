import os
from omegaconf import OmegaConf
from hydra.utils import instantiate


class SimpleTraner:
    def __init__(self, **args):
        import wandb

        args = OmegaConf.create(args)
        dl = instantiate(args.dataset, _recursive_=False)
        datasets = dl.datasets
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)
        wandb.init(
            config=args,
            group=args.wandb_group,
            project=args.wandb_project,
            dir=args.wandb_dir,
        )

        self.train_data = datasets["train"]
        if "dev" in datasets:
            self.eval_data = datasets["dev"]
            args["evaluate_during_training"] = True
        else:
            self.eval_data = None
            args["evaluate_during_training"] = False

        self.test_data = datasets["test"]

        self.args = args


class SimpleTrainerNER(SimpleTraner):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.ner import NERModel

        args = self.args
        if args.labels is None:
            labels = list(self.train_data["labels"].unique())

        # Create a NERModel
        model = NERModel(
            args.model_type,
            args.model_uri,
            labels=labels,
            cuda_device=args.cuda_device,
            args=OmegaConf.to_container(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_data=self.eval_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions


class SimpleTrainerMultiLabel(SimpleTraner):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.classification import MultiLabelClassificationModel

        args = self.args
        # Create a Model
        model = MultiLabelClassificationModel(
            args.model_type,
            args.model_uri,
            num_labels=args.num_labels,
            args=OmegaConf.to_container(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_df=self.eval_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions


class SimpleTrainerClassification(SimpleTraner):
    def __init__(self, **args):
        super().__init__(**args)

    def train(self):
        from simpletransformers.classification import ClassificationModel

        args = self.args
        if args.num_labels is None:
            args.num_labels = len(self.train_data["labels"].unique())

        # Create a NERModel
        model = ClassificationModel(
            args.model_type,
            args.model_uri,
            num_labels=args.num_labels,
            cuda_device=args.cuda_device,
            args=OmegaConf.to_container(args),
        )

        # Train the model
        model.train_model(self.train_data, eval_df=self.eval_data)

        # Evaluate the model
        result, model_outputs, predictions = model.eval_model(self.test_data)

        # Check predictions
        # print(predictions[:5])
        return result, model_outputs, predictions
