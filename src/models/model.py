import random
import warnings
from abc import ABC, abstractmethod

import numpy as np
import sklearn.exceptions
import torch
import torch.nn as nn

from src.config.sweep_config_args import sweep_config_args

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

# TODO: remove unnecessary methods
# TODO: remove unwanted self variables and covert to function variables
class Model(ABC):
    def __init__(
            self,
            dataset,
            dataset_dir,
            model_name,
            model_type,
            embedding_model_name,
            sweep_config_enable=False,
            wandb_project_name=None,
            wandb_agent_count=None,
            save_model=False,
            output_dir=None,
            evaluate_model=False,
            is_Wandb=False,
            training_strategy=None,
            tokenizer_name=None,
            embedding_class=None,
            fake_percentage = 1,
            **kwargs
    ):
        self.embedding_matrix = None
        self.pos_label = None
        self.label_list = None
        self.data_module = None
        self.test_dataloader = None
        self.optimizer = None
        self.model = None
        self.val_dataloader = None
        self.train_dataloader = None
        self.sweep_config_enable = sweep_config_enable
        self.wandb_project_name = wandb_project_name
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.save_model = save_model
        self.output_dir = output_dir
        self.wandb_agent_count = wandb_agent_count
        self.evaluate_model = evaluate_model
        self.is_Wandb = is_Wandb
        self.model_name = model_name
        self.model_type = model_type
        self.training_strategy = training_strategy
        self.embedding_model_name = embedding_model_name
        self.tokenizer_name = tokenizer_name
        self.embedding_class = embedding_class
        self.fake_percentage = fake_percentage

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

        self.loss_fn = nn.CrossEntropyLoss()
        self.set_seed(42)

        if wandb_project_name and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.wandb_project = None

        if self.sweep_config_enable:
            self.sweep_id = wandb.sweep(sweep_config_args, project=wandb_project_name)

    @abstractmethod
    def train(self, config):
        pass

    @staticmethod
    def set_seed(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def wandb_train(self):
        if self.sweep_config_enable:
            with wandb.init(config=sweep_config_args) as run:
                self.train(wandb.config)
            wandb.finish()

    # Exposed methods
    def train_with_sweep(self):
        print(self.wandb_agent_count)
        wandb.agent(self.sweep_id, self.wandb_train, count=self.wandb_agent_count)

    def best_model_train(self, params):
        if self.evaluate_model:
            with wandb.init(config=params) as run:
                print('write code for evaluate')
            wandb.finish()
        else:
            with wandb.init(config=params) as run:
                self.train(wandb.config)
            wandb.finish()
