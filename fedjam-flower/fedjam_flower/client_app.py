"""fedjam-flower: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedjam_flower.task import (
    get_weights, load_data, set_weights,
    test, train, set_seed
)
from fedjam_flower.models import get_model, cosine_annealing

import timm
from torch import nn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs, context):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.context = context
        self.is_lora = context["is_lora"]
        self.num_server_rounds = context["num-server-rounds"]
        self.is_timm = context["is_timm"]
        self.is_warmup = context["is_warmup"]
        self.random_seed = context["random_seed"]
        self.is_multimodal = 'multimodal' in context["model_name"].lower()

    def fit(self, parameters, config):
        # Set seed for reproducibility
        set_seed(self.random_seed)
        set_weights(self.model, parameters, is_lora=self.is_lora)
        current_round = int(config["current_round"])

        new_lr = cosine_annealing(
            current_round,
            self.context["num-server-rounds"],
            self.context["learning-rate-max"],
            self.context["learning-rate-min"],
        )

        # (Warmup) Stage 1: Head-only training for first N rounds
        if self.is_warmup:
            if current_round <= 0.1 * self.num_server_rounds:
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = True

        print(f"Training...", flush=True)
        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
            lr=new_lr,
            is_lora=self.is_lora,
            is_timm=self.is_timm,
            is_multimodal=self.is_multimodal,
        )
        return (
            get_weights(self.model, is_lora=self.is_lora),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters, is_lora=self.is_lora)
        loss, accuracy = test(self.model, self.valloader, self.device, is_lora=self.is_lora, is_timm=self.is_timm,
                              is_multimodal=self.is_multimodal)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    model_name = context.run_config["model_name"]
    model = get_model(model_name, context.run_config["is_lora"], context.run_config["is_timm"])
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data_dir = context.node_config.get("data_dir") or context.run_config["data_dir"]
    batch_size = context.run_config["batch_size"]
    classes_per_partition = context.run_config["classes_per_partition"]
    is_multimodal = 'multimodal' in model_name.lower()
    trainloader, valloader = load_data(partition_id, num_partitions,
                                       data_dir=data_dir, batch_size=batch_size,
                                       classes_per_partition=classes_per_partition,
                                       is_multimodal=is_multimodal)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(model, trainloader, valloader, local_epochs,
                        context.run_config).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
