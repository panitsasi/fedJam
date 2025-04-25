"""fedjam-flower: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedjam_flower.task import get_weights, load_data, set_weights, test, train
from fedjam_flower.models import get_model, cosine_annealing

import timm
from torch import nn


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, valloader, local_epochs, context):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.context = context

    def fit(self, parameters, config):
        set_weights(self.model, parameters)

        # new_lr = cosine_annealing(
        #     int(config["current_round"]),
        #     self.context["num-server-rounds"],
        #     self.context["learning-rate-max"],
        #     self.context["learning-rate-min"],
        # )

        train_loss = train(
            self.model,
            self.trainloader,
            self.local_epochs,
            self.device,
            # learning_rate=new_lr,
        )
        return (
            get_weights(self.model),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    model = get_model(context.run_config["model_name"])
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(model, trainloader, valloader, local_epochs,
                        context.run_config).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
