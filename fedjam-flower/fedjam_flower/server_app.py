"""fedjam-flower: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedjam_flower.task import get_weights, set_weights, load_data, test

import timm
import torch
from torch import nn
from fedjam_flower.models import get_model

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_evaluate_fn(context: Context):
    model_name = context.run_config["model_name"]
    
    # Use timestamp to distinguish different runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("runs", model_name, f"{timestamp}")
    writer = SummaryWriter(log_dir)

    def evaluate(server_round: int, parameters, config):
        if server_round != 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(model_name).to(device).eval()
            _, testloader = load_data(0, 1)
            set_weights(model, parameters)
            loss, accuracy = test(model, testloader, device)
            print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/test', loss, server_round)
            writer.add_scalar('Accuracy/test', accuracy, server_round)

            return loss, {"accuracy": accuracy}
        else:
            print(f"Skipping server-side evaluation at first round {server_round}")
            return 0.0, {"accuracy": None}

    return evaluate


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    print(f"Context: {context}")
    model = get_model(context.run_config["model_name"])
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)
    evaluate = get_evaluate_fn(context)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        # min_fit_clients=1, # uncomment for centralized (1 client)
        initial_parameters=parameters,
        evaluate_fn=evaluate,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
