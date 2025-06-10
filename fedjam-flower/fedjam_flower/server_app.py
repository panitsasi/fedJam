"""fedjam-flower: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedjam_flower.task import (
    get_weights, set_weights, load_data,
    test, set_seed
)

import timm
import torch
from torch import nn
from fedjam_flower.models import get_model

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_evaluate_fn(context: Context):
    model_name = context.run_config["model_name"]
    num_clients = context.run_config["num_clients"]
    fft = context.run_config["fft"]
    gain = context.run_config["gain"]
    data_dir = context.run_config["data_dir"]
    batch_size = context.run_config["batch_size"]
    is_lora = context.run_config["is_lora"]
    is_timm = context.run_config["is_timm"]
    dataset_version = context.run_config["dataset_version"]
    classes_per_partition = context.run_config["classes_per_partition"]
    is_multimodal = 'multimodal' in model_name.lower()
    
    # Use timestamp to distinguish different runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"runs/{dataset_version}/{model_name}_clients_{num_clients}"
    if fft != 0:
        filename += f"_fft_{fft}"
    if gain != 0:
        filename += f"_gain_{gain}"
    if is_lora:
        filename += "_lora"
    if classes_per_partition != 4:
        filename += f"_non_iid_{classes_per_partition}"
    else:
        filename += "_iid"
    log_dir = os.path.join(filename, f"{timestamp}")
    writer = SummaryWriter(log_dir)

    # Save the entire run_config to a JSON file
    config_filename = os.path.join(log_dir, "run_config.json")
    with open(config_filename, 'w') as f:
        json.dump(context.run_config, f, indent=4)

    # Log the entire run_config dictionary as a single JSON string in TensorBoard
    writer.add_text("run_config", json.dumps(context.run_config, indent=4))

    def evaluate(server_round: int, parameters, config):
        if server_round != 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(model_name, is_lora, is_timm).to(device).eval()
            _, testloader = load_data(0, 1, data_dir=data_dir, batch_size=batch_size,
                                       is_multimodal=is_multimodal)
            set_weights(model, parameters, is_lora=is_lora)
            loss, accuracy = test(model, testloader, device, is_lora=is_lora, is_timm=is_timm,
                                  is_multimodal=is_multimodal)
            print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/test', loss, server_round)
            writer.add_scalar('Accuracy/test', accuracy, server_round)

            return loss, {"accuracy": accuracy}
        else:
            print(f"Skipping server-side evaluation at first round {server_round}")
            return 0.0, {"accuracy": None}

    return evaluate


def get_on_fit_config(is_lora: bool):
    """Return a function that will be used to construct the config that the
    client's fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["is_lora"] = is_lora
        return fit_config

    return fit_config_fn

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    model = get_model(context.run_config["model_name"], context.run_config["is_lora"], context.run_config["is_timm"])
    is_lora = context.run_config["is_lora"]
    ndarrays = get_weights(model, is_lora=is_lora)
    parameters = ndarrays_to_parameters(ndarrays)
    evaluate = get_evaluate_fn(context)

    # Set seed for reproducibility
    set_seed(context.run_config["random_seed"])

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        # min_fit_clients=1, # uncomment for centralized (1 client)
        initial_parameters=parameters,
        evaluate_fn=evaluate,
        on_fit_config_fn=get_on_fit_config(is_lora),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
