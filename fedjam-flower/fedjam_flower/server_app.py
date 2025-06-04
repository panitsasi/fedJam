from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from fedjam_flower.task import get_weights, set_weights, load_data, test
from fedjam_flower.helper_functions import set_seed
import torch
from fedjam_flower.models import get_model
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from flwr.server.strategy import FedAvg, FedProx, FedAdam

def get_evaluate_fn(context: Context):
    model_name = context.run_config["model_name"]
    num_clients = context.run_config["num_clients"]
    fft = context.run_config["fft"]
    gain = context.run_config["gain"]
    data_dir = context.run_config["data_dir"]
    batch_size = context.run_config["batch_size"]
    is_lora = context.run_config["is_lora"]
    is_timm = context.run_config["is_timm"]
    channels = context.run_config["channels"]
    classes = context.run_config["classes"]
    dataset_version = context.run_config["dataset_version"]
    num_classes_per_partition = context.run_config["num_classes_per_partition"]
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"runs/{dataset_version}/{model_name}_clients_{num_clients}_classes_per_partition_{num_classes_per_partition}"
    if fft != 0:
        filename += f"_fft_{fft}"
    if gain != 0:
        filename += f"_gain_{gain}"
    if is_lora:
        filename += "_lora"
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
            model = get_model(model_name, is_lora, is_timm, channels, classes).to(device).eval()

            # _, testloader = load_data(0, 1, data_dir=data_dir, batch_size=batch_size)

            #Ask Iason here
            # _, testloader = load_data(
            #     partition_id=0,
            #     num_partitions=1,
            #     data_dir=data_dir,
            #     batch_size=batch_size,
            #     num_classes_per_partition=num_classes_per_partition,
            #     use_multi_channel_dataset=use_multi_channel_dataset

            # )

            _, testloader = load_data(
                partition_id=0,
                num_partitions=1,
                data_dir=data_dir,
                batch_size=batch_size,
                num_classes_per_partition = context.run_config["classes"],
                use_multi_channel_dataset = context.run_config["use_multi_channel_dataset"]
            )

            set_weights(model, parameters, is_lora=is_lora)
            loss, accuracy = test(model, testloader, device, is_lora=is_lora, is_timm=is_timm)
            print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
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

    model = get_model(context.run_config["model_name"], context.run_config["is_lora"], 
                      context.run_config["is_timm"], context.run_config["channels"], context.run_config["classes"])

    ndarrays = get_weights(model, is_lora=context.run_config["is_lora"])
    parameters = ndarrays_to_parameters(ndarrays)
    evaluate = get_evaluate_fn(context)
    set_seed(context.run_config["seed"])

    strategy_name = context.run_config["strategy"]

    if strategy_name == "fed_avg":
        strategy = FedAvg(
            fraction_fit=context.run_config["fraction-fit"],
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=get_on_fit_config(context.run_config["is_lora"]),
        )
    elif strategy_name == "fed_prox":
        strategy = FedProx(
            fraction_fit=context.run_config["fraction-fit"],
            fraction_evaluate=1.0,
            min_available_clients=2,
            proximal_mu=0.1,
            initial_parameters=parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=get_on_fit_config(context.run_config["is_lora"]),
        )

    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")


    config = ServerConfig(num_rounds = context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)