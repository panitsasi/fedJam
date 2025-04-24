"""fedjam-flower: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    # global fds
    # if fds is None:
    #     partitioner = IidPartitioner(num_partitions=num_partitions)
    #     fds = FederatedDataset(
    #         dataset="uoft-cs/cifar10",
    #         partitioners={"train": partitioner},
    #     )
    # partition = fds.load_partition(partition_id)

    print("Loading data...", flush=True)

    # Directly from a directory
    dataset_dict = load_dataset("imagefolder", data_dir="/home/ioannis/Desktop/spectrograms/spectro_flower_format")
    # Note that what we just loaded is a DatasetDict, we need to choose a single split
    # and assign it to the partitioner.dataset
    # e.g. "train" split but that depends on the structure of your directory
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    print("Building partitioner...", flush=True)

    partitioner1 = IidPartitioner(num_partitions=num_partitions)
    partitioner2 = IidPartitioner(num_partitions=num_partitions)
    partitioner1.dataset = train_dataset
    train_partition = partitioner1.load_partition(partition_id)

    partitioner2.dataset = test_dataset
    test_partition = partitioner2.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    pytorch_transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # print(batch, flush=True)
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)
    trainloader = DataLoader(train_partition, batch_size=128, shuffle=True, num_workers=8)
    testloader = DataLoader(test_partition, batch_size=128, num_workers=8)
    print(f"Done with load_data(), flush=True")
    print(f"trainloader: {len(trainloader.dataset)}")
    print(f"testloader: {len(testloader.dataset)}")
    return trainloader, testloader


def train(model, trainloader, epochs, device):
    """Train the model on the training set."""
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(model, testloader, device):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
