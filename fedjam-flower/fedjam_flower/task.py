"""fedjam-flower: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_dataset
from fedjam_flower.custom_augment import CustomAugmenter


dataset_dict = None  # Cache FederatedDataset



def load_data(partition_id: int, num_partitions: int):
    # Only initialize `FederatedDataset` once
    print(f"Loading dataset {partition_id} / {num_partitions}", flush=True)
    global dataset_dict
    if dataset_dict is None:
        dataset_dict = load_dataset("imagefolder", data_dir="/home/iofeidis/workspace/spectrograms_old/spectro_flower_format")
    
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    partitioner1 = IidPartitioner(num_partitions=num_partitions)
    partitioner2 = IidPartitioner(num_partitions=num_partitions)
    partitioner1.dataset = train_dataset
    train_partition = partitioner1.load_partition(partition_id)

    partitioner2.dataset = test_dataset
    test_partition = partitioner2.load_partition(partition_id)

    print(f"Before custom", flush=True)
    pytorch_transforms = transforms.Compose([
        CustomAugmenter(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_partition = train_partition.with_transform(apply_transforms)
    test_partition = test_partition.with_transform(apply_transforms)
    trainloader = DataLoader(train_partition, batch_size=128, shuffle=True, num_workers=8)
    testloader = DataLoader(test_partition, batch_size=128, num_workers=8)
    return trainloader, testloader


def train(model, trainloader, epochs, device, learning_rate=5e-5):
    """Train the model on the training set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
