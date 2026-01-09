"""fedjam-flower: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import numpy as np

from datasets import load_dataset, load_from_disk
from fedjam_flower.custom_augment import CustomAugmenter

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

dataset_dict = None  # Cache FederatedDataset



def load_data(partition_id: int, num_partitions: int, data_dir: str = None,
              batch_size: int = 128, classes_per_partition: int = 4,
              is_multimodal: bool = False, modality: str = "both"):
    """
    Load data with modality selection.
    Args:
        modality: One of "both", "image", or "timeseries"
    """
    # Only initialize `FederatedDataset` once
    print(f"Loading dataset {partition_id + 1} / {num_partitions}", flush=True)
    global dataset_dict
    if dataset_dict is None:
        if is_multimodal:
            dataset_dict = load_dataset(data_dir)

            # Encode string labels to integers
            labels = sorted(set(dataset_dict["train"]["label"]))
            label2id = {lbl: i for i, lbl in enumerate(labels)}

            def encode_label(example):
                example["label"] = label2id[example["label"]]
                return example

            dataset_dict = dataset_dict.map(encode_label)
        else:
            dataset_dict = load_dataset("imagefolder", data_dir=data_dir)
    
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    # Using PathologicalPartitioner to specify number of classes per partition (IID)
    partitioner1 = PathologicalPartitioner(
        num_partitions=num_partitions, partition_by="label",
        num_classes_per_partition=classes_per_partition,
        class_assignment_mode="deterministic"
    )
    partitioner2 = PathologicalPartitioner(
        num_partitions=num_partitions, partition_by="label",
        num_classes_per_partition=classes_per_partition,
        class_assignment_mode="deterministic"
    )
    partitioner1.dataset = train_dataset
    train_partition = partitioner1.load_partition(partition_id)

    partitioner2.dataset = test_dataset
    test_partition = partitioner2.load_partition(partition_id)

    if is_multimodal:
        transform = transforms.Compose([
            CustomAugmenter(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        def collate_fn(batch):
            if modality in ["both", "image"]:
                images = [transform(example["image"]) for example in batch]
                images = torch.stack(images)
            else:
                images = None

            if modality in ["both", "timeseries"]:
                timeseries = [torch.tensor(example["timeseries"], dtype=torch.float32) for example in batch]
                timeseries = torch.stack(timeseries)
            else:
                timeseries = None

            labels = torch.tensor([example["label"] for example in batch])

            if modality == "both":
                return images, timeseries, labels
            elif modality == "image":
                return {"image": images, "label": labels}
            else:  # timeseries
                return {"timeseries": timeseries, "label": labels}

        trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    else:
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

        trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_partition, batch_size=batch_size)
    return trainloader, testloader


def train(model, trainloader, epochs, device, lr=5e-5, is_lora=False, is_timm=False,
          is_multimodal=False):
    """Train the model on the training set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    running_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            if is_multimodal:
                if isinstance(batch, (tuple, list)):  # Full multimodal data
                    images, timeseries, labels = batch
                    images = images.to(device) if images is not None else None
                    timeseries = timeseries.to(device) if timeseries is not None else None
                    labels = labels.to(device)
                    logits = model(images, timeseries)
                else:  # Single modality dict format
                    labels = batch["label"].to(device)
                    if "image" in batch:
                        logits = model(images=batch["image"].to(device))
                    else:
                        logits = model(timeseries=batch["timeseries"].to(device))
            else:
                images, labels = batch["image"], batch["label"]
                images, labels = images.to(device), labels.to(device)
                if is_timm:
                    logits = model(images)
                else:
                    logits = model(pixel_values=images).logits

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(model, testloader, device, is_lora=False, is_timm=False,
         is_multimodal=False):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            if is_multimodal:
                if isinstance(batch, (tuple, list)):  # Full multimodal data
                    images, timeseries, labels = batch
                    images = images.to(device) if images is not None else None
                    timeseries = timeseries.to(device) if timeseries is not None else None
                    labels = labels.to(device)
                    logits = model(images, timeseries)
                else:  # Single modality dict format
                    labels = batch["label"].to(device)
                    if "image" in batch:
                        logits = model(images=batch["image"].to(device))
                    else:
                        logits = model(timeseries=batch["timeseries"].to(device))
            else:
                images, labels = batch["image"], batch["label"]
                images, labels = images.to(device), labels.to(device)
                if is_timm:
                    logits = model(images)
                else:
                    logits = model(pixel_values=images).logits
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (torch.max(logits.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy


def get_weights(model, is_lora=False):
    if is_lora:
        state_dict = get_peft_model_state_dict(model)
    else:
        state_dict = model.state_dict()
    return [val.cpu().numpy() for _, val in state_dict.items()]


def set_weights(model, parameters, is_lora=False):
    if is_lora:
        peft_state_dict_keys = get_peft_model_state_dict(model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(model, state_dict)
    else:
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)



def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False