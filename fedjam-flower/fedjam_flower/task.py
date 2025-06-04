from collections import OrderedDict
from torch.utils.data import Dataset
import os
import torch
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader
from collections import defaultdict
from torchvision import transforms
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from collections import Counter
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from augmenter.custom_augment import CustomAugmenter
from augmenter.custom_augment_multi_channel import CustomAugmenterForMultiChannel



dataset_dict = None  

def load_spectrogram_dataset(partition_id: int, num_partitions: int, data_dir: str = None, 
                       batch_size: int = 128, num_classes_per_partition: int = 4):

    print(f"Loading dataset {partition_id + 1} / {num_partitions}", flush=True)

    global dataset_dict
    if dataset_dict is None:
        dataset_dict = load_dataset("imagefolder", data_dir=data_dir)
    
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    partitioner1 = PathologicalPartitioner(
        num_partitions=num_partitions, partition_by="label", num_classes_per_partition=num_classes_per_partition
    )
    partitioner2 = PathologicalPartitioner(
        num_partitions=num_partitions, partition_by="label", num_classes_per_partition=num_classes_per_partition
    )
    partitioner1.dataset = train_dataset
    train_partition = partitioner1.load_partition(partition_id)

    partitioner2.dataset = test_dataset
    test_partition = partitioner2.load_partition(partition_id)


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
    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_partition, batch_size=batch_size, num_workers=8)
    return trainloader, testloader


class Multi_Channel_Dataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None, transform=None):
        self.samples = []
        self.class_to_idx = class_to_idx or self._find_classes(root_dir)
        self.transform = transform

        for class_name, class_idx in self.class_to_idx.items():
            class_folder = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_folder):
                if fname.endswith(".pt"):
                    path = os.path.join(class_folder, fname)
                    self.samples.append((path, class_idx))

    def _find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path)
        image = data if isinstance(data, torch.Tensor) else data["image"]
        
        if self.transform:
            image = self.transform(image)

        return {"image": image, "label": label}



class CustomLabelPartitioner:
    def __init__(self, dataset, num_partitions, num_classes_per_partition=None):
        self.dataset = dataset
        self.num_partitions = num_partitions
        self.num_classes_per_partition = num_classes_per_partition
        self.partitions = self._create_partitions()

    def _create_partitions(self):
        
        label_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset):
            label = sample["label"]
            label_to_indices[label].append(idx)

        # Shuffle indices for each class to avoid bias
        for label in label_to_indices:
            random.shuffle(label_to_indices[label])

        partitions = [[] for _ in range(self.num_partitions)]

        # Track how many clients have been assigned to each class
        class_to_clients = defaultdict(list)

        labels = sorted(label_to_indices.keys())
        num_classes = len(labels)

        # Assign classes to clients
        for client_id in range(self.num_partitions):
            assigned_classes = random.sample(labels, self.num_classes_per_partition)
            for cls in assigned_classes:
                class_to_clients[cls].append(client_id)

        # Distribute disjoint samples for each class across its assigned clients
        for cls, indices in label_to_indices.items():
            assigned_clients = class_to_clients[cls]
            if len(assigned_clients) == 0:
                continue  

            split_indices = np.array_split(indices, len(assigned_clients))
            for client_id, split in zip(assigned_clients, split_indices):
                partitions[client_id].extend(split.tolist())

        return partitions

    def load_partition(self, partition_id):
        return Subset(self.dataset, self.partitions[partition_id])


def load_spectrogram_KPI_dataset(partition_id: int, num_partitions: int, data_dir: str = None,
              batch_size: int = 128, num_classes_per_partition=None, channels: int = 3):
    
    print(f"Loading dataset {partition_id + 1} / {num_partitions}", flush=True)

    mean = [0.485, 0.456, 0.406] + [0.5] * max(0, channels - 3)
    std  = [0.229, 0.224, 0.225] + [0.25] * max(0, channels - 3)

    pytorch_transforms = transforms.Compose([
        CustomAugmenterForMultiChannel(),
        transforms.Normalize(mean=mean[:channels], std=std[:channels])
    ])

    train_dataset = Multi_Channel_Dataset(os.path.join(data_dir, "train"), transform=pytorch_transforms)
    test_dataset = Multi_Channel_Dataset(os.path.join(data_dir, "test"), transform=pytorch_transforms)

    train_partitioner = CustomLabelPartitioner(
        train_dataset, num_partitions=num_partitions, num_classes_per_partition=num_classes_per_partition, 
    )
    test_partitioner = CustomLabelPartitioner(
        test_dataset, num_partitions=num_partitions, num_classes_per_partition=num_classes_per_partition,
    )

    train_partition = train_partitioner.load_partition(partition_id)
    test_partition = test_partitioner.load_partition(partition_id)

    trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader


def train(model, trainloader, epochs, device, lr=5e-5, is_lora=False, is_timm=False):
    """Train the model on the training set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            if is_timm:
                outputs = model(images)
            else:
                outputs = model(pixel_values=images).logits
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(model, testloader, device, is_lora=False, is_timm=False):
    """Validate the model on the test set."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(device), labels.to(device)
            if is_timm:
                outputs = model(images)
            else:
                outputs = model(pixel_values=images).logits
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


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


def load_data(partition_id: int, num_partitions: int, data_dir: str = None,
              batch_size: int = 128, num_classes_per_partition=None, use_multi_channel_dataset: bool = True, channels: int = 3):
    
    if use_multi_channel_dataset:
        return load_spectrogram_KPI_dataset(
            partition_id=partition_id,
            num_partitions=num_partitions,
            data_dir=data_dir,
            batch_size=batch_size,
            num_classes_per_partition=num_classes_per_partition,
            channels=channels
        )
    else:
        return load_spectrogram_dataset(
            partition_id=partition_id,
            num_partitions=num_partitions,
            data_dir=data_dir,
            batch_size=batch_size, 
            num_classes_per_partition=num_classes_per_partition
        )