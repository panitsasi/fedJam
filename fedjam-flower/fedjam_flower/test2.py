import os
import torch
from collections import Counter, defaultdict
import numpy as np
import random
from torch.utils.data import Subset
from torch.utils.data import Dataset


class Multi_Channel_Dataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None):
        self.samples = []
        self.class_to_idx = class_to_idx or self._find_classes(root_dir)

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





dataset = Multi_Channel_Dataset("/mnt/fedJam/spectro_flower_all_tensors/train")
partitioner = CustomLabelPartitioner(dataset, num_partitions=10, num_classes_per_partition=3)

for i in range(10):
    subset = partitioner.load_partition(i)
    labels = [dataset[idx]["label"] for idx in subset.indices]
    label_counts = Counter(labels)
    print(f"\n📦 Client {i} partition (total: {len(subset)} samples):")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} samples")


