from datasets import load_dataset
from flwr_datasets.partitioner import PathologicalPartitioner
from collections import Counter

# Parameters
data_dir = "/mnt/fedJam/spectro_flower_all"
num_partitions = 20
num_classes_per_partition = 2

# Load dataset once
dataset_dict = load_dataset("imagefolder", data_dir=data_dir)
train_dataset = dataset_dict["train"]

# Print label names (optional)
label_names = train_dataset.features["label"].names
print("Label Mapping:")
for i, name in enumerate(label_names):
    print(f"  {i}: {name}")

# 🔢 Count total samples per class in the full training set
print("\n📊 Total samples per class in the full training set:")
total_label_counts = Counter(train_dataset["label"])
for label_id, count in total_label_counts.items():
    print(f"  Label {label_id} ({label_names[label_id]}): {count} samples")

# 🔁 Loop through clients
for partition_id in range(num_partitions):
    print(f"\n📦 Client {partition_id + 1}/{num_partitions}:")

    partitioner = PathologicalPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        num_classes_per_partition=num_classes_per_partition
    )
    partitioner.dataset = train_dataset
    client_partition = partitioner.load_partition(partition_id)

    # Count labels per client
    label_counts = Counter(client_partition["label"])
    for label_id, count in label_counts.items():
        label_name = label_names[label_id]
        print(f"  Label {label_id} ({label_name}): {count} samples")
