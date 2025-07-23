import os
from datasets import load_from_disk, DatasetDict
from collections import Counter
import zipfile

# === Parameters ===
dataset_path = "/home/username/workspace/hf_dataset.down75.len256.cols_all"
samples_per_class = 50
seed = 42
classes = ['benign', 'pulse', 'single_tone', 'wideband']  # Your 4 classes
label_column = 'label'  # Adjust if needed
output_dir = "./hf_dataset_subset"
zip_filename = "hf_dataset_subset.zip"

# === Load dataset ===
dataset = load_from_disk(dataset_path)

# === Stratified subset function ===
def stratified_subset(ds, n_per_class, classes, label_col, seed=42):
    indices = []
    ds = ds.shuffle(seed=seed)
    for cls in classes:
        # If labels are strings:
        cls_indices = [i for i, label in enumerate(ds[label_col]) if label == cls]
        # If labels are integers matching index of class:
        if len(cls_indices) == 0:
            cls_idx = classes.index(cls)
            cls_indices = [i for i, label in enumerate(ds[label_col]) if label == cls_idx]
        selected = cls_indices[:n_per_class]
        indices.extend(selected)
    return ds.select(indices)

# === Create subsets for train and test ===
subset_train = stratified_subset(dataset['train'], samples_per_class, classes, label_column, seed)
subset_test = stratified_subset(dataset['test'], samples_per_class, classes, label_column, seed)

subset_dataset = DatasetDict({'train': subset_train, 'test': subset_test})

# === Save subset to disk ===
subset_dataset.save_to_disk(output_dir)

# === Compress subset directory into zip ===
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, output_dir)
            zipf.write(filepath, arcname)

print(f"Subset dataset saved to '{output_dir}' and compressed as '{zip_filename}'.")

# === Print overview and statistics ===
def print_overview_and_stats(dataset):
    print("\n=== Dataset overview ===")
    for split in dataset.keys():
        print(f"Split '{split}': {len(dataset[split])} samples")

    print("\n=== Class distribution per split ===")
    for split in dataset.keys():
        labels = dataset[split][label_column]
        label_counts = Counter(labels)
        print(f"\nSplit '{split}':")
        for cls in classes:
            # Count depending on label type (string or int)
            count = label_counts[cls] if cls in label_counts else label_counts.get(classes.index(cls), 0)
            print(f"  Class '{cls}': {count} samples")
print_overview_and_stats(subset_dataset)
