"""
Builds a multimodal dataset from spectrogram images and corresponding KPIs
in a Hugging Face Dataset format, required for running with flower.

Requires:
- The multimodal dataset files (spectro_flower_multimodal) in base_path.

Usage:
- Modify Constants
- Modify the `base_path` variable to point to your dataset directory.
- Run the script to create a Hugging Face DatasetDict with 'train' and 'test' splits.
"""


from datasets import Dataset, DatasetDict, Features, Value, Image, Array2D, concatenate_datasets
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

# Constants
MAX_LEN = 19000
NUM_FEATURES = 5 # Number of features in the timeseries data (SNR, Latency, etc.)
CHUNK_SIZE = 5000  # To avoid Arrow overflow

def pad_or_truncate(arr, max_len=MAX_LEN):
    pad_size = max_len - len(arr)
    if pad_size > 0:
        return np.pad(arr, ((0, pad_size), (0, 0)), mode="constant")
    else:
        return arr[:max_len]

def load_split(split_dir):
    samples = []
    label_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

    for label_dir in tqdm(label_dirs, desc=f"Processing '{split_dir.name}' split"):
        label = label_dir.name
        image_dir = label_dir / "images"
        kpi_dir = label_dir / "kpis"

        image_paths = list(image_dir.glob("*.png"))

        for image_path in tqdm(image_paths, desc=f"  Loading class '{label}'", leave=False):
            stem = image_path.stem
            csv_path = kpi_dir / f"{stem}.csv"

            if not csv_path.exists():
                continue

            try:
                image = PILImage.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
                continue

            try:
                df = pd.read_csv(csv_path).drop(columns=["Time"], errors="ignore")
                df = df.fillna(0.0)
                timeseries = pad_or_truncate(df.to_numpy(dtype=np.float32))
            except Exception as e:
                print(f"Failed to load CSV {csv_path}: {e}")
                continue

            samples.append({
                "image": image,
                "timeseries": timeseries,
                "label": label
            })

    return samples

def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# === Load raw data ===
base_path = Path("/home/ioannis/Desktop/spectrograms/spectro_flower_multimodal")
train_data = load_split(base_path / "train")
test_data = load_split(base_path / "test")

# === Define dataset features ===
print("\n🔧 Defining dataset features...")
features = Features({
    "image": Image(),
    "timeseries": Array2D(dtype="float32", shape=(MAX_LEN, NUM_FEATURES)),
    "label": Value("string")
})

# === Create train Dataset ===
print(f"\n📦 Building 'train' split with {len(train_data)} samples...")
train_chunks = []
for chunk in tqdm(list(chunk_list(train_data, CHUNK_SIZE)), desc="  Creating train chunks"):
    ds = Dataset.from_list(chunk).cast(features)
    train_chunks.append(ds)
train_dataset = concatenate_datasets(train_chunks)

# === Create test Dataset ===
print(f"\n📦 Building 'test' split with {len(test_data)} samples...")
test_chunks = []
for chunk in tqdm(list(chunk_list(test_data, CHUNK_SIZE)), desc="  Creating test chunks"):
    ds = Dataset.from_list(chunk).cast(features)
    test_chunks.append(ds)
test_dataset = concatenate_datasets(test_chunks)

# === Combine into DatasetDict ===
print("\n🧩 Creating DatasetDict...")
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# === Save to disk ===
print("\n💾 Saving to disk...")
dataset.save_to_disk("/home/ioannis/Desktop/spectrograms/hf_dataset")

print("\n✅ Done! Dataset summary:")
print(dataset)
