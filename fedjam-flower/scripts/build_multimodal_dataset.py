"""
Multimodal Dataset Builder for Flower FL — Spectrogram + KPI (Hugging Face Format)

This script constructs a Hugging Face `DatasetDict` for multimodal learning tasks using:
- Timeseries KPI data (e.g., SNR, latency, throughput) from `.csv` files
- Spectrogram images from `.png` files

Each data sample consists of:
- `image`: RGB spectrogram image (PIL.Image)
- `timeseries`: Downsampled and fixed-length 2D float array of selected KPIs
- `label`: String label derived from directory name

Configurable Global Variables
-------------------------------
- `MAX_LEN`: Max number of timesteps after downsampling (e.g., 256)
- `DOWNSAMPLE_FACTOR`: Keep every N-th row from raw KPI sequences (e.g., 75 from ~18,000 rows → ~240)
- `KPI_COLUMNS`: List of KPI column names to use, e.g., `["SNR", "Latency"]`. Set to `None` to include all.
- `OUTPUT_DIR_ROOT`: Base output path for generated datasets

Output
--------
- The dataset is saved to a dynamically generated folder name based on the configuration:
  Format: `hf_dataset.down{DOWNSAMPLE_FACTOR}.len{MAX_LEN}.cols_{col_tag}`
  Example: `hf_dataset.down75.len256.cols_snr-lat`

Requirements
---------------
- Directory structure must be:
  base_path/
    ├── train/
    │    ├── class1/
    │    │     ├── images/*.png
    │    │     └── kpis/*.csv
    └── test/
         ├── class2/
         │     ├── images/*.png
         │     └── kpis/*.csv

- CSV files must share consistent columns across samples and may optionally include a "Time" column (which is ignored).

Usage
-------
- Edit the constants at the top of the script.
- Run the script:
    `python create_multimodal_dataset.py`
- Load with Hugging Face:
    `dataset = datasets.load_from_disk(<output_path>)`

"""


from datasets import Dataset, DatasetDict, Features, Value, Image, Array2D, concatenate_datasets
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

# === Dataset Configuration ===
MAX_LEN = 256                      # Max sequence length after downsampling (256, )
DOWNSAMPLE_FACTOR = 75             # Reduces ~18000 rows → ~240 (e.g., 50, 75)
KPI_COLUMNS = None                 # e.g. ["SNR", "Latency"] or None to use all
CHUNK_SIZE = 5000                  # For Arrow memory safety

# === Output directory (dynamically named) ===
OUTPUT_DIR_ROOT = "/home/iofeidis/workspace/hf_dataset_gain_30"

# === Global feature count placeholder (set dynamically) ===
NUM_FEATURES = None


def pad_or_truncate(arr, max_len=MAX_LEN):
    pad_size = max_len - len(arr)
    if pad_size > 0:
        return np.pad(arr, ((0, pad_size), (0, 0)), mode="constant")
    else:
        return arr[:max_len]


def load_split(split_dir):
    global NUM_FEATURES
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

                if KPI_COLUMNS is not None:
                    df = df[KPI_COLUMNS]  # Subset the KPI columns
                df = df.iloc[::DOWNSAMPLE_FACTOR].reset_index(drop=True)

                df = df.fillna(0.0)
                arr = df.to_numpy(dtype=np.float32)

                if NUM_FEATURES is None:
                    NUM_FEATURES = arr.shape[1]

                timeseries = pad_or_truncate(arr)
            except Exception as e:
                print(f"Failed to process {csv_path}: {e}")
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
base_path = Path("/home/ioannis/Desktop/spectrograms/spectro_flower_multimodal_gain_30")
train_data = load_split(base_path / "train")
test_data = load_split(base_path / "test")

# === Check NUM_FEATURES was correctly set
if NUM_FEATURES is None:
    raise ValueError("No timeseries data found to determine NUM_FEATURES.")

# === Define dataset features ===
print(f"\n🔧 Defining dataset features with shape=({MAX_LEN}, {NUM_FEATURES})...")
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
# === Build dynamic output folder name
col_tag = "all" if KPI_COLUMNS is None else "-".join([c[:3].lower() for c in KPI_COLUMNS])
output_name = f"hf_dataset_gain_30.down{DOWNSAMPLE_FACTOR}.len{MAX_LEN}.cols_{col_tag}"
output_path = Path(OUTPUT_DIR_ROOT).parent / output_name

print(f"\n💾 Saving to disk at: {output_path}")
dataset.save_to_disk(str(output_path))


print("\n✅ Done! Dataset summary:")
print(dataset)