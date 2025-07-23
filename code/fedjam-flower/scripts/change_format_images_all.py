"""
Script to restructure spectrograms into a flower-friendly format.

Usage:
Modify the `SRC_ROOT` and `DST_ROOT` variables to point to your source and destination directories.

Run the script with:
python change_format_images_all.py
"""


import os
import shutil
import random
from pathlib import Path

# === CONFIG ===
SRC_ROOT = Path("/home/username/Desktop/spectrograms/dataset_v3/spectrograms")
DST_ROOT = Path("spectro_flower_all_new")  # Final output directory
SPLIT_RATIO = 0.8

random.seed(42)

def extract_class(label: str) -> str:
    if "benign" in label:
        return "benign"
    elif "pulse" in label:
        return "pulse"
    elif "single_tone" in label:
        return "single_tone"
    elif "wideband" in label:
        return "wideband"
    else:
        raise ValueError(f"Unknown class label in: {label}")

# === Gather by class ===
all_class_files = {
    "benign": [],
    "pulse": [],
    "single_tone": [],
    "wideband": [],
}

total_before = 0
total_after = {"train": 0, "test": 0}

for class_dir in SRC_ROOT.iterdir():
    if class_dir.is_dir():
        class_label = extract_class(class_dir.name)
        files = list(class_dir.glob("*.*"))
        all_class_files[class_label].extend(files)
        print(f"📂 Found {len(files)} files in '{class_dir.name}' → class '{class_label}'")
        total_before += len(files)

# === Create folders ===
for split in ["train", "test"]:
    for class_label in all_class_files:
        (DST_ROOT / split / class_label).mkdir(parents=True, exist_ok=True)

# === Copy and split ===
for class_label, files in all_class_files.items():
    random.shuffle(files)
    split_idx = int(len(files) * SPLIT_RATIO)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    print(f"\n🔄 Splitting class '{class_label}' → Train: {len(train_files)}, Test: {len(test_files)}")

    for split_name, file_list in [("train", train_files), ("test", test_files)]:
        target_dir = DST_ROOT / split_name / class_label
        for i, src_path in enumerate(file_list):
            ext = src_path.suffix
            prefix = src_path.parent.name.replace('/', '_')
            new_name = f"{class_label}_{prefix}_{src_path.stem}_{i}{ext}"
            dst_path = target_dir / new_name

            if dst_path.exists():
                print(f"⚠️ Skipping (already exists): {dst_path.name}")
                continue

            shutil.copy(src_path, dst_path)
            total_after[split_name] += 1

# === Summary ===
print("\n✅ Dataset restructuring complete!")
print(f"🔢 Total samples before split: {total_before}")
print(f"📊 Total after rearranging → Train: {total_after['train']}, Test: {total_after['test']}")
print(f"📁 Output directory: {DST_ROOT.resolve()}")

