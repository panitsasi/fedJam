import os
import shutil
import random
from pathlib import Path

# === CONFIG ===
SRC_ROOT = Path("dataset")  # original folder with _preprocessed subfolders
DST_ROOT = Path("spectro_flower_format")  # target destination
SPLIT_RATIO = 0.8  # train/test split

# Rename mapping from old folders to clean class names
CLASS_MAP = {
    "benign_preprocessed": "benign",
    "pulse_preprocessed": "pulse",
    "single_tone_preprocessed": "single_tone",
    "wideband_preprocessed": "wideband",
}

random.seed(42)  # reproducibility

# === Ensure directory structure ===
for split in ["train", "test"]:
    for class_name in CLASS_MAP.values():
        (DST_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)

# === Process Each Folder ===
for src_dir_name, class_label in CLASS_MAP.items():
    src_dir = SRC_ROOT / src_dir_name
    images = list(src_dir.glob("*.*"))
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    for split_name, file_list in [("train", train_files), ("test", test_files)]:
        target_dir = DST_ROOT / split_name / class_label
        for i, src_path in enumerate(file_list):
            ext = src_path.suffix  # keep original extension
            new_name = f"{class_label}_{src_path.stem}_{i}{ext}"
            dst_path = target_dir / new_name

            # Prevent overwriting
            if dst_path.exists():
                print(f"⚠️ Skipping (already exists): {dst_path.name}")
                continue

            shutil.copy(src_path, dst_path)

print(f"✅ All done! Dataset created at {DST_ROOT.resolve()}")
