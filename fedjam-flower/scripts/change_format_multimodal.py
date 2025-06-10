"""
Script to restructure spectrograms and KPIs (multimodal) into a flower-friendly format.

Usage:
Modify the `SRC_ROOT` and `DST_ROOT` variables to point to your source and destination directories.

Run the script with:
python change_format_multimodal.py
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

# ========== Configuration ==========
SPECTROGRAM_DIR = Path("/home/ioannis/Desktop/spectrograms/dataset_v3/spectrograms")
KPI_DIR = Path("/home/ioannis/Desktop/spectrograms/time_series_KPIs")
OUTPUT_DIR = Path("spectro_flower_multimodal")
SPLIT_RATIO = 0.8
RANDOM_SEED = 42
# ===================================


def extract_jamming_type(folder_name):
    """Extracts jamming type from folder name."""
    if folder_name.startswith("benign"):
        return "benign"
    elif "pulse" in folder_name:
        return "pulse"
    elif "single_tone" in folder_name:
        return "single_tone"
    elif "wideband" in folder_name:
        return "wideband"
    return None


def collect_image_csv_pairs():
    """Collects all valid (image, csv) pairs indexed by jamming type."""
    data = defaultdict(list)
    stats = defaultdict(lambda: {'seen': 0, 'used': 0, 'skipped': 0})

    for folder in os.listdir(SPECTROGRAM_DIR):
        jamming_type = extract_jamming_type(folder)
        if jamming_type is None:
            continue

        image_dir = SPECTROGRAM_DIR / folder
        kpi_dir = KPI_DIR / folder

        if not image_dir.exists() or not kpi_dir.exists():
            continue

        for file in os.listdir(image_dir):
            if not file.endswith(".png"):
                continue

            stats[jamming_type]['seen'] += 1
            original_path = image_dir / file

            try:
                base_id = file.split('_')[-1].replace(".png", "")
                csv_file = f"{base_id}.csv"
                csv_path = kpi_dir / csv_file

                if csv_path.exists():
                    prefix = image_dir.name.replace('/', '_')
                    new_img_name = f"{jamming_type}_{prefix}_{base_id}.png"
                    data[jamming_type].append((original_path, csv_path, new_img_name))
                    stats[jamming_type]['used'] += 1
                else:
                    stats[jamming_type]['skipped'] += 1
            except Exception as e:
                stats[jamming_type]['skipped'] += 1

    return data, stats


def make_class_dirs(base_dir, class_names):
    """Creates output directories for each class."""
    for split in ["train", "test"]:
        for cls in class_names:
            (base_dir / split / cls / "images").mkdir(parents=True, exist_ok=True)
            (base_dir / split / cls / "kpis").mkdir(parents=True, exist_ok=True)


def split_and_copy_dataset(data):
    """Splits dataset and copies files to new structure."""
    random.seed(RANDOM_SEED)
    copy_stats = defaultdict(lambda: {'train': 0, 'test': 0})

    for cls, pairs in data.items():
        random.shuffle(pairs)
        split_idx = int(len(pairs) * SPLIT_RATIO)
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]

        print(f"\n🔄 Splitting class '{cls}': Total={len(pairs)}, Train={len(train_pairs)}, Test={len(test_pairs)}")

        for split_name, split_data in zip(["train", "test"], [train_pairs, test_pairs]):
            for img_path, csv_path, new_img_name in split_data:
                img_out = OUTPUT_DIR / split_name / cls / "images" / new_img_name
                csv_out = OUTPUT_DIR / split_name / cls / "kpis" / new_img_name.replace(".png", ".csv")

                shutil.copy2(img_path, img_out)
                shutil.copy2(csv_path, csv_out)
                copy_stats[cls][split_name] += 1

    return copy_stats


def main():
    data, stats = collect_image_csv_pairs()

    if not data:
        print("❌ No valid image-CSV pairs found.")
        return

    make_class_dirs(OUTPUT_DIR, data.keys())
    copy_stats = split_and_copy_dataset(data)

    # === Report Summary ===
    print("\n📊 Dataset Build Summary:")
    total_seen = total_used = total_skipped = 0
    for cls, cls_stats in stats.items():
        seen = cls_stats['seen']
        used = cls_stats['used']
        skipped = cls_stats['skipped']
        print(f"  - {cls}: Seen={seen}, Used={used}, Skipped={skipped}")
        total_seen += seen
        total_used += used
        total_skipped += skipped

    print("\n📦 Per-class final split counts:")
    for cls, splits in copy_stats.items():
        print(f"  - {cls}: Train={splits['train']}, Test={splits['test']}")

    print(f"\n✅ Total images seen: {total_seen}")
    print(f"✅ Total valid pairs copied: {total_used}")
    print(f"⚠️  Total skipped due to missing CSVs: {total_skipped}")
    print(f"🧮 Total samples (used image–CSV pairs): {total_used}")
    print(f"📁 Output dataset created at: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

