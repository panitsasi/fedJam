import os
import torch
from tqdm import tqdm


NUM_CHANNELS_TO_KEEP = 7  # Change this to 5, 6, or 7 as needed

# ===== Source and Destination Paths =====
src_root = "/mnt/fedJam/spectro_flower_8_channels"
dst_root = f"/mnt/fedJam/spectro_flower_{NUM_CHANNELS_TO_KEEP}_channels"

os.makedirs(dst_root, exist_ok=True)

# Iterate over "train" and "test" folders
for split in ["train", "test"]:
    src_split_path = os.path.join(src_root, split)
    dst_split_path = os.path.join(dst_root, split)
    os.makedirs(dst_split_path, exist_ok=True)

    for class_name in os.listdir(src_split_path):
        src_class_path = os.path.join(src_split_path, class_name)
        dst_class_path = os.path.join(dst_split_path, class_name)
        os.makedirs(dst_class_path, exist_ok=True)

        for fname in tqdm(os.listdir(src_class_path), desc=f"{split}/{class_name}"):
            if fname.endswith(".pt"):
                src_file = os.path.join(src_class_path, fname)
                dst_file = os.path.join(dst_class_path, fname)

                data = torch.load(src_file)
                if isinstance(data, dict) and "image" in data:
                    data["image"] = data["image"][:NUM_CHANNELS_TO_KEEP]
                elif isinstance(data, torch.Tensor):
                    data = data[:NUM_CHANNELS_TO_KEEP]
                else:
                    print(f"Unexpected format in file: {src_file}")
                    continue

                torch.save(data, dst_file)
