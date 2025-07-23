"""
Train a simple multimodal model using a Hugging Face dataset with images and timeseries data
for jamming classification.
Works with flower datasets for partitioning.
"""


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_from_disk
from tqdm import tqdm
import timm

# === Load Dataset in Hugging Face Format ===
dataset = load_from_disk("/home/username/Desktop/spectrograms/hf_dataset")

# Encode string labels to integers
labels = sorted(set(dataset["train"]["label"]))
label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for lbl, i in label2id.items()}

def encode_label(example):
    example["label"] = label2id[example["label"]]
    return example

dataset = dataset.map(encode_label)

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def collate_fn(batch):
    images = [transform(example["image"]) for example in batch]
    timeseries = [torch.tensor(example["timeseries"], dtype=torch.float32) for example in batch]
    labels = [example["label"] for example in batch]
    return (
        torch.stack(images),
        torch.stack(timeseries),
        torch.tensor(labels)
    )

#######
# use flower datasets for partitioning
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Using PathologicalPartitioner to specify number of classes per partition (IID)
partitioner1 = PathologicalPartitioner(
    num_partitions=12, partition_by="label",
    num_classes_per_partition=3,
    class_assignment_mode="deterministic" # Ensures same partitioning across runs
)

partitioner2 = PathologicalPartitioner(
    num_partitions=12, partition_by="label",
    num_classes_per_partition=3
)
partitioner1.dataset = train_dataset
train_partition = partitioner1.load_partition(0)

partitioner2.dataset = test_dataset
test_partition = partitioner2.load_partition(0)


train_loader = DataLoader(train_partition, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_partition, batch_size=64, shuffle=False, collate_fn=collate_fn)


# === Simple Multimodal Model (Mobilenet + GRU) ===
class MultiModalNet(nn.Module):
    def __init__(self, num_classes, ts_input_dim=5):
        super().__init__()
        self.vision_model = timm.create_model(
            "mobilenetv3_small_100.lamb_in1k", pretrained=True, num_classes=0
        )
        self.vision_output_dim = 1024 # Output dimension of MobileNetV3 small

        self.ts_hidden_dim = 128
        self.rnn = nn.GRU(input_size=ts_input_dim, hidden_size=self.ts_hidden_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(self.vision_output_dim + self.ts_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, timeseries):
        img_feat = self.vision_model(images)              # (B, 1024)
        _, ts_feat = self.rnn(timeseries)                 # (1, B, 128)
        ts_feat = ts_feat.squeeze(0)                      # (B, 128)
        combined = torch.cat([img_feat, ts_feat], dim=1)  # (B, 1152)
        return self.fc(combined)

# === Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalNet(num_classes=len(labels)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
for epoch in range(1):
    model.train()
    total_loss = 0.0
    for images, timeseries, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, timeseries, labels = images.to(device), timeseries.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images, timeseries)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Training Loss: {total_loss / len(train_loader):.4f}")

# === Evaluation ===
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, timeseries, labels in tqdm(test_loader, desc="Evaluating"):
        images, timeseries, labels = images.to(device), timeseries.to(device), labels.to(device)
        preds = model(images, timeseries).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\n✅ Test Accuracy: {correct / total:.2%}")
