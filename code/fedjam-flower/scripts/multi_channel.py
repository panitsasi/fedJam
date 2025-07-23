import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import timm
from torchvision import transforms, datasets
from transformers import ViTImageProcessor

CONFIG = {
    'resnet50': {
        'model_name': 'resnet50',
        'num_classes': 4,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'root_dir': '/home/username/Desktop/spectrograms/dataset_v3_copy/spectro_flower_all_tensors/train',
        'class_map': {'benign': 0, 'pulse': 1, 'single_tone': 2, 'wideband': 3},
        'in_channels': 8,
    }
}


def get_resnet(model_name='resnet50', num_classes=4, pretrained=True):
    model_fn = getattr(models, model_name)
    model = model_fn(pretrained=pretrained)    
    if model.conv1.in_channels == 3:
        pretrained_weights = model.conv1.weight.clone()
    else:
        pretrained_weights = None    
        model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)    
        with torch.no_grad():
            if pretrained_weights is not None:
                model.conv1.weight[:, :3] = pretrained_weights
                if model.conv1.weight.shape[1] > 3:
                    nn.init.kaiming_normal_(model.conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')
            else:
                nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')    
                model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(model, trainloader, epochs, device, lr=5e-5, is_lora=False, is_timm=False):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        for batch_idx, batch in enumerate(trainloader):
            # Handle both dict-style and tuple-style batches
            if isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                images, labels = batch
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if is_timm:
                outputs = model(images)
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                print(f"Batch [{batch_idx + 1}/{len(trainloader)}] "
                      f"Loss: {batch_loss:.4f} "
                      f"Acc: {batch_acc:.2f}%")
        
        # Print epoch progress
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        print(f"\nEpoch Summary:")
        print(f"Training Loss: {epoch_loss:.4f}")
        print(f"Training Accuracy: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc/100.


def test(model, testloader, device, is_lora=False, is_timm=False):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print("\nEvaluating Model:")
    print("-" * 60)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            # Handle both dict-style and tuple-style batches
            if isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                images, labels = batch
            
            images, labels = images.to(device), labels.to(device)
            if is_timm:
                outputs = model(images)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 5 == 0:  # Print every 5 batches
                batch_loss = running_loss / (batch_idx + 1)
                batch_acc = 100. * correct / total
                print(f"Batch [{batch_idx + 1}/{len(testloader)}] "
                      f"Loss: {batch_loss:.4f} "
                      f"Acc: {batch_acc:.2f}%")
    
    test_loss = running_loss / len(testloader)
    test_acc = 100. * correct / total
    print(f"\nTest Summary:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_loss, test_acc/100.


class TensorDatasetFromFolders(Dataset):
    def __init__(self, root_dir, class_map, transform=None):
        self.samples = []
        self.transform = transform
        for class_name, label in class_map.items():
            folder_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(folder_path):
                if fname.endswith('.pt'):
                    self.samples.append((os.path.join(folder_path, fname), label))    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(path)
        if self.transform:
            tensor = self.transform(tensor)
        return {"image": tensor, "label": label}


def build_model(name, config):
    if name.startswith("resnet"):
        return get_resnet(model_name=name, num_classes=config['num_classes'], pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {name}")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config = CONFIG['resnet50']
    
    # Initialize model
    # model = build_model(model_config['model_name'], model_config).to(device)

    model = timm.create_model('timm/mobilenetv3_small_100.lamb_in1k', pretrained=True, in_chans=model_config['in_channels'],
                              num_classes=model_config['num_classes'])
    model = model.to(device)




    # 1) Load Dataset .pt
    dataset = TensorDatasetFromFolders(model_config['root_dir'], model_config['class_map'])


    # 2) Load dataset images
    # get model specific transforms (normalization, resize)
    # data_config = timm.data.resolve_model_data_config(model)
    # transform = timm.data.create_transform(**data_config, is_training=False)

    # dataset_path = "/home/username/Desktop/spectrograms/dataset_v3/spectro_flower_all/train"
    # dataset = datasets.ImageFolder(root=dataset_path, transform=transform)


    # Split dataset into train and validation sets
    labels = [label for _, label in dataset.samples]
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=0.2, 
        stratify=labels, 
        random_state=77
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=model_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=model_config['batch_size'])
    

    # Training
    train_loss, train_acc = train(
        model, 
        train_loader, 
        model_config['num_epochs'], 
        device
    )
    
    # Testing
    val_loss, val_acc = test(model, val_loader, device)
    
    print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()