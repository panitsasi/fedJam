"""flowertune: A Flower / FlowerTune app."""

import math

import torch
from collections import OrderedDict
from flwr.common.typing import NDArrays
from flwr.common import Context, ndarrays_to_parameters
import sys
import timm
import torch.nn as nn

from transformers import ViTForImageClassification, ViTImageProcessor
from peft import (
    LoraConfig,
    get_peft_model
)

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_name: str, is_lora: bool, is_timm: bool, modality: str = "both") -> nn.Module:
    # Initialize model parameters
    if model_name == 'multimodal':
        # Use custom multimodal model with specified modality
        has_vision = modality in ["both", "image"]
        has_timeseries = modality in ["both", "timeseries"]
        if modality == "image":
            return MultiModalNet(num_classes=4, ts_input_dim=5, 
                               has_vision=True, 
                               has_timeseries=False)
        elif modality == "timeseries":
            return MultiModalNet(num_classes=4, ts_input_dim=5, 
                               has_vision=False, 
                               has_timeseries=True)
        else:  # both
            return MultiModalNet(num_classes=4, ts_input_dim=5, 
                               has_vision=True, 
                               has_timeseries=True)

    if is_timm:
        # Use timm model
        model = timm.create_model(model_name, pretrained=True, num_classes=4)
    else:
        # Use Hugging Face model
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=4,
            ignore_mismatched_sizes=True,
        )

    if is_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value", "projection"], # Targeting query, value, and the output projection
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )

        model = get_peft_model(model, lora_config)
        print(f"Created LoRA model", flush=True)
        model.print_trainable_parameters()

    return model


class MultiModalNet(nn.Module):
    def __init__(self, num_classes, ts_input_dim=5, has_vision=True, has_timeseries=True):
        super().__init__()
        self.has_vision = has_vision
        self.has_timeseries = has_timeseries
        
        # Vision components
        self.vision_output_dim = 1024 if has_vision else 0
        if has_vision:
            self.vision_model = timm.create_model(
                "mobilenetv3_small_100.lamb_in1k", pretrained=True, num_classes=0
            )
        
        # Timeseries components - fixed dimensions for consistency
        self.ts_input_dim = ts_input_dim
        self.ts_hidden_dim = 128 if has_timeseries else 0
        if has_timeseries:
            self.rnn = nn.GRU(
                input_size=self.ts_input_dim,
                hidden_size=self.ts_hidden_dim,
                batch_first=True
            )
        
        # Combined features dimension
        self.combined_dim = self.vision_output_dim + self.ts_hidden_dim
        
        # Ensure FC layers have consistent dimensions
        self.fc = nn.Sequential(
            nn.Linear(self.combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images=None, timeseries=None):
        batch_size = images.shape[0] if images is not None else timeseries.shape[0]
        device = next(self.parameters()).device
        
        # Handle vision features
        if self.has_vision:
            if images is None:
                img_feat = torch.zeros(batch_size, self.vision_output_dim, device=device)
            else:
                img_feat = self.vision_model(images)
        else:
            img_feat = torch.tensor([], device=device)
            
        # Handle timeseries features
        if self.has_timeseries:
            if timeseries is None:
                ts_feat = torch.zeros(batch_size, self.ts_hidden_dim, device=device)
            else:
                _, ts_feat = self.rnn(timeseries)
                ts_feat = ts_feat.squeeze(0)
        else:
            ts_feat = torch.tensor([], device=device)
            
        # Combine available features
        features = []
        if self.has_vision:
            features.append(img_feat)
        if self.has_timeseries:
            features.append(ts_feat)
            
        combined = torch.cat(features, dim=1)
        return self.fc(combined)

def get_size_in_mb(obj):
    """Calculate size of PyTorch objects in MB."""
    if isinstance(obj, dict):
        # For dictionary of tensors
        size_bytes = sum(
            sys.getsizeof(tensor.storage().data_ptr()) + tensor.nelement() * tensor.element_size()
            for tensor in obj.values()
            if isinstance(tensor, torch.Tensor)
        )

    elif isinstance(obj, torch.Tensor):
        # For single tensor
        size_bytes = sys.getsizeof(obj.storage().data_ptr()) + obj.nelement() * obj.element_size()
    else:
        raise TypeError("Object must be either a tensor or dictionary of tensors")
    
    return size_bytes / (1024 * 1024)  # Convert to MB


def get_model_size_in_mb(model):
    """Calculate size of PyTorch model in MB."""
    param_size = 0
    buffer_size = 0
    
    # Print debug information
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            # print(f"{name}: {param.nelement() * param.element_size() / (1024 * 1024):.2f} MB")
            param_size += param.nelement() * param.element_size()
    
    # Calculate buffer sizes (for BatchNorm running_mean, running_var, etc)
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb