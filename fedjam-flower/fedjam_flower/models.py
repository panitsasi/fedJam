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


def get_model(model_name: str, is_lora: bool, is_timm: bool) -> nn.Module:
    # Initialize model parameters
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
            r=32,
            lora_alpha=32,
            target_modules=["query", "key", "value", "projection"], # Targeting query, value, and the output projection
            lora_dropout=0.15,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"Created LoRA model", flush=True)

    return model


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