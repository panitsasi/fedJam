"""flowertune: A Flower / FlowerTune app."""

import math

import torch
from collections import OrderedDict
from flwr.common.typing import NDArrays
from flwr.common import Context, ndarrays_to_parameters
import sys
import timm
import torch.nn as nn


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_name: str) -> nn.Module:
    # Initialize model parameters
    model = timm.create_model(model_name, pretrained=True, num_classes=4)
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