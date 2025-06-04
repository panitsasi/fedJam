import random
import numpy as np
import torch
import sys
import math

# For reference, to remember the complete names of the models
MODEL_REGISTRY = {
    "densenet":           {"model_name": "densenet121", "is_timm": True},
    "efficientnet":       {"model_name": "efficientnet_b0", "is_timm": True},
    "resnet":             {"model_name": "resnet50", "is_timm": True},
    "mobilenet":          {"model_name": "mobilenetv2_100", "is_timm": True},
    "tinyvit":            {"model_name": "tiny_vit_5m_224", "is_timm": True},
    "vit":                {"model_name": "google/vit-base-patch16-224-in21k", "is_timm": False},
    "mobilevit":          {"model_name": "mobilevit_s", "is_timm": True}
}


def set_seed(seed: int = 42):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


def cosine_annealing(current_round: int, total_round: int, lrate_max: float = 0.001, lrate_min: float = 0.0):
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_size_in_mb(obj):
    """Calculate size of PyTorch objects in MB."""

    if isinstance(obj, dict):
        size_bytes = sum(
            sys.getsizeof(tensor.storage().data_ptr()) + tensor.nelement() * tensor.element_size()
            for tensor in obj.values()
            if isinstance(tensor, torch.Tensor)
        )

    elif isinstance(obj, torch.Tensor):
        size_bytes = sys.getsizeof(obj.storage().data_ptr()) + obj.nelement() * obj.element_size()
    else:
        raise TypeError("Object must be either a tensor or dictionary of tensors")
    
    return size_bytes / (1024 * 1024)  


def get_model_size_in_mb(model):
    """Calculate size of PyTorch model in MB."""

    param_size = 0
    buffer_size = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb