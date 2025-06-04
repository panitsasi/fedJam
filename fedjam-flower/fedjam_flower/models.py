import torch
import timm
import torch.nn as nn
import torch
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model


def modify_first_conv_layer(pretrained_conv, in_channels):
    
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=pretrained_conv.out_channels,
        kernel_size=pretrained_conv.kernel_size,
        stride=pretrained_conv.stride,
        padding=pretrained_conv.padding,
        bias=pretrained_conv.bias is not None
    )

    with torch.no_grad():
        if pretrained_conv.in_channels >= 3 and in_channels >= 3:
            new_conv.weight[:, :3] = pretrained_conv.weight[:, :3]
            if in_channels > 3:
                new_conv.weight[:, 3:] = torch.randn_like(new_conv.weight[:, 3:]) * 0.01
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

    return new_conv

def get_model(model_name: str, is_lora: bool, is_timm: bool, in_channels: int = 3, num_classes: int = 4):

    if is_timm:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        if in_channels != 3:
            if hasattr(model, 'conv_stem'):
                print("[INFO] Patching conv_stem")
                model.conv_stem = modify_first_conv_layer(model.conv_stem, in_channels)

            elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
                print("[INFO] Patching patch_embed.proj")
                model.patch_embed.proj = modify_first_conv_layer(model.patch_embed.proj, in_channels)

            elif hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'conv1'):
                try:
                    conv_candidate = model.patch_embed.conv1.conv
                    if isinstance(conv_candidate, nn.Conv2d):
                        print("[INFO] Patching TinyViT conv at patch_embed.conv1.conv")
                        model.patch_embed.conv1.conv = modify_first_conv_layer(conv_candidate, in_channels)
                    else:
                        raise TypeError("Expected Conv2d at patch_embed.conv1.conv")
                except Exception as e:
                    print(f"[ERROR] Failed to patch TinyViT input conv: {e}")
                    raise

            elif hasattr(model, 'conv1'):
                print("[INFO] Patching conv1")
                model.conv1 = modify_first_conv_layer(model.conv1, in_channels)

            elif hasattr(model, 'features') and hasattr(model.features, 'conv0'):
                print("[INFO] Patching features.conv0")
                model.features.conv0 = modify_first_conv_layer(model.features.conv0, in_channels)

            else:
                print("[DEBUG] Attributes on model:")
                print(dir(model))
                print("[DEBUG] model.patch_embed =", getattr(model, "patch_embed", "Not Found"))
                raise NotImplementedError(f"Unknown input format for model: {model_name}")

    else:

        config = ViTConfig.from_pretrained(model_name)
        config.num_channels = in_channels
        config.num_labels = num_classes

        model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        if in_channels != 3:
            print("[INFO] Adjusting ViT patch embedding for custom input channels.")
            patch_embed = model.vit.embeddings.patch_embeddings.projection
            new_proj = modify_first_conv_layer(patch_embed, in_channels)
            model.vit.embeddings.patch_embeddings.projection = new_proj

            model.config.num_channels = in_channels
            print(f"[DEBUG] Patched ViT to use {in_channels} input channels")
            
        if is_lora:
            lora_config = LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=["query", "key", "value", "output.dense"], 
                lora_dropout=0.15,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("Created LoRA model", flush=True)

    return model






