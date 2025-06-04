import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import random

class CustomAugmenterForMultiChannel:
    def __init__(self, 
                 salt_pepper_prob=0.01,
                 gaussian_std=0.1,
                 cutout_h=60,
                 cutout_w=60,
                 cutout_fill=[0.5, 0.5, 0.5],
                 resize=(224, 224)):
        self.sp_prob = salt_pepper_prob
        self.gauss_std = gaussian_std
        self.cutout_h = cutout_h
        self.cutout_w = cutout_w
        self.cutout_fill = cutout_fill
        self.resize = resize
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.resize_tf = transforms.Resize(resize)

    def __call__(self, x):
        # x is a multi-channel torch.Tensor: [C, H, W]
        first3 = x[:3, :, :]
        remaining = x[3:, :, :] if x.shape[0] > 3 else None

        # Convert first 3 channels to PIL image
        img = self.to_pil(first3)

        # Apply RGB augmentations
        img = self.augment_rgb(img)

        # Convert back to tensor
        first3_aug = self.to_tensor(img)

        if remaining is not None:
            # No need to resize, just concatenate, because they have the same dimension.
            x_aug = torch.cat([first3_aug, remaining], dim=0)
        else:
            x_aug = first3_aug

        return x_aug

    def augment_rgb(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")

        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.5))
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        if random.random() < 0.2:
            img = ImageOps.flip(img)
        if random.random() < 0.3:
            img = img.rotate(random.choice([15, -15, 30, -30]))
        if random.random() < 0.7:
            color_jitter = transforms.ColorJitter(
                random.uniform(0, 0.4),
                random.uniform(0, 0.4),
                random.uniform(0, 0.4),
                random.uniform(0, 0.1)
            )
            img = color_jitter(img)

        img_np = np.array(img) / 255.0
        if random.random() < 0.3:
            img_np = self.add_salt_pepper(img_np, self.sp_prob)
        if random.random() < 0.3:
            img_np = self.add_gaussian_noise(img_np, self.gauss_std)
        if random.random() < 0.3:
            img_np = self.apply_cutout(img_np, self.cutout_h, self.cutout_w, self.cutout_fill)

        img = Image.fromarray((img_np * 255).astype(np.uint8))
        img = self.resize_tf(img)
        return img

    def add_salt_pepper(self, image_np, prob=0.01):
        output = np.copy(image_np)
        h, w, _ = output.shape
        num_salt = int(prob * h * w)
        num_pepper = int(prob * h * w)
        coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
        output[coords[0], coords[1], :] = 1.0
        coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
        output[coords[0], coords[1], :] = 0.0
        return output

    def add_gaussian_noise(self, image_np, std=0.1):
        return np.clip(image_np + np.random.normal(0, std, image_np.shape), 0.0, 1.0)

    def apply_cutout(self, image_np, max_cutout_h, max_cutout_w, fill=[0.5, 0.5, 0.5]):
        h, w, _ = image_np.shape
        cutout_h = random.randint(1, max_cutout_h)
        cutout_w = random.randint(1, max_cutout_w)
        x = random.randint(0, max(0, w - cutout_w))
        y = random.randint(0, max(0, h - cutout_h))
        image_np = image_np.copy()
        for c in range(3):
            image_np[y:y+cutout_h, x:x+cutout_w, c] = fill[c]
        return image_np
