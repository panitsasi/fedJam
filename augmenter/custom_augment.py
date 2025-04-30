import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import random

class CustomAugmenter:
    def __init__(self, 
                 salt_pepper_prob=0.01,
                 gaussian_std=0.1,
                 cutout_h=60,
                 cutout_w=60,
                 cutout_fill=[0.5, 0.5, 0.5]):
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.sp_prob = salt_pepper_prob
        self.gauss_std = gaussian_std
        self.cutout_h = cutout_h
        self.cutout_w = cutout_w
        self.cutout_fill = cutout_fill

    def __call__(self, img):
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
        img = self.resize(img)
        return self.to_tensor(img)

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
