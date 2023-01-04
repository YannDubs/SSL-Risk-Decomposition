from __future__ import annotations
from torchvision import transforms

from .helpers import check_import
import random

try:
    from PIL import ImageOps, ImageFilter
except ImportError:
    pass

__all__ = ["get_augmentations"]

def get_normalization(mode="imagenet"):
    if mode == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif mode == "imagenet228":
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
    elif mode == "half":
        mean = [0.5000, 0.5000, 0.5000]
        std = [0.5000, 0.5000, 0.5000]
    elif mode == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        raise ValueError(f"Unknown normalize: {mode}")

    return transforms.Normalize(mean=mean, std=std)

def get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                      normalize="imagenet",
                      pre_resize=256,
                      final_size=224):
    return transforms.Compose([
        transforms.Resize(pre_resize, interpolation=interpolation),
        transforms.CenterCrop(final_size),
        transforms.ToTensor(),
        get_normalization(mode=normalize)
    ])

class GaussianBlur():
    def __init__(self, radius_min=0.1, radius_max=2.):
        check_import("PIL", "GaussianBlur")
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class Solarization():
    def __init__(self):
        check_import("PIL", "Solarization")

    def __call__(self, img):
        return ImageOps.solarize(img)