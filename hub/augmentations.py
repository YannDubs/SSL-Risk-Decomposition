from __future__ import annotations
from torchvision import transforms

__all__ = ["get_augmentations"]

def get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                      normalize="imagenet",
                      pre_resize=256):
    if normalize == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == "imagenet228":
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
    elif normalize == "half":
        mean = [0.5000, 0.5000, 0.5000]
        std = [0.5000, 0.5000, 0.5000]
    else:
        raise ValueError(f"Unknown normalize: {normalize}")

    return transforms.Compose([
        transforms.Resize(pre_resize, interpolation=interpolation),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
