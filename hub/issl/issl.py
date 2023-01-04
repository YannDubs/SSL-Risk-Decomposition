import torch

__all__ = ["get_issl_models"]

from hub.augmentations import GaussianBlur, get_normalization
from hub.helpers import rm_module
from torchvision import transforms

def get_issl_models(model, is_train_transform=False):
    with rm_module("hub"):
        encoder = torch.hub.load("YannDubs/Invariant-Self-Supervised-Learning:main",
                                 model)
        preprocessor = torch.hub.load("YannDubs/Invariant-Self-Supervised-Learning:main",
                                      "preprocessor")

    if is_train_transform:
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.14, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
                )], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            get_normalization(mode="imagenet")])

    return encoder, preprocessor
