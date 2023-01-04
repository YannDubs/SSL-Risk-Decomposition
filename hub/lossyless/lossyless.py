import torch

from hub.augmentations import get_normalization
from hub.helpers import rm_module
from torchvision import transforms

__all__ = ["get_lossyless_models"]

def get_lossyless_models(model, is_train_transform=False):
    with rm_module("hub"):
        encoder, preprocessor = torch.hub.load("YannDubs/lossyless:main",
                                 model)
        encoder = encoder.float()

    if is_train_transform:
        # the real augmentation in CLIP is actually the mapping to the text
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            get_normalization(mode="imagenet")])

    return encoder, preprocessor
