import torch
from torchvision import transforms

from hub.augmentations import get_augmentations
from hub.helpers import rm_module, VITWrapper

__all__ = ["get_dino_models"]

def get_dino_models(model, representation="cls"):
    with rm_module("utils"):
        encoder = torch.hub.load("facebookresearch/dino:main", model)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                     normalize="imagenet", pre_resize=256)
    encoder = VITWrapper(encoder, representation=representation)

    return encoder, preprocessor