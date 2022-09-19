import torch
from torchvision import transforms

from hub.augmentations import get_augmentations
from utils.helpers import rm_module
from hub.dino.vit_wrapper import VITWrapper

__all__ = ["get_dino_models"]

def get_dino_models(model, representation="cls"):
    with rm_module("utils"):
        encoder = torch.hub.load("facebookresearch/dino:main", model)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                     normalize="imagenet", pre_resize=256)
    encoder = VITWrapper(encoder, representation=representation)

    return encoder, preprocessor