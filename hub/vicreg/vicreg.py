import torch
from hub.augmentations import get_augmentations
from torchvision import transforms

__all__ = ["get_vicreg_models"]

def get_vicreg_models(model):
    encoder = torch.hub.load('facebookresearch/vicreg:main', model)
    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)
    return encoder, preprocessor
