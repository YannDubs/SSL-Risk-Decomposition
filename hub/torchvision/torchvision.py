import torch
import torchvision
from torchvision import transforms
from hub.augmentations import get_augmentations

__all__ = ["get_torchvision_models"]

def get_torchvision_models(model):

    encoder = torchvision.models.__dict__[model](weights="IMAGENET1K_V1")
    if "resnet" in model:
        encoder.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Feature extraction for model={model} not implemented.")
    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor