import torch
from hub.augmentations import get_augmentations, get_normalization, GaussianBlur, Solarization
from torchvision import transforms
from hub.helpers import EncoderIndexing

__all__ = ["get_vicregl_models"]

def get_vicregl_models(model, is_train_transform=False):
    encoder = torch.hub.load('facebookresearch/vicregl:main', model)
    encoder = EncoderIndexing(encoder, index=1)

    if is_train_transform:

        preprocessor = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            get_normalization(mode="imagenet")
        ])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                         normalize="imagenet", pre_resize=256)

    return encoder, preprocessor
