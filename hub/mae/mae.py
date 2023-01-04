import types

from torch.hub import load_state_dict_from_url
from torchvision import transforms

from hub.augmentations import get_augmentations, get_normalization
from hub.helpers import VITWrapper, get_intermediate_layers

import timm

__all__ = ["get_mae_models"]


MAE_MODELS = {"mae_vitB16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
              "mae_vitL16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
                "mae_vitH14": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
                "mae_vitB16_ft1k": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
              "mae_vitL16_ft1k": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth",
                "mae_vitH14_ft1k": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth",
              }

def get_mae_models(name, architecture, representation="cls", is_train_transform=False):

    state_dict = load_state_dict_from_url(
        url=MAE_MODELS[name],
        map_location="cpu",
        file_name=name
    )["model"]

    encoder = timm.create_model(architecture, pretrained=False, num_classes=0)
    encoder.load_state_dict(state_dict, strict=True)

    # makes timm compatible with VITWrapper
    encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
    encoder = VITWrapper(encoder, representation=representation)

    if is_train_transform:
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            get_normalization(mode="imagenet")])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor