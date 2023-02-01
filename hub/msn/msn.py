import types
from functools import partial

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision import transforms

from hub.augmentations import GaussianBlur, get_augmentations, get_normalization
from hub.helpers import VITWrapper, get_intermediate_layers

import timm
from timm.models.vision_transformer import VisionTransformer

__all__ = ["get_msn_models"]


MSN_MODELS = {"msn_vits16_ep800": "https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar",
               "msn_vitb16_ep600": "https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar",
               "msn_vitl16_ep600": "https://dl.fbaipublicfiles.com/msn/vitl16_600ep.pth.tar",
                "msn_vitb4_ep300": "https://dl.fbaipublicfiles.com/msn/vitb4_300ep.pth.tar",
                "msn_vitl7_ep200": "https://dl.fbaipublicfiles.com/msn/vitl7_200ep.pth.tar",
               }

def get_msn_models(name, architecture, representation="cls", is_train_transform=False):

    state_dict = load_state_dict_from_url(
        url=MSN_MODELS[name],
        map_location="cpu",
        file_name=name
    )["target_encoder"]

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()
                  if "fc." not in k}

    if architecture == "vit_base_patch4_224":
        encoder = VisionTransformer(
            patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)

    elif architecture == "vit_large_patch7_224":
        encoder = VisionTransformer(
            patch_size=7, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)

    else:
        encoder = timm.create_model(architecture, num_classes=0)


    encoder.load_state_dict(state_dict, strict=True)

    # makes timm compatible with VITWrapper
    encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
    encoder = VITWrapper(encoder, representation=representation)

    if is_train_transform:
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            get_normalization(mode="imagenet")])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor