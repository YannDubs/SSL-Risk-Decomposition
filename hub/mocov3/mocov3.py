import types
from functools import partial

from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.nn as nn


from hub.augmentations import get_augmentations
from hub.helpers import VITWrapper, get_intermediate_layers
import torchvision.models as tmodels

from timm.models.vision_transformer import VisionTransformer
import timm

__all__ = ["get_mocov3_models"]



MOCOV3_MODELS = {"mocov3_rn50_ep100": "https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar",
                "mocov3_rn50_ep300": "https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar",
                "mocov3_rn50_ep1000": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
                "mocov3_vitS_ep300": "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
                "mocov3_vitB_ep300": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
               }

def get_mocov3_models(name, architecture, representation="cls"):

    if architecture == "vit_small_patch16_224":
        # MOCO ViTs uses 12 heads instead of the standard 6
        encoder = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)
        encoder.head = nn.Identity()

    elif "vit" in architecture:
        encoder = timm.create_model(architecture, pretrained=False, num_classes=0)
        encoder.head = nn.Identity()

    elif architecture == "resnet50":
        encoder = tmodels.resnet.resnet50(num_classes=0)
        encoder.fc = nn.Identity()

    else:
        raise ValueError(f"Unknown architecture={architecture}.")

    state_dict = load_state_dict_from_url(
        url=MOCOV3_MODELS[name],
        map_location="cpu",
        file_name=name
    )['state_dict']

    state_dict = {k.replace("module.base_encoder.", ""): v for k, v in state_dict.items()
                  if k.startswith('module.base_encoder') and "fc." not in k and "head." not in k}

    encoder.load_state_dict(state_dict, strict=True)

    if "vit" in architecture:
        # makes timm compatible with VITWrapper
        encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
        encoder = VITWrapper(encoder, representation=representation)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor