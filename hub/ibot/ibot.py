import types

from torch.hub import load_state_dict_from_url
from torchvision import transforms

from hub.augmentations import get_augmentations
from hub.helpers import VITWrapper, get_intermediate_layers

import timm

__all__ = ["get_ibot_models"]

IBOT_MODELS = {"vits16": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
               "vitb16": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
               "vitb16_in22k": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint_student.pth",
                "vitl16": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth",
               }


def get_ibot_models(name, model, architecture, representation="cls"):
    state_dict = load_state_dict_from_url(
        url=IBOT_MODELS[model],
        map_location="cpu",
        file_name=name
    )["state_dict"]

    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("head.")}

    encoder = timm.create_model(architecture, pretrained=False, num_classes=0)
    encoder.load_state_dict(state_dict, strict=True)

    # makes timm compatible with VITWrapper
    encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
    encoder = VITWrapper(encoder, representation=representation)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor