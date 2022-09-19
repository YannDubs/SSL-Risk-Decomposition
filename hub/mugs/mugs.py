import types

from torch.hub import load_state_dict_from_url
from torchvision import transforms

from hub.augmentations import get_augmentations
from hub.helpers import VITWrapper, get_intermediate_layers

import timm

__all__ = ["get_mugs_models"]

MUGS_MODELS = {"mugs_vits16_ep100": "https://drive.google.com/u/0/uc?id=1V2TyArzr7qY93UFglPBHRfYVyAMEfsHR&export=download&confirm=t&uuid=45b83a34-48c5-4791-965f-5bfdb03cb0c9",
               "mugs_vits16_ep300": "https://drive.google.com/u/0/uc?id=1ZAPQ0HiDZO5Uk7jVqF46H6VbGxunZkuf&export=download&confirm=t&uuid=8915c73e-1531-497f-aa8d-7d0aa53ba77d",
               "mugs_vits16_ep800": "https://drive.google.com/u/0/uc?id=1KMdhxxWc2JXAiFqVxX584V4RvlJgckGq&export=download&confirm=t&uuid=890403de-c80f-47a7-8487-5abf5b3f4044",
               "mugs_vitb16_ep400": "https://drive.google.com/u/0/uc?id=13NUziwToBXBmS7n7V_1Z5N6EG_7bcncW&export=download&confirm=t&uuid=86bc09fe-1494-4d92-b34e-6581aa5f5ca5",
               "mugs_vitl16_ep250": "https://drive.google.com/uc?export=download&id=1K76a-YnFYcmDXUZ_UlYVYFrWOt2a6733&confirm=t&uuid=4cfaa659-24a0-4694-bd88-aba03643fa86",
               }

def get_mugs_models(name, architecture, representation="cls"):

    state_dict = load_state_dict_from_url(
        url=MUGS_MODELS[name],
        map_location="cpu",
        file_name=name
    )["state_dict"]

    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("relation_blocks.")}

    encoder = timm.create_model(architecture, pretrained=False, num_classes=0)
    encoder.load_state_dict(state_dict, strict=True)

    # makes timm compatible with VITWrapper
    encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
    encoder = VITWrapper(encoder, representation=representation)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor