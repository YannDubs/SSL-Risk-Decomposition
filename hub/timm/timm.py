import types
from hub.helpers import VITWrapper, get_intermediate_layers

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

__all__ = ["get_timm_models"]

def get_timm_models(model, representation_vit="cls", pretrained=True):
    encoder = timm.create_model(model, pretrained=pretrained, num_classes=0)  # remove last classifier layer
    config = resolve_data_config({}, model=encoder)

    if "vit" in model.lower():
        encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
        encoder = VITWrapper(encoder, representation=representation_vit)

    preprocessor = create_transform(**config)

    return encoder, preprocessor