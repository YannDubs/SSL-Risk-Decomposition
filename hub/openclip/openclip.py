import types

import open_clip

__all__ = ["get_openclip_models"]

from hub.helpers import VITWrapperCLIP, get_intermediate_layers_clip


def get_openclip_models(model, arch, is_train_transform=False, representation="cls", family="vit"):
    encoder, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(arch, pretrained=model)
    encoder = encoder.visual  # only keep the image model

    if family == "vit":
        encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers_clip, encoder)
        encoder = VITWrapperCLIP(encoder, representation=representation)

    preprocessor = preprocess_train if is_train_transform else preprocess_val

    return encoder, preprocessor

