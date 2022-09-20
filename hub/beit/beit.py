import types
from torch.hub import load_state_dict_from_url
from torchvision import transforms


from hub.augmentations import get_augmentations
from hub.helpers import VITWrapper, get_intermediate_layers

import timm

__all__ = ["get_beit_models"]


BEIT_MODELS = {
"beit_vitB16_pt22k": "https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth",
"beit_vitL16_pt22k": "https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth",
"beitv2_vitB16_pt1k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth",
"beitv2_vitL16_pt1k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth",
}

def get_beit_models(name, architecture, representation="cls", normalize="imagenet"):

    state_dict = load_state_dict_from_url(
        url=BEIT_MODELS[name],
        map_location="cpu",
        file_name=name
    )['model']

    encoder = timm.create_model(architecture,
                                pretrained=False,
                                num_classes=0)
    encoder.load_state_dict(state_dict, strict=True)

    # makes timm compatible with VITWrapper
    encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, encoder)
    encoder = VITWrapper(encoder, representation=representation)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor