import types

import torch
import clip
from torchvision import transforms

__all__ = ["get_clip_models"]

from hub.augmentations import get_normalization
from hub.helpers import VITWrapperCLIP, get_intermediate_layers_clip


def get_clip_models(model, is_train_transform=False, representation="cls", family="vit"):
    encoder, preprocessor = clip.load(model, "cpu", jit=False)
    encoder = encoder.visual.float()  # only keep the image model

    # not clear from code and paper, but it seems that in CLIP they use the features before the projection
    # which is not the default in the codebase
    # see https://github.com/openai/CLIP/issues/85#issuecomment-815581348

    if family == "resnet":
        # as discussed here: https://github.com/openai/CLIP/issues/42 the projection head is proj of attn
        # set it manually to identity while ensuring that still linear layer:

        # I'm really not sure whether they also do so for resnets they didn't say anything about it
        # but it seems a little complex to do so => makes me think they don't
        # I opened an issue: https://github.com/openai/CLIP/issues/211 but no answer yet
        N = encoder.attnpool.c_proj.in_features
        identity = torch.nn.Linear(N, N)
        torch.nn.init.zeros_(identity.bias)
        identity.weight.data.copy_(torch.eye(N))
        encoder.attnpool.c_proj = identity

    elif family == "vit":
        encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers_clip, encoder)
        encoder = VITWrapperCLIP(encoder, representation=representation)

    if is_train_transform:
        # the real augmentation in CLIP is actually the mapping to the text
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            get_normalization(mode="clip")])

    return encoder, preprocessor

