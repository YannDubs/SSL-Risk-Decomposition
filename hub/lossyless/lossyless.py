import torch
from utils.helpers import rm_module

__all__ = ["get_lossyless_models"]

def get_lossyless_models(model):
    with rm_module("hub"):
        encoder, preprocessor = torch.hub.load("YannDubs/lossyless:main",
                                 model)
        encoder = encoder.float()
    return encoder, preprocessor
