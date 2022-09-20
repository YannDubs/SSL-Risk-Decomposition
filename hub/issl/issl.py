import torch

__all__ = ["get_issl_models"]

from hub.helpers import rm_module


def get_issl_models(model):
    with rm_module("hub"):
        encoder = torch.hub.load("YannDubs/Invariant-Self-Supervised-Learning:main",
                                 model)
        preprocessor = torch.hub.load("YannDubs/Invariant-Self-Supervised-Learning:main",
                                      "preprocessor")
    return encoder, preprocessor
