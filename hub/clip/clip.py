import torch
import clip

__all__ = ["get_clip_models"]

def get_clip_models(model):
    encoder, preprocessor = clip.load(model, "cpu", jit=False)
    encoder = encoder.visual.float()  # only keep the image model

    # not clear from code and paper, but it seems that in CLIP they use the features before the projection which is not the default in the codebase
    # see https://github.com/openai/CLIP/issues/85#issuecomment-815581348
    if hasattr(encoder, "proj"):  # ViT
        # this will remove the the projection in the forward pass
        # https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py#L233
        encoder.proj = None
    else:  # Resnet
        # as discussed here: https://github.com/openai/CLIP/issues/42 the projection head is proj of attn
        # set it manually to identity while ensuring that still linear layer:
        N = encoder.attnpool.c_proj.in_features
        identity = torch.nn.Linear(N, N)
        torch.nn.init.zeros_(identity.bias)
        identity.weight.data.copy_(torch.eye(N))
        encoder.attnpool.c_proj = identity

    return encoder, preprocessor
