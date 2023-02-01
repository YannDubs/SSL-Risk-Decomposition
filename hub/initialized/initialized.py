import torch
from torchvision import transforms

from hub.augmentations import get_augmentations
import torchvision

__all__ = ["get_initialized_models"]


def get_initialized_models(model, new_dim=None, **kwargs):

    if model in ["resnet50w2", "resnet50w4", "resnet50w5"]:
        import hub.swav.resnet50w as resnet50w
        encoder = resnet50w.__dict__[model](**kwargs)
    elif "resnet" in model:
        encoder = torchvision.models.__dict__[model](weights=None)
        if new_dim is not None:
            from hub.riskdec.resnet_dim import update_dim_resnet_
            update_dim_resnet_(encoder, z_dim=new_dim, bottleneck_channel=512, is_residual=None)
        encoder.fc = torch.nn.Identity()
    else:
        from hub.timm import get_timm_models
        encoder, _ = get_timm_models(model, **kwargs)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor
