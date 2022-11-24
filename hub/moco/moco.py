from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.nn as nn


from hub.augmentations import get_augmentations
import torchvision.models as tmodels

__all__ = ["get_moco_models"]


MOCO_MODELS = {
    "mocov1_rn50_ep200" : "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar",
    "mocov2_rn50_ep200" : "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar",
    "mocov2_rn50_ep800" : "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
}

def get_moco_models(name):

    encoder = tmodels.resnet.resnet50(num_classes=0)
    encoder.fc = nn.Identity()

    state_dict = load_state_dict_from_url(
        url=MOCO_MODELS[name],
        map_location="cpu",
        file_name=name
    )['state_dict']

    state_dict = {k.replace("module.encoder_q.", ""): v for k, v in state_dict.items()
                  if k.startswith('module.encoder_q') and "module.encoder_q.fc." not in k }

    encoder.load_state_dict(state_dict, strict=True)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor