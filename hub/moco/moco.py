from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.nn as nn


from hub.augmentations import GaussianBlur, get_augmentations, get_normalization
import torchvision.models as tmodels

__all__ = ["get_moco_models"]


MOCO_MODELS = {
    "mocov1_rn50_ep200" : "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar",
    "mocov2_rn50_ep200" : "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar",
    "mocov2_rn50_ep800" : "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
}

def get_moco_models(name, is_train_transform=False):

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

    if is_train_transform:
        if "mocov1" in name:
            preprocessor = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                get_normalization(mode="imagenet")])
        else:
            preprocessor = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                get_normalization(mode="imagenet")])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor