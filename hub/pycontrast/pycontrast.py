from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.nn as nn


from hub.augmentations import GaussianBlur, get_augmentations, get_normalization
import torchvision.models as tmodels

__all__ = ["get_pycontrast_models"]


PYCONTRAST_MODELS = {
    "infomin_rn50_200ep": "https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACgeWc3nQ4P4zUh48KWqhxBa/InfoMin_200.pth?dl=1",
    "infomin_rn50_800ep": "https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth?dl=1",
    }
# I'm not adding CMC as it uses half parameters
# not adding PIRL and MOCO and NPID as we have many of those already

def get_pycontrast_models(name, is_train_transform=False):

    encoder = tmodels.resnet.resnet50(num_classes=0)
    encoder.fc = nn.Identity()

    state_dict = load_state_dict_from_url(
        url=PYCONTRAST_MODELS[name],
        map_location="cpu",
        file_name=name
    )['model']

    state_dict = {k.replace("module.encoder.", ""): v for k, v in state_dict.items()
                  if k.startswith("module.encoder.")
                  }

    encoder.load_state_dict(state_dict, strict=True)

    if is_train_transform:
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.RandomGrayscale(p=0.2),
            # TODO should add jigsaw
            transforms.ToTensor(),
            get_normalization(mode="imagenet")])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor