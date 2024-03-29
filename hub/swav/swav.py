import torch
from hub.augmentations import GaussianBlur, get_augmentations, get_normalization
from torchvision import transforms
import torchvision.models as tmodels
from torch.hub import load_state_dict_from_url

__all__ = ["get_swav_models"]


SWAV_MODELS = {"resnet50": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
               "resnet50_ep100": "https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar",
                "resnet50_ep200": "https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar",
                "resnet50_ep400": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar",
                "resnet50_ep200_bs256": "https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar",
               "resnet50_ep400_bs256": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar",
                "resnet50w2": "https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar",
                "resnet50w4": "https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar",
                "resnet50w5": "https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar",
                "resnet50_ep400_2x224": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_2x224_pretrain.pth.tar",
                "dc2_rn50_ep400_2x224": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_2x224_pretrain.pth.tar",
                "dc2_rn50_ep400_2x160_4x96": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_pretrain.pth.tar",
                "dc2_rn50_ep800_2x224_6x96": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar",
               "selav2_rn50_ep400_2x224": "https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_2x224_pretrain.pth.tar",
                "selav2_rn50_ep400_2x160_4x96": "https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar",
               }

def get_swav_models(name, model, architecture, is_train_transform=False):

    if architecture == "resnet50":
        encoder = tmodels.resnet.resnet50(num_classes=0)
        encoder.fc = torch.nn.Identity()

    elif architecture == "resnet50w2":
        from hub.swav.resnet50w import resnet50w2
        encoder = resnet50w2()

    elif architecture == "resnet50w4":
        from hub.swav.resnet50w import resnet50w4
        encoder = resnet50w4()

    elif architecture == "resnet50w5":
        from hub.swav.resnet50w import resnet50w5
        encoder = resnet50w5()

    else:
        raise ValueError(f"Unknown architecture={architecture}")

    state_dict = load_state_dict_from_url(
        url=SWAV_MODELS[model],
        map_location="cpu",
        file_name=name
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("projection_head.") and not k.startswith("prototypes.")}


    encoder.load_state_dict(state_dict, strict=True)

    if is_train_transform:
        preprocessor = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.14, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
                )], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            get_normalization(mode="imagenet228")])
    else:
        preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                         normalize="imagenet228", pre_resize=256)

    return encoder, preprocessor
