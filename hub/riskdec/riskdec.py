import torch
from hub.augmentations import GaussianBlur, get_augmentations, get_normalization
import torchvision.models as tmodels
from torchvision import transforms
from .resnet_dim import update_dim_resnet_

__all__ = ["get_riskdec_models"]


RISKDEC_MODELS = {"dissl_resnet50_dNone_e100_m2_augLarge": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/dissl_resnet50_dNone_e100_m2_augLarge.torch",
                  "dissl_resnet50_dNone_e100_m2_augSmall": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/dissl_resnet50_dNone_e100_m2_augSmall.torch",
                  "dissl_resnet50_dNone_e100_m2_headTLinSLin": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/dissl_resnet50_dNone_e100_m2_headTLinSLin.torch",
                  "dissl_resnet50_dNone_e100_m2_headTMlpSMlp": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/dissl_resnet50_dNone_e100_m2_headTMlpSMlp.torch",
                  "dissl_resnet50_d4096_e100_m2": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/dissl_resnet50_d4096_e100_m2.torch",
                    "simclr_resnet50_dNone_e100_m2": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2.torch",
                    "simclr_resnet50_dNone_e100_m2_data010": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2_data010.torch",
                    "simclr_resnet50_dNone_e100_m2_data030": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2_data030.torch",
                    "simclr_resnet50_dNone_e100_m2_headTLinSLin": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2_headTLinSLin.torch",
                  "simclr_resnet50_dNone_e100_m2_headTMlpSLin": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2_headTMlpSLin.torch",
                "simclr_resnet50_dNone_e100_m2_headTMlpSMlp": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2_headTMlpSMlp.torch",
                "simclr_resnet50_dNone_e100_m2_headTNoneSNone": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_dNone_e100_m2_headTNoneSNone.torch",
                "simclr_resnet50_d8192_e100_m2": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/simclr_resnet50_d8192_e100_m2.torch",
                "speccl_resnet50_bs384_ep100": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/speccl_bs384_ep100.pth",
                  }


def get_riskdec_models(model, dim=None, is_speccl=False, is_train_transform=False):

    encoder = tmodels.resnet.resnet50(num_classes=0)
    ckpt_path = RISKDEC_MODELS[model]
    state_dict = torch.hub.load_state_dict_from_url(url=ckpt_path, map_location="cpu")

    if is_speccl:
        state_dict = state_dict["state_dict"]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()
                      if "backbone." in k and "proj_resnet" not in k and "fc." not in k}


    if dim is not None:
        update_dim_resnet_(encoder, z_dim=dim, bottleneck_channel=512, is_residual=True)

        # torchvision models do not have a resizer
        state_dict = {k.replace("resizer", "avgpool.0", 1) if k.startswith("resizer") else k: v
                      for k, v in state_dict.items()}

    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(state_dict, strict=True)

    if is_train_transform:
        if is_speccl:
            preprocessor = transforms.Compose([
                transforms.RandomResizedCrop(224,
                                             scale=(0.2, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    )], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.ToTensor(),
                get_normalization(mode="imagenet")])
        else:
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
                get_normalization(mode="imagenet")])
    else:
        if is_speccl:
            preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BICUBIC,
                                             normalize="imagenet", pre_resize=256)
        else:
            preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                             normalize="imagenet", pre_resize=256)

    return encoder, preprocessor
