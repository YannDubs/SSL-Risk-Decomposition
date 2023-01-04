import torch
from hub.augmentations import GaussianBlur, get_augmentations, get_normalization
from torchvision import transforms
import torchvision.models as tmodels
from torch.hub import load_state_dict_from_url

__all__ = ["get_simsiam_models"]

SIMSIAM_MODELS = {"simsiam_rn50_bs512_ep100": "https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar",
                  "simsiam_rn50_bs256_ep100": "https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar"
                  }

def get_simsiam_models(name, is_train_transform=False):

        encoder = tmodels.resnet.resnet50(pretrained=False, num_classes=0)
        encoder.fc = torch.nn.Identity()

        state_dict = load_state_dict_from_url(
            url=SIMSIAM_MODELS[name],
            map_location="cpu",
            file_name=name
        )['state_dict']

        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith(f'module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        encoder.load_state_dict(state_dict, strict=True)

        if is_train_transform:
            preprocessor = transforms.Compose([
                transforms.RandomResizedCrop(224,
                                             scale=(0.2, 1.0),
                                             interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    )], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                get_normalization(mode="imagenet")])
        else:
            preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                         normalize="imagenet", pre_resize=256)

        return encoder, preprocessor
