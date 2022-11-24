from torch.hub import load_state_dict_from_url
from torchvision import transforms
import torch.nn as nn


from hub.augmentations import get_augmentations
import torchvision.models as tmodels

__all__ = ["get_mmselfsup_models"]


MMSELFSUP_MODELS = {
    "relativeloc_rn50_70ep_mmselfsup": "https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220225-84784688.pth",
    "odc_rn50_440ep_mmselfsup": "https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k_20220225-a755d9c0.pth",
    "densecl_rn50_200ep_mmselfsup": "https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20220225-8c7808fe.pth",
    "simsiam_rn50_bs256_ep200_mmselfsup": "https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20220225-2f488143.pth",
    "simclr_rn50_bs256_ep200_mmselfsup": "https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20220428-46ef6bb9.pth",
    "simclr_rn50_bs4096_ep200_mmselfsup": "https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k_20220428-8c24b063.pth",
    "deepcluster_rn50_bs512_ep200_mmselfsup": "https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k_20220428-8c24b063.pth",
    }

def get_mmselfsup_models(name):

    encoder = tmodels.resnet.resnet50(num_classes=0)
    encoder.fc = nn.Identity()

    state_dict = load_state_dict_from_url(
        url=MMSELFSUP_MODELS[name],
        map_location="cpu",
        file_name=name
    )['state_dict']

    encoder.load_state_dict(state_dict, strict=True)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor