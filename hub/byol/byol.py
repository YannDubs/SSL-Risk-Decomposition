from __future__ import annotations

from pathlib import Path
import logging


from hub.helpers import download_url_tmp
import re
import torch
import numpy as np
from . import resnet_byol
from torchvision import transforms

from torch.hub import load_state_dict_from_url
from hub.augmentations import get_augmentations
import dill

logger = logging.getLogger(__name__)
CURR_DIR = Path(__file__).parent


__all__ = ["get_byol_models"]

BYOL_MODELS = {
    "pretrain_res200x2": "https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res200x2.pkl",
    "pretrain_res50x1": "https://drive.google.com/uc?export=download&id=1nwaOpgmjpiOxJez7gUKQmYEiQIJe5Yss",
    "res50x1_batchsize_2048": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_2048.pkl",
    "res50x1_batchsize_1024": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_1024.pkl",
    "res50x1_batchsize_512": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_512.pkl",
    "res50x1_batchsize_256": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_256.pkl",
    "res50x1_batchsize_128": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_128.pkl",
    "res50x1_batchsize_64": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_batchsize_64.pkl",
    "res50x1_no_grayscale": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_no_grayscale.pkl",
    "res50x1_no_color": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_no_color.pkl",
    "res50x1_crop_and_blur_only": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_crop_and_blur_only.pkl",
    "res50x1_crop_only": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_crop_only.pkl",
    "res50x1_crop_and_color_only": "https://storage.googleapis.com/deepmind-byol/checkpoints/ablations/res50x1_crop_and_color_only.pkl",
}

def get_byol_models(name, model, architecture= "resnet50"):
    """Loads the BYOL encoder and preprocessor."""
    ckpt_path = CURR_DIR / "pretrained_models" / f"{name}.pth"

    if not ckpt_path.exists():
        is_already_converted = ".pkl" not in BYOL_MODELS[model]

        if is_already_converted:
            state_dict = load_state_dict_from_url(
                url=BYOL_MODELS[model],
                map_location="cpu",
                file_name=name
            )
        else:
            with download_url_tmp(BYOL_MODELS[model]) as tmp:
                with open(tmp, 'rb') as f:
                    ckpt = dill.load(f)
                    state_dict = convert_byol(ckpt, architecture=architecture)
                torch.save(state_dict, ckpt_path)
                logger.info(f"Saved model at {ckpt_path}")
    else:
        state_dict = torch.load(ckpt_path)

    encoder = resnet_byol.__dict__[architecture]()
    encoder.load_state_dict(state_dict, strict=True)
    encoder.fc = torch.nn.Identity()

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)


    return encoder, preprocessor




def convert_byol(ckpt, architecture="resnet50"):
    """
    Converts the tensorflow BYOL checkpoints to pytorch. This
    is copied from https://github.com/ajtejankar/byol-convert.
    """

    weights = ckpt['experiment_state'].online_params
    bn_states = ckpt['experiment_state'].online_state

    state_dict = {}
    for k, v in zip(weights.keys(), weights.values()):
        if 'projector' in k or 'predictor' in k:
            continue
        f_k = k
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/(conv|batchnorm)_([0-9])',
            lambda m: 'layer{}.{}.{}{}'.format(int(m[1])+1, int(m[2]), m[3], int(m[4])+1),
            f_k
        )
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/shortcut_(conv|batchnorm)',
            lambda m: 'layer{}.{}.{}'.format(int(m[1])+1, int(m[2]), 'downsample.' + m[3])\
                .replace('conv', '0').replace('batchnorm', '1'),
            f_k
        )
        f_k = re.sub(
            '.*initial_(conv|batchnorm)(_1)?',
            lambda m: '{}'.format(m[1] + '1'),
            f_k
        )
        f_k = f_k.replace('batchnorm', 'bn')
        f_k = f_k.replace('classifier', 'fc')
        for p_k, p_v in zip(v.keys(), v.values()):
            p_k = p_k.replace('w', '.weight')
            p_k = p_k.replace('b', '.bias')
            p_k = p_k.replace('offset', '.bias')
            p_k = p_k.replace('scale', '.weight')
            ff_k = f_k + p_k
            p_v = torch.from_numpy(p_v)
            # print(k, ff_k, p_v.shape)
            if 'conv' in ff_k or 'downsample.0' in ff_k:
                state_dict[ff_k] = p_v.permute(3, 2, 0, 1)
            elif 'bn' in ff_k or 'downsample.1' in ff_k:
                state_dict[ff_k] = p_v.squeeze()
            elif 'fc.weight' in ff_k:
                state_dict[ff_k] = p_v.permute(1, 0)
            else:
                state_dict[ff_k] = p_v

    for k, v in zip(bn_states.keys(), bn_states.values()):
        if 'projector' in k or 'predictor' in k:
            continue
        f_k = k
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/(conv|batchnorm)_([0-9])',
            lambda m: 'layer{}.{}.{}{}'.format(int(m[1])+1, int(m[2]), m[3], int(m[4])+1),
            f_k
        )
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/shortcut_(conv|batchnorm)',
            lambda m: 'layer{}.{}.{}'.format(int(m[1])+1, int(m[2]), 'downsample.' + m[3])\
                .replace('conv', '0').replace('batchnorm', '1'),
            f_k
        )
        f_k = re.sub(
            '.*initial_(conv|batchnorm)',
            lambda m: '{}'.format(m[1] + '1'),
            f_k
        )
        f_k = f_k.replace('batchnorm', 'bn')
        f_k = f_k.replace('/~/mean_ema', '.running_mean')
        f_k = f_k.replace('/~/var_ema', '.running_var')
        assert np.abs(v['average'] - v['hidden']).sum() == 0
        state_dict[f_k] = torch.from_numpy(v['average']).squeeze()

    pt_state_dict = resnet_byol.__dict__[architecture]().state_dict()
    pt_state_dict = {k: v for k, v in pt_state_dict.items() if 'tracked' not in k}

    assert len(pt_state_dict) == len(state_dict)
    for (k, v), (pk, pv) in zip(sorted(list(state_dict.items())), sorted(list(pt_state_dict.items()))):
        assert k == pk
        assert v.shape == pv.shape

    return state_dict