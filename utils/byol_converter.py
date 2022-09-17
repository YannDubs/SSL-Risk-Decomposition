import re
import torch
import numpy as np
from . import resnet_byol

def convert_byol(ckpt, architecture="resnet50"):


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