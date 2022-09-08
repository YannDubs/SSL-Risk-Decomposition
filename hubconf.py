
dependencies = [ "torch", "torchvision" ]

import torch
import torchvision


def preprocessor():
    """Preprocessor for all our pretrained models."""
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _replace_dict_prefix(d, prefix, replace_with = ""):
    return { k.replace(prefix, replace_with, 1) if k.startswith(prefix) else k: v for k,v in d.items()}

def _issl(objective, base, dim=None, sffx="", pretrained=True, **kwargs):
    resnet = torchvision.models.__dict__[base](**kwargs)
    if dim is not None:
        from utils.hub import update_dim_resnet_ as _update_dim_resnet_
        _update_dim_resnet_(resnet, z_dim=dim, bottleneck_channel=512, is_residual=True)
    resnet.fc = torch.nn.Identity()

    if pretrained:
        dir_path = "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1"
        ckpt_path = f"{dir_path}/{objective}_{base}_d{dim}{sffx}.torch"
        state_dict = torch.hub.load_state_dict_from_url(url=ckpt_path, map_location="cpu")
        # torchvision models do not have a resizer
        state_dict = _replace_dict_prefix(state_dict, "resizer", replace_with="avgpool.0")
        resnet.load_state_dict(state_dict, strict=True)

    return resnet

def dissl_resnet50_dNone_e100_m2_augLarge(pretrained=True, **kwargs):
    return _issl(objective="dissl", base="resnet50", dim=None, sffx="_e100_m2_augLarge", pretrained=pretrained, **kwargs)

def dissl_resnet50_dNone_e100_m2_augSmall(pretrained=True, **kwargs):
    return _issl(objective="dissl",base="resnet50", dim=None, sffx="_e100_m2_augSmall", pretrained=pretrained, **kwargs)

def dissl_resnet50_dNone_e100_m2_headTLinSLin(pretrained=True, **kwargs):
    return _issl(objective="dissl",base="resnet50", dim=None, sffx="_e100_m2_headTLinSLin", pretrained=pretrained, **kwargs)

def dissl_resnet50_dNone_e100_m2_headTMlpSMlp(pretrained=True, **kwargs):
    return _issl(objective="dissl",base="resnet50", dim=None, sffx="_e100_m2_headTMlpSMlp", pretrained=pretrained, **kwargs)

def dissl_resnet50_d4096_e100_m2(pretrained=True, **kwargs):
    return _issl(objective="dissl", base="resnet50", dim=4096, sffx="_e100_m2", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2(pretrained=True, **kwargs):
    return _issl(objective="simclr", base="resnet50", dim=None, sffx="_e100_m2", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2_data010(pretrained=True, **kwargs):
    return _issl(objective="simclr", base="resnet50", dim=None, sffx="_e100_m2_data010", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2_data030(pretrained=True, **kwargs):
    return _issl(objective="simclr",base="resnet50", dim=None, sffx="_e100_m2_data030", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2_headTLinSLin(pretrained=True, **kwargs):
    return _issl(objective="simclr", base="resnet50", dim=None, sffx="_e100_m2_headTLinSLin", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2_headTMlpSLin(pretrained=True, **kwargs):
    return _issl(objective="simclr", base="resnet50", dim=None, sffx="_e100_m2_headTMlpSLin", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2_headTMlpSMlp(pretrained=True, **kwargs):
    return _issl(objective="simclr", base="resnet50", dim=None, sffx="_e100_m2_headTMlpSMlp", pretrained=pretrained, **kwargs)

def simclr_resnet50_dNone_e100_m2_headTNoneSNone(pretrained=True, **kwargs):
    return _issl(objective="simclr", base="resnet50", dim=None, sffx="_e100_m2_headTNoneSNone", pretrained=pretrained, **kwargs)

