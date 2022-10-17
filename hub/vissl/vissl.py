from __future__ import annotations

import torch
from hub.augmentations import get_augmentations
from hub.helpers import replace_module_prefix
import torchvision.models as tmodels
from torchvision import transforms

from torch.hub import load_state_dict_from_url
from .resnext_vissl import ResNeXt

__all__ = ["get_vissl_models"]

# NB: The barlow models are actually from VISSL but because they were saved with pickly they need vissl for loading
# to aviod dependencies on vissl I simply resaved it properly so that everyone can load without VISSL
VISSL_MODELS = {"barlow_rn50": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/barlow_rn50.torch",
                "barlow_rn50_ep300": "https://github.com/YannDubs/SSL-Risk-Decomposition/releases/download/v0.1/barlow_rn50_ep300.torch",
                "mocov2_rn50_vissl": "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch",
                "rotnet_rn50_in1k": "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch",
                "rotnet_rn50_in22k": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_in22k_ep105.torch",
                "simclr_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch",
                "simclr_rn50_ep200": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_200ep_simclr_8node_resnet_16_07_20.a816c0ef/model_final_checkpoint_phase199.torch",
                "simclr_rn50_ep400": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_400ep_simclr_8node_resnet_16_07_20.36b338ef/model_final_checkpoint_phase399.torch",
                "simclr_rn50_ep800": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch",
                "simclr_rn50_bs4096_ep100": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_100ep_simclr_8node_resnet_16_07_20.8edb093e/model_final_checkpoint_phase99.torch",
                "simclr_rn50w2": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/model_final_checkpoint_phase999.torch",
                "simclr_rn50w2_ep100": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w2_100ep_simclr_8node_resnet_16_07_20.05b37ec3/model_final_checkpoint_phase99.torch",
                "simclr_rn50w4": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w4_1000ep_bs32_16node_simclr_8node_resnet_28_07_20.9e20b0ae/model_final_checkpoint_phase999.torch",
                "simclr_rn101": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch",
                "simclr_rn101_ep100": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_100ep_simclr_8node_resnet_16_07_20.1ff6cb4b/model_final_checkpoint_phase99.torch",
                "jigsaw_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in1k_goyal19.torch",
                "jigsaw_rn50_in22k": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in22k_goyal19.torch",
                "colorization_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_in1k_goyal19.torch",
                "colorization_rn50_in22k": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_in22k_goyal19.torch",
                "clusterfit_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch",
                "npid_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_1node_200ep_4kneg_npid_8gpu_resnet_23_07_20.9eb36512/model_final_checkpoint_phase199.torch",
                "npidpp_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_pp/4node_800ep_32kneg_cosine_resnet_23_07_20.75432662/model_final_checkpoint_phase799.torch",
                "npidpp_rn50w2": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_pp/4node_800ep_32kneg_cosine_resnetw2_23_07_20.b7f4016c/model_final_checkpoint_phase799.torch",
                "pirl_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch",
                "pirl_rn50_ep200": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_200ep_pirl_jigsaw_4node_resnet_22_07_20.ffd17b75/model_final_checkpoint_phase199.torch",
                "pirl_rn50_headMLP": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_800ep_mlphead_gblur/model_final_checkpoint_phase799.torch",
                "pirl_rn50_ep200_headMLP": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_200ep_mlp_gblur/model_final_checkpoint_phase199.torch",
                "pirl_rn50w2": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/w2_400ep/model_final_checkpoint_phase399.torch",
                "pirl_rn50w2_headMLP": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50w2_400ep_mlphead_gblur/model_final_checkpoint_phase399.torch",
               }

def get_vissl_models(name, architecture= "resnet50", width_multiplier= 1):


    state_dict = load_state_dict_from_url(url=VISSL_MODELS[name],
                                          map_location="cpu",
                                          file_name=name)


    if "classy_state_dict" in state_dict.keys():
        state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
    elif "model_state_dict" in state_dict.keys():
        state_dict = state_dict["model_state_dict"]

    if width_multiplier == 1:
        state_dict = replace_module_prefix(state_dict, "_feature_blocks.")
        encoder = tmodels.resnet.__dict__[architecture](pretrained=False, num_classes=0)
        encoder.fc = torch.nn.Identity()

        if "jigsaw" in name:
            # jigsaw has the part which predicts the jigsaw puzzle in the head
            state_dict = {k: v for k, v in state_dict.items()
                          if ("fc1" not in k) and ("fc2" not in k) }

    else:
        depth = int(architecture.split("resnet")[-1])
        encoder = ResNeXt(width_multiplier=width_multiplier, depth=depth)

    encoder.load_state_dict(state_dict, strict=True)

    preprocessor = get_augmentations(interpolation=transforms.InterpolationMode.BILINEAR,
                                     normalize="imagenet", pre_resize=256)

    return encoder, preprocessor