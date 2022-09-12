from __future__ import annotations

import ast
import os
import types
from pathlib import Path
from typing import Callable, Optional, Union
import logging

import torch
from torch import nn

from utils.helpers import ImgPil2LabTensor, check_import, replace_module_prefix, rm_module, download_url
from torchvision import transforms
import torchvision
import pytorch_lightning as pl
import torchvision.models as tmodels

from torch.hub import load_state_dict_from_url

try:
    import clip
except ImportError:
    pass

try:
    import vissl
except ImportError:
    pass

try:
    import transformers
except ImportError:
    pass

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.models.vision_transformer import _create_vision_transformer
except ImportError:
    pass


logger = logging.getLogger(__name__)
CURR_DIR = Path(__file__).parent

DINO_PREPROCESSOR = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

SIMCLR_PYTORCH_PREPROCESSOR = transforms.Compose([
        # slightly different than how trained where they use proportion of 0.875
        # but you already resized it to 256
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # DO NOT NORMALIZE
    ])

SWAV_PREPROCESSOR = transforms.Compose([
        transforms.Resize(256),  # interpolation bilinear instead of usual bicubic
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.228, 0.224, 0.225])  # strange that 228 instread of standard 229
    ])


VISSL_PREPROCESSOR = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


VIT_PREPROCESSOR = transforms.Compose([
        transforms.Resize(248, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5000, 0.5000, 0.5000], [0.5000, 0.5000, 0.5000])
    ])

TORCHVISION_PREPROCESSOR = VISSL_PREPROCESSOR
MOCO_PREPROCESSOR = VISSL_PREPROCESSOR
SIMSIAM_PREPROCESSOR = VISSL_PREPROCESSOR
MAE_PREPROCESSOR = VISSL_PREPROCESSOR
IBOT_PREPROCESSOR = DINO_PREPROCESSOR
MUGS_PREPROCESSOR = DINO_PREPROCESSOR  # not clear what they use in their paper
MSN_PREPROCESSOR = VISSL_PREPROCESSOR
RISKDEC_PREPROCESSOR = VISSL_PREPROCESSOR

SWAV_MODELS = {"resnet50": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
               "resnet50_ep100": "https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar",
                "resnet50_ep200": "https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar",
                "resnet50_ep400": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar",
                "resnet50_ep200_bs256": "https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_bs256_pretrain.pth.tar",
               "resnet50_ep400_bs256": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_bs256_pretrain.pth.tar",
                "resnet50w2": "https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar",
                "resnet50w4": "https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w4_400ep_pretrain.pth.tar",
                "resnet50w5": "https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w5_400ep_pretrain.pth.tar",
                "resnet50_ep400_2x224": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_2x224_pretrain.pth.tar"
               }

SIMSIAM_MODELS = {"simsiam_rn50_bs512_ep100": "https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar",
                  "simsiam_rn50_bs256_ep100": "https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar"
                  }

MOCO_MODELS = {"mocov3_rn50_ep100": "https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar",
                "mocov3_rn50_ep300": "https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar",
                "mocov3_rn50_ep1000": "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar",
                "mocov3_vitS_ep300": "https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar",
                "mocov3_vitB_ep300": "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar",
               }

MAE_MODELS = {"mae_vitB16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
              "mae_vitL16": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
                "mae_vitH14": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
                "mae_vitB16_ft1k": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
              "mae_vitL16_ft1k": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth",
                "mae_vitH14_ft1k": "https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth",
              }

VISSL_MODELS = {"barlow_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch",
                "barlow_rn50_ep300": "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_300ep_resnet50.torch",
                "mocov2_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch",
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
                "npidpp_rn50w2": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_4node_800ep_32kneg_cosine_rn50w2_npid++_4nodes_resnet_27_07_20.b7f4016c/model_final_checkpoint_phase799.torch",
                "pirl_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_pirl_jigsaw_4node_resnet_22_07_20.34377f59/model_final_checkpoint_phase799.torch",
                "pirl_rn50_ep200": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl_jigsaw_4node_200ep_pirl_jigsaw_4node_resnet_22_07_20.ffd17b75/model_final_checkpoint_phase199.torch",
                "pirl_rn50_headMLP": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_800ep_mlphead_gblur/model_final_checkpoint_phase799.torch",
                "pirl_rn50_ep200_headMLP": "https://dl.fbaipublicfiles.com/vissl/model_zoo/pirl/r50_200ep_mlp_gblur/model_final_checkpoint_phase199.torch",
                "dc2_rn50_ep400_2x224": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_2x224_pretrain.pth.tar",
                "dc2_rn50_ep400_2x224+4x96": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_400ep_pretrain.pth.tar",
                }

IBOT_MODELS = {"vits16": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
               "vitb16": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
               "vitb16_in22k": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_pt22k/checkpoint_student.pth"}

MUGS_MODELS = {"mugs_vits16_ep100": "https://drive.google.com/u/0/uc?id=1V2TyArzr7qY93UFglPBHRfYVyAMEfsHR&export=download&confirm=t&uuid=45b83a34-48c5-4791-965f-5bfdb03cb0c9",
               "mugs_vits16_ep300": "https://drive.google.com/u/0/uc?id=1ZAPQ0HiDZO5Uk7jVqF46H6VbGxunZkuf&export=download&confirm=t&uuid=8915c73e-1531-497f-aa8d-7d0aa53ba77d",
               "mugs_vits16_ep800": "https://drive.google.com/u/0/uc?id=1KMdhxxWc2JXAiFqVxX584V4RvlJgckGq&export=download&confirm=t&uuid=890403de-c80f-47a7-8487-5abf5b3f4044",
               "mugs_vitb16_ep400": "https://drive.google.com/u/0/uc?id=13NUziwToBXBmS7n7V_1Z5N6EG_7bcncW&export=download&confirm=t&uuid=86bc09fe-1494-4d92-b34e-6581aa5f5ca5",
               "mugs_vitl16_ep250": "https://drive.google.com/uc?export=download&id=1K76a-YnFYcmDXUZ_UlYVYFrWOt2a6733&confirm=t&uuid=4cfaa659-24a0-4694-bd88-aba03643fa86",
               }

MSN_MODELS = {"msn_vits16_ep800": "https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar",
               "msn_vitb16_ep600": "https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar",
               "msn_vitl16_ep600": "https://dl.fbaipublicfiles.com/msn/vitl16_600ep.pth.tar",
                "msn_vitb4_ep300": "https://dl.fbaipublicfiles.com/msn/vitb4_300ep.pth.tar",
                "msn_vitl7_ep200": "https://dl.fbaipublicfiles.com/msn/vitl7_200ep.pth.tar",
               }

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
                  }


# manually downloaded from https://github.com/AndrewAtanov/simclr-pytorch
SIMCLR_PYTORCH = {"simclr_rn50_bs512_ep100": CURR_DIR / "pretrained_models/resnet50_imagenet_bs512_epochs100.pth.tar"}

def available_models(mode: Optional[list[str]]=None) -> dict[str, list[str]]:
    """Return all available model names for given modes. If mode is None, return all."""
    available = dict()

    if mode is None or "clip" in mode:
        check_import("clip", "clip in available_models")
        available["clip"] = list(clip.available_models())

    if mode is None or "dino" in mode:
        with rm_module("utils"):
            available["dino"] = list(torch.hub.list("facebookresearch/dino:main"))

    if mode is None or "dissl" in mode:
        available["dissl"] = list(torch.hub.list("YannDubs/Invariant-Self-Supervised-Learning:main"))

    if mode is None or "riskdec" in mode:
        available["riskdec"] = list(RISKDEC_MODELS.keys())

    if mode is None or "swav" in mode:
        available["swav"] = list(SWAV_MODELS.keys())

    if mode is None or "simsiam" in mode:
        available["simsiam"] = list(SIMSIAM_MODELS.keys())

    if mode is None or "moco" in mode:
        available["moco"] = list(MOCO_MODELS.keys())

    if mode is None or "vissl" in mode:
        # more models available at `https://github.com/facebookresearch/swav` e.g. different epochs and batch-size
        available["vissl"] = list(VISSL_MODELS.keys())

    if mode is None or "beit" in mode:
        # see https://huggingface.co/models
        available["beit"] = "check https://huggingface.co/models?sort=downloads&search=beit"

    if mode is None or "mae" in mode:
        available["mae"] = list(MAE_MODELS.keys())

    if mode is None or "mae" in mode:
        available["mae"] = list(MAE_MODELS.keys())

    if mode is None or "mugs" in mode:
        available["mugs"] = list(MUGS_MODELS.keys())

    if mode is None or "msn" in mode:
        available["msn"] = list(MSN_MODELS.keys())

    if mode is None or "ibot" in mode:
        available["ibot"] = list(IBOT_MODELS.keys())

    if mode is None or "torchvision" in mode:
        available["torchvision"] = torchvision.models.__dict__.keys()

    if mode is None or "timm" in mode:
        available["timm"] = timm.list_models(pretrained=True)
        # there are a lot you can search using wild cards like  `timm.list_models('vit_*', pretrained=True)`

    if mode is None or "simclr-pytorch" in mode:
        available["simclr-pytorch"] = "manual download needed from https://github.com/AndrewAtanov/simclr-pytorch"

    return available

def load_representor(name : str, mode: str, model: str) -> Union[Callable, Callable]:
    """Return the encoder and the preprocessor."""
    if mode == "clip":
        check_import("clip", "mode=clip in load_representor")
        encoder, preprocess = clip.load(model, "cpu", jit=False)  # might have to try False
        encoder = encoder.visual.float()  # only keep the image model

        if hasattr(encoder, "proj"):  # ViT
            # not clear form the code, but when doing linear probing they remove the projection
            # https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py#L233
            encoder.proj = None
        else:  # Resnet
            # as discussed here: https://github.com/openai/CLIP/issues/42 the projection head is proj of attn
            # set it manually to identity while ensuring that still linear layer:
            N = encoder.attnpool.c_proj.in_features
            identity = torch.nn.Linear(N, N)
            nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            encoder.attnpool.c_proj = identity

    elif mode in ["dino", "dino_last", "dino_extractS", "dino_extractB"]:
        with rm_module("utils"):
            # dirty but if not there's collision of modules
            encoder = torch.hub.load("facebookresearch/dino:main", model)

        if mode in ["dino_extractS", "dino_extractB"]:
            # normal dino uses the clf of the last n layers for representation. For dino last just uses forward
            extract_mode = "small" if "extractS" in mode else "base"
            encoder = VITDinoWrapper(encoder, extract_mode=extract_mode, repo="dino")

        preprocess = DINO_PREPROCESSOR

    elif mode in ["ibot", "mugs", "mae", "msn"]:

        check_import("timm", f"mode={mode}  in load_representor")

        if mode == "ibot":
            MODELS = IBOT_MODELS
            preprocess = IBOT_PREPROCESSOR
            key = "state_dict"
        elif mode == "mugs":
            MODELS = MUGS_MODELS
            preprocess = MUGS_PREPROCESSOR
            key = "state_dict"
        elif mode == "mae":
            MODELS = MAE_MODELS
            preprocess = MAE_PREPROCESSOR
            key = "model"
        elif mode == "msn":
            MODELS = MSN_MODELS
            preprocess = MSN_PREPROCESSOR
            key = "target_encoder"
        else:
            raise ValueError(f"Unknown mode={mode}.")

        if "vits16" in model.lower():
            arch = 'vit_small_patch16_224'
        elif "vitb16" in model.lower():
            arch = 'vit_base_patch16_224'
        elif "vitl16" in model.lower():
            arch = 'vit_large_patch16_224'
        elif "vith14" in model.lower():
            arch = 'vit_huge_patch14_224'
        else:
            raise ValueError(f"Unknown model={model}.")

        encoder = timm.create_model(arch, pretrained=False, num_classes=0)

        if "drive" in MODELS[model]:
            ckpt_path = CURR_DIR / "pretrained_models"/f"{model}.pth"
            if not ckpt_path.exists():
                download_url(MODELS[model], CURR_DIR / "pretrained_models", filename=f"{model}.pth")
            state_dict = torch.load(ckpt_path)[key]
            msg = encoder.load_state_dict(state_dict, strict=False)  # will have relation blocks for MUGS => use strict false
            assert len(msg.missing_keys) == 0
            for m in msg.unexpected_keys:
                if "relation_blocks" not in m:
                    raise ValueError(f"Unexpected key {m}.")
        else:
            state_dict = load_state_dict_from_url(
                url=MODELS[model],
                map_location="cpu",
                file_name=name
            )[key]

            if mode == "ibot":
                state_dict = {k: v for k, v in state_dict.items()
                                if not k.startswith("head.")}

            elif mode == "msn":
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()
                              if "fc." not in k}

            encoder.load_state_dict(state_dict, strict=True)

        extract_mode = "small" if "vits" in model.lower() else "base"
        encoder = VITDinoWrapper(encoder, extract_mode=extract_mode, repo="timm")

    elif mode == "simclr-pytorch":
        encoder = tmodels.resnet.resnet50(pretrained=False, num_classes=0)
        encoder.fc = torch.nn.Identity()
        state_dict = torch.load(SIMCLR_PYTORCH[model], map_location="cpu")['state_dict']
        state_dict = {k.replace("convnet.", ""): v for k, v in state_dict.items()
                      if "convnet." in k and "fc." not in k}
        encoder.load_state_dict(state_dict)
        preprocess = SIMCLR_PYTORCH_PREPROCESSOR

    elif mode == "swav":
        arch = model.split("_")[0]

        if arch == "resnet50":
            encoder = tmodels.resnet.resnet50(pretrained=False, num_classes=0)

        elif arch == "resnet50w2":
            from utils.resnet50w import resnet50w2
            encoder = resnet50w2(output_dim=0, eval_mode=True)

        elif arch == "resnet50w4":
            from utils.resnet50w import resnet50w4
            encoder = resnet50w4(output_dim=0, eval_mode=True)

        elif arch == "resnet50w5":
            from utils.resnet50w import resnet50w5
            encoder = resnet50w5(output_dim=0, eval_mode=True)

        else:
            raise ValueError(f"Unknown arch={arch}")

        state_dict = load_state_dict_from_url(
            url=SWAV_MODELS[model],
            map_location="cpu",
            file_name=name
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        if arch == "resnet50":
            encoder.fc = torch.nn.Identity()

        encoder.load_state_dict(state_dict, strict=False)
        preprocess = SWAV_PREPROCESSOR

    elif mode in ["moco","simsiam"]:


        models = MOCO_MODELS if mode == "moco" else SIMSIAM_MODELS

        if "vit" in model:
            check_import("timm", f"mode={mode} with vit in load_representor")
            if "vitS" in model:
                # MOCO ViTs uses 12 heads instead of the standard 6
                size = "small"
                embedding_dim = 384
            else:
                size = "base"
                embedding_dim = 768
            encoder = _create_vision_transformer(f'vit_{size}_patch16_224', pretrained=False,
                                                 patch_size=16, embed_dim=embedding_dim, depth=12, num_heads=12,
                                                 num_classes=0, mlp_ratio=4, qkv_bias=True)
            linear_keyword = "head"
        elif "rn50" in model:
            encoder = tmodels.resnet.resnet50(pretrained=False, num_classes=0)
            linear_keyword = "fc"
        else:
            raise ValueError(f"Unknown model={model}.")

        state_dict = load_state_dict_from_url(
            url=models[model],
            map_location="cpu",
            file_name=name
        )['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith(f'module.base_encoder.{linear_keyword}'):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        setattr(encoder, linear_keyword, torch.nn.Identity())
        encoder.load_state_dict(state_dict, strict=True)
        preprocess = MOCO_PREPROCESSOR if mode == "moco" else SIMSIAM_PREPROCESSOR

    elif mode in ["timm", "sup_dino_extractS", "sup_dino_extractB"]:
        check_import("timm", "mode=timm/sup_dino in load_representor")
        encoder = timm.create_model(model, pretrained=True, num_classes=0)  # remove last classifier layer
        config = resolve_data_config({}, model=encoder)
        preprocess = create_transform(**config)

        if "sup_dino" in mode:
            extract_mode = "small" if "extractS" in mode else "base"
            encoder = VITDinoWrapper(encoder, extract_mode=extract_mode, repo="timm")

    elif mode == "beit":
        check_import("transformers", "mode=beit in load_representor")
        extractor = transformers.BeitFeatureExtractor.from_pretrained(f"{model}")
        preprocess = lambda img: extractor(img, return_tensors="pt")['pixel_values'][0]
        model = transformers.BeitModel.from_pretrained(f"{model}")
        encoder = HuggingSelector(model, "pooler_output")


    elif mode == "dissl":
        encoder = torch.hub.load("YannDubs/Invariant-Self-Supervised-Learning:main", model)
        preprocess = torch.hub.load("YannDubs/Invariant-Self-Supervised-Learning:main", "preprocessor")

    elif mode == "riskdec":
        preprocess = RISKDEC_PREPROCESSOR
        encoder = tmodels.resnet.resnet50(pretrained=False, num_classes=0)
        # TODO use metadata
        dim = ast.literal_eval(model.split("_d")[1].split("_")[0]) # take dim from name
        if dim is not None:
            from utils.hub import update_dim_resnet_ as _update_dim_resnet_
            _update_dim_resnet_(encoder, z_dim=dim, bottleneck_channel=512, is_residual=True)
        encoder.fc = torch.nn.Identity()
        ckpt_path = RISKDEC_MODELS[model]
        state_dict = torch.hub.load_state_dict_from_url(url=ckpt_path, map_location="cpu")
        # torchvision models do not have a resizer
        state_dict = {k.replace("resizer", "avgpool.0", 1) if k.startswith("resizer") else k: v
                      for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=True)

    elif mode == "vissl":
        preprocess = VISSL_PREPROCESSOR

        arch = model.split("_")[1]
        check_import("vissl", "mode=vissl in load_representor")
        state_dict = load_state_dict_from_url(url=VISSL_MODELS[model], map_location="cpu", file_name=name)
        if "classy_state_dict" in state_dict.keys():
            state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in state_dict.keys():
            state_dict = state_dict["model_state_dict"]

        is_vissl = (arch in ["rn50w2","rn50w4"]) or ("colorization" in  model)
        is_torchvision = not is_vissl

        if is_torchvision:
            state_dict = replace_module_prefix(state_dict, "_feature_blocks.")
            architectures = dict(rn50=tmodels.resnet.resnet50,
                                 rn101=tmodels.resnet.resnet101)
            architecture = architectures[arch]
            encoder = architecture(pretrained=False, num_classes=0)
            encoder.fc = torch.nn.Identity()

        else:
            # TODO remove dependencies on VISSL
            from vissl.config import AttrDict
            from vissl.models.trunks.resnext import ResNeXt
            # annoying but VISSL doesn't have defaults in the code (only hydra)
            dflt_rn_cfg = AttrDict({"INPUT_TYPE": "rgb",
                                    "ACTIVATION_CHECKPOINTING": {"USE_ACTIVATION_CHECKPOINTING": False,
                                                                 "NUM_ACTIVATION_CHECKPOINTING_SPLITS": 2},
                                    "TRUNK": {"RESNETS": {"DEPTH": 50, "WIDTH_MULTIPLIER": 1, "NORM": "BatchNorm",
                                                          "GROUPNORM_GROUPS": 32, "STANDARDIZE_CONVOLUTIONS": False,
                                                          "GROUPS": 1, "ZERO_INIT_RESIDUAL": False,
                                                          "WIDTH_PER_GROUP": 64, "LAYER4_STRIDE": 2}}})
            if arch == "rn50w2":
                dflt_rn_cfg.TRUNK.RESNETS.WIDTH_MULTIPLIER = 2

            elif arch == "rn50w3":
                dflt_rn_cfg.TRUNK.RESNETS.WIDTH_MULTIPLIER = 3

            elif arch == "rn50w4":
                dflt_rn_cfg.TRUNK.RESNETS.WIDTH_MULTIPLIER = 4

            elif arch == "rn50w5":
                dflt_rn_cfg.TRUNK.RESNETS.WIDTH_MULTIPLIER = 5

            elif "colorization_rn50" in model:
                dflt_rn_cfg.INPUT_TYPE = "lab"
                dflt_rn_cfg.TRUNK.RESNETS.LAYER4_STRIDE = 1
                # initialize here in case you don't have cv2 needed for `ImgPil2LabTensor`
                preprocess = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                ImgPil2LabTensor()
                            ])
            else:
                raise ValueError(f"Unknown model={model}")

            encoder = ResNeXt(dflt_rn_cfg, "resnet")
            encoder.feat_eval_mapping = None

        # issues with strict for jigsaw, colorization
        encoder.load_state_dict(state_dict, strict=False)

    elif mode == "torchvision":
        encoder = torchvision.models.__dict__[model](pretrained=True)
        if "resnet" in model:
            encoder.fc = torch.nn.Identity()
        else:
            raise ValueError(f"Feature extraction for model={model} not implemented.")
        preprocess = TORCHVISION_PREPROCESSOR

    else:
        raise ValueError(f"Unknown mode={mode}.")

    representor = LightningWrapper(encoder)
    return representor, preprocess


class VITDinoWrapper(nn.Module):
    """
    Follows https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf
    /eval_linear.py#L196
    VIT dino should use only the CLS of the last layer for large, but concat last 4 for small.

    Parameters
    ----------
    encoder : model
        VIT model.

    arch : {"*vits*", "*vitb*", "*vit_base*", "*vit_small*"}
        Architecture of the VIT model.

    extract_mode : {"small", "base"}
        Whether to use the standard extractor for the small vit ([CLS]) or for the base vit (last 4 layers).

    repo : {"dino", "timm"}
        What implementation the model uses.
    """
    def __init__(self, encoder : nn.Module, extract_mode: str, repo : str):
        super().__init__()
        self.encoder = encoder
        self.repo = repo
        self.set_repo(self.repo)

        if extract_mode == "base":
            self.n_last_blocks = 1
            self.avgpool_patchtokens = True
        elif extract_mode == "small":
            self.n_last_blocks = 4
            self.avgpool_patchtokens = False
        else:
            raise ValueError(f"Unknown extract_mode={extract_mode}")

    def set_repo(self, repo):
        if repo == "dino":
            pass
        elif repo == "timm":
            self.encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, self.encoder)
        else:
            raise ValueError(f"Unknown repo={repo}.")

    def forward(self, x: torch.Tensor):
        intermediate_output = self.encoder.get_intermediate_layers(x, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if self.avgpool_patchtokens:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)

        return output


def get_intermediate_layers(self, x, n=1):
    """Replicates https://github.com/facebookresearch/dino/blob/3247a0cacb4c0642270469e06facf96e895f56de
    /vision_transformer.py#L225 for TIMM ViT models."""

    ### prepare_tokens ###
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = self.pos_drop(x + self.pos_embed)
    ######################

    output = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if len(self.blocks) - i <= n:
            output.append(self.norm(x))

    return output

class LightningWrapper(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        out = self.encoder(x)
        if isinstance(out, (tuple, list)):
            # for vissl models like rn50w2 will return list
            assert len(out) == 1
            out = out[0]
        return out

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x).cpu(), y.cpu()

class HuggingSelector(nn.Module):
    """Wrapper around hugging face model to select correct output while enable `.cuda()` etc."""
    def __init__(self, model : nn.Module, select : str):
        super().__init__()
        self.model = model
        self.select = select

    def forward(self, x : torch.Tensor):
        #, output_hidden_states = True
        return self.model(x)[self.select]