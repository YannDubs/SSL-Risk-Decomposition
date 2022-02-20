from __future__ import annotations

import types
from typing import Callable, Optional, Union

import torch
from torch import nn

from utils.helpers import ImgPil2LabTensor, check_import, replace_module_prefix, rm_module
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
except ImportError:
    pass

DINO_PREPROCESSOR = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

SWAV_PREPROCESSOR = transforms.Compose([
        transforms.Resize(256), # interpolation bilinear instead of usual bicubic
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

SWAV_MODELS = {"resnet50": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
               "resnet50_ep100": "https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar",
                "resnet50_ep200": "https://dl.fbaipublicfiles.com/deepcluster/swav_200ep_pretrain.pth.tar",
                "resnet50_ep400": "https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar"}

VISSL_MODELS = {"barlow_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch",
                "mocov2_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/moco_v2_1node_lr.03_step_b32_zero_init/model_final_checkpoint_phase199.torch",
                "rotnet_rn50_in1k": "https://dl.fbaipublicfiles.com/vissl/model_zoo/rotnet_rn50_in1k_ep105_rotnet_8gpu_resnet_17_07_20.46bada9f/model_final_checkpoint_phase125.torch",
                "rotnet_rn50_in22k": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_in22k_ep105.torch",
                "simclr_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_1000ep_simclr_8node_resnet_16_07_20.afe428c7/model_final_checkpoint_phase999.torch",
                "simclr_rn50w2": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50w2_1000ep_simclr_8node_resnet_16_07_20.e1e3bbf0/model_final_checkpoint_phase999.torch",
                "simclr_rn101": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch",
                "jigsaw_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_jigsaw_in1k_goyal19.torch",
                "colorization_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_colorization_in1k_goyal19.torch",
                "clusterfit_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/converted_vissl_rn50_rotnet_16kclusters_in1k_ep105.torch",
                "npid_rn50": "https://dl.fbaipublicfiles.com/vissl/model_zoo/npid_1node_200ep_4kneg_npid_8gpu_resnet_23_07_20.9eb36512/model_final_checkpoint_phase199.torch"
                }

def available_models(mode: Optional[list[str]]=None) -> dict[str, list[str]]:
    """Return all available model names for given modes. If mode is None, return all."""
    available = dict()

    if  mode is None or "clip" in mode :
        check_import("clip", "clip in available_models")
        available["clip"] = list(clip.available_models())

    if  mode is None or "dino" in mode :
        with rm_module("utils"):
            available["dino"] = list(torch.hub.list("facebookresearch/dino:main"))

    if  mode is None or "swav" in mode :
        available["swav"].update(SWAV_MODELS)

    if  mode is None or "vissl" in mode :
        # more models available at `https://github.com/facebookresearch/swav` e.g. different epochs and batch-size
        available["vissl"] = VISSL_MODELS

    if  mode is None or "beit" in mode :
        # see https://huggingface.co/models
        available["beit"] = "check https://huggingface.co/models?sort=downloads&search=beit"

    if  mode is None or "torchvision" in mode :
        available["torchvision"] = torchvision.models.__dict__.keys()

    if  mode is None or "timm" in mode :
        available["timm"] = timm.list_models(pretrained=True)
        # there are a lot you can search using wild cards like  `timm.list_models('vit_*', pretrained=True)`

    return available

def load_representor(name : str, mode: str, model: str) -> Union[Callable, Callable]:
    """Return the encoder and the preprocessor."""
    if mode == "clip":
        check_import("clip", "mode=clip in load_representor")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model, device, jit=False)  # might have to try False
        encoder = model.visual.float()  # only keep the image model

        if hasattr(encoder, "proj"):
            # not clear form the code, but when doing linear probing they remove the projection
            # https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py#L233
            encoder.proj = None
        else:
            # as discussed here: https://github.com/openai/CLIP/issues/42 the projection head is proj of attn
            # set it manually to identity while ensuring that still linear layer:
            N = encoder.attnpool.c_proj.in_features
            identity = torch.nn.Linear(N, N)
            nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            encoder.attnpool.c_proj = identity

    elif mode == "dino":
        arch = model.split("_")[1]
        with rm_module("utils"):
            # dirty but if not there's collision of modules
            encoder = torch.hub.load("facebookresearch/dino:main", model)

        if "vit" in arch:
            encoder = VITDinoWrapper(encoder, arch)

        preprocess = DINO_PREPROCESSOR

    elif mode == "swav":
        encoder = tmodels.resnet.resnet50(pretrained=False, num_classes=0)
        state_dict = load_state_dict_from_url(
            url=SWAV_MODELS[model],
            map_location="cpu",
            file_name=name
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.fc = torch.nn.Identity()
        encoder.load_state_dict(state_dict, strict=False)
        preprocess = SWAV_PREPROCESSOR

    elif mode in ["timm", "sup_dino"]:
        check_import("timm", "mode=timm/sup_dino in load_representor")
        encoder = timm.create_model(model, pretrained=True, num_classes=0) # remove last classifier layer
        config = resolve_data_config({}, model=encoder)
        preprocess = create_transform(**config)

        if mode == "sup_dino":
            encoder = VITDinoWrapper(encoder, model, repo="timm")

    elif mode == "beit":
        check_import("transformers", "mode=beit in load_representor")
        extractor = transformers.BeitFeatureExtractor.from_pretrained(f"{model}")
        preprocess = lambda img: extractor(img, return_tensors="pt")['pixel_values'][0]
        model = transformers.BeitModel.from_pretrained(f"{model}")
        encoder = HuggingSelector(model, "pooler_output")

    elif mode == "vissl":
        preprocess = VISSL_PREPROCESSOR

        arch = model.split("_")[1]
        check_import("vissl", "mode=vissl in load_representor")
        state_dict = load_state_dict_from_url(url=VISSL_MODELS[model], map_location="cpu", file_name=name)
        if "classy_state_dict" in state_dict.keys():
            state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in state_dict.keys():
            state_dict = state_dict["model_state_dict"]

        is_vissl = (arch in ["rn50w2"]) or (model in ["colorization_rn50"])
        is_torchvision = not is_vissl

        if is_torchvision:
            state_dict = replace_module_prefix(state_dict, "_feature_blocks.")
            architectures = dict(rn50=tmodels.resnet.resnet50,
                                 rn101=tmodels.resnet.resnet101)
            architecture = architectures[arch]
            encoder = architecture(pretrained=False, num_classes=0)
            encoder.fc = torch.nn.Identity()

        else:
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
            elif model == "colorization_rn50":
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

    repo : {"dino", "timm"}
        What implementation the model uses.
    """
    def __init__(self, encoder : nn.Module, arch : str, repo : str="dino"):
        super().__init__()
        self.encoder = encoder
        self.arch = arch
        self.repo = repo
        self.set_repo(self.repo)

    def set_repo(self, repo):
        if repo == "dino":
            pass
        elif repo == "timm":
            self.encoder.get_intermediate_layers = types.MethodType(get_intermediate_layers, self.encoder)
        else:
            raise ValueError(f"Unknown repo={repo}.")

    def forward(self, x: torch.Tensor):
        arch = self.arch.lower()
        if ("vitb" in arch) or ("base" in arch):
            n_last_blocks = 1
            avgpool_patchtokens = True
        elif ("vits" in arch) or ("small" in arch):
            n_last_blocks = 4
            avgpool_patchtokens = False
        else:
            raise ValueError(f"Unknown arch={self.arch}")

        intermediate_output = self.encoder.get_intermediate_layers(x, n_last_blocks)
        out = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool_patchtokens:
            out = torch.cat((out.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            out = out.reshape(out.shape[0], -1)

        return out


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