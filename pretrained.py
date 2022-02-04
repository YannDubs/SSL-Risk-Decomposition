from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import nn

from utils.helpers import check_import, replace_module_prefix, rm_module
from torchvision import transforms
import torchvision
import pytorch_lightning as pl
from torchvision.models.resnet import resnet50, wide_resnet50_2, resnet101

from torch.hub import load_state_dict_from_url

try:
    import clip
except ImportError:
    pass

try:
    import transformers
except ImportError:
    pass

try:
    import vissl
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
                "simclr_rn101": "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn101_1000ep_simclr_8node_resnet_16_07_20.35063cea/model_final_checkpoint_phase999.torch"}


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

    if  mode is None or "swav" in mode :
        # more models available at `https://github.com/facebookresearch/swav` e.g. different epochs and batch-size
        available["vissl"] = VISSL_MODELS

    if  mode is None or "beit" in mode :
        # see https://huggingface.co/models
        available["beit"] = "check https://huggingface.co/models?sort=downloads&search=beit"

    if  mode is None or "torchvision" in mode :
        available["torchvision"] = torchvision.models.__dict__.keys()

    if  mode is None or "vit" in mode :
        available["vit"] = "check https://huggingface.co/models?sort=downloads&search=vit"

    return available

def load_representor(mode: str, model: str) -> Union[Callable, Callable]:
    """Return the encoder and the preprocessor."""


    if mode == "clip":
        check_import("clip", "mode=clip in load_representor")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model, device, jit=False)  # might have to try False
        encoder = model.visual.float()  # only keep the image model

    elif mode == "dino":
        arch = model.split("_")[1]
        with rm_module("utils"):
            # dirty but if not there's collision of modules
            encoder = torch.hub.load("facebookresearch/dino:main", model)

        if "vit" in arch:
            encoder = VITDinoWrapper(encoder, arch)

        preprocess = DINO_PREPROCESSOR

    elif mode == "swav":
        encoder = resnet50(pretrained=False, num_classes=0)
        state_dict = load_state_dict_from_url(
            url=SWAV_MODELS[model],
            map_location="cpu",
            file_name=model
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.fc = torch.nn.Identity()
        encoder.load_state_dict(state_dict)
        preprocess = SWAV_PREPROCESSOR

    elif mode == "beit":
        check_import("transformers", "mode=beit in load_representor")
        extractor = transformers.BeitFeatureExtractor.from_pretrained(f"{model}")
        preprocess = lambda img : extractor(img, return_tensors="pt")['pixel_values'][0]
        model = transformers.BeitModel.from_pretrained(f"{model}")
        encoder = HuggingSelector(model, "pooler_output")

    elif mode == "vit":
        check_import("transformers", "mode=vit in load_representor")
        extractor = transformers.ViTFeatureExtractor.from_pretrained(f"{model}")
        preprocess = lambda img : extractor(img, return_tensors="pt")['pixel_values'][0]
        model = transformers.ViTForImageClassification.from_pretrained(f"{model}")
        encoder = HuggingSelector(model, "logits")

    elif mode == "vissl":
        arch = model.split("_")[1]
        check_import("vissl", "mode=vissl in load_representor")
        state_dict = load_state_dict_from_url(url=VISSL_MODELS[model], map_location="cpu", file_name=model)
        if "classy_state_dict" in state_dict.keys():
            state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in state_dict.keys():
            state_dict = state_dict["model_state_dict"]

        if arch in "rn50":
            state_dict = replace_module_prefix(state_dict, "_feature_blocks.")
            encoder = resnet50(pretrained=False, num_classes=0)
            encoder.fc = torch.nn.Identity()
        elif arch == "rn101":
            state_dict = replace_module_prefix(state_dict, "_feature_blocks.")
            encoder = resnet101(pretrained=False, num_classes=0)
            encoder.fc = torch.nn.Identity()
        elif arch == "rn50w2":
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
            dflt_rn_cfg.TRUNK.RESNETS.WIDTH_MULTIPLIER = 2
            encoder = ResNeXt(dflt_rn_cfg, "resnet")
            encoder.feat_eval_mapping = None
        else:
            raise ValueError(f"Unknown arch={arch}.")

        encoder.load_state_dict(state_dict)
        preprocess = VISSL_PREPROCESSOR

    elif mode == "torchvision":
        encoder = torchvision.models.__dict__[model](pretrained=True)
        preprocess = TORCHVISION_PREPROCESSOR

    else:
        raise ValueError(f"Unknown mode={mode}.")

    representor = LightningWrapper(encoder)
    return representor, preprocess

class HuggingSelector(nn.Module):
    """Wrapper around hugging face model to select correct output while enable `.cuda()` etc."""
    def __init__(self, model : nn.Module, select : str):
        super().__init__()
        self.model = model
        self.select = select

    def forward(self, x : torch.Tensor):
        #, output_hidden_states = True
        return self.model(x)[self.select]

class VITDinoWrapper(nn.Module):
    def __init__(self, encoder, arch):
        super().__init__()
        self.encoder = encoder
        self.arch = arch

    def forward(self, x: torch.Tensor):
        """
        Follows https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/eval_linear.py#L196
        VIT dino should use only the CLS of the last layer for large, but concat last 4 for small.
        """
        if "vitb" in self.arch.lower():
            n_last_blocks = 1
            avgpool_patchtokens = True
        elif "vits" in self.arch.lower():
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