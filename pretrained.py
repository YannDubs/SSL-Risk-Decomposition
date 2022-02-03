from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import nn

from utils.helpers import check_import, replace_module_prefix, rm_module
from torchvision import transforms
import torchvision
import pytorch_lightning as pl
from torchvision.models.resnet import resnet50

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

SWAV_ADD_MODELS = {"resnet50_ep100": "https://dl.fbaipublicfiles.com/deepcluster/swav_100ep_pretrain.pth.tar",
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
        # more models available at `https://github.com/facebookresearch/swav` e.g. different epochs and batch-size
        available["swav"] = list(torch.hub.list("facebookresearch/swav:main"))
        available["swav"].update(SWAV_ADD_MODELS)

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

class HuggingSelector(nn.Module):
    """Wrapper around hugging face model to select correct output while enable `.cuda()` etc."""
    def __init__(self, model : nn.Module, select : str):
        super().__init__()
        self.model = model
        self.select = select

    def forward(self, x : torch.Tensor):
        return self.model(x)[self.select]

def load_representor(mode: str, model: str) -> Union[Callable, Callable]:
    """Return the encoder and the preprocessor."""
    if mode == "clip":
        check_import("clip", "mode=clip in load_representor")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model, device, jit=False)  # might have to try False
        encoder = model.visual.float()  # only keep the image model

    elif mode == "dino":
        with rm_module("utils"):
            # dirty but if not there's collision of modules
            encoder = torch.hub.load("facebookresearch/dino:main", model)
        encoder.fc = torch.nn.Identity()
        preprocess = DINO_PREPROCESSOR

    elif mode == "swav":
        try:
            encoder = torch.hub.load("facebookresearch/swav:main", model)
        except RuntimeError:
            encoder = resnet50(pretrained=False)
            state_dict = load_state_dict_from_url(
                url=SWAV_ADD_MODELS[model],
                map_location="cpu",
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            encoder.load_state_dict(state_dict, strict=False)

        encoder.fc = torch.nn.Identity()
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
        check_import("vissl", "mode=vissl in load_representor")
        state_dict = load_state_dict_from_url(url=VISSL_MODELS[model], map_location="cpu" )
        if "classy_state_dict" in state_dict.keys():
            state_dict = state_dict["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in state_dict.keys():
            state_dict = state_dict["model_state_dict"]
        state_dict = replace_module_prefix(state_dict, "_feature_blocks.")
        encoder = resnet50(pretrained=False)
        encoder.fc = torch.nn.Identity()
        encoder.load_state_dict(state_dict, strict=False)
        preprocess = VISSL_PREPROCESSOR

    elif mode == "torchvision":
        encoder = torchvision.models.__dict__[model](pretrained=True)
        preprocess = TORCHVISION_PREPROCESSOR

    else:
        raise ValueError(f"Unknown mode={mode}.")

    representor = LightningWrapper(encoder)
    return representor, preprocess

class LightningWrapper(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        return self.encoder(x)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x).cpu(), y.cpu()