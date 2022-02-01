from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from torch import nn

from utils.helpers import check_import, rm_module
from torchvision import transforms
import pytorch_lightning as pl

try:
    import clip
except ImportError:
    pass

try:
    import transformers
except ImportError:
    pass


try:
    # TODO: use something else because bolts SSL is depreciated
    from pl_bolts.models.self_supervised import SimCLR
    from pl_bolts.models.self_supervised import SwAV
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

    if  mode is None or "beit" in mode :
        # see https://huggingface.co/models
        available["hugging"] = "check https://huggingface.co/models?sort=downloads&search=beit"

    # TODO: simclr, more swav,  vissl, barlow twins
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
        encoder = torch.hub.load("facebookresearch/swav:main", model)
        encoder.fc = torch.nn.Identity()
        preprocess = SWAV_PREPROCESSOR

    elif mode == "beit":
        check_import("transformers", "mode=beit in load_representor")
        extractor = transformers.BeitFeatureExtractor.from_pretrained(f"{model}")
        preprocess = lambda img : extractor(img, return_tensors="pt")['pixel_values'][0]
        model = transformers.BeitModel.from_pretrained(f"{model}")
        encoder = HuggingSelector(model, "pooler_output")

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