from __future__ import annotations

import contextlib
from tqdm import tqdm
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@contextlib.contextmanager
def rm_module(module: str) -> Iterator[None]:
    """Temporarily remove module from sys.Modules."""
    is_module_loaded = module in sys.modules
    try:
        if is_module_loaded:
            val = sys.modules[module]
            del sys.modules[module]
        yield
    finally:
        if is_module_loaded:
            sys.modules[module] = val

class DownloadProgressBar(tqdm):
    """Progress bar for downloading files."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# Modified from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_url(url, save_path):
    """Download a url to `save_path`."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:

        f, _ = urllib.request.urlretrieve(
            url, filename=save_path, reporthook=t.update_to
        )

    return f

@contextlib.contextmanager
def download_url_tmp(url):
    try:
        yield download_url(url, None)
    finally:
        urllib.request.urlcleanup()

class EncoderIndexing(nn.Module):
    def __init__(self, encoder : nn.Module, index: int):
        super().__init__()
        self.encoder = encoder
        self.index = index

    def forward(self, x: torch.Tensor):
        return self.encoder(x)[self.index]

class VITWrapper(nn.Module):
    """
    Follows https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf
    /eval_linear.py#L196

    Parameters
    ----------
    encoder : model
        VIT model.

    representation : {"cls", "cls+avg", "4xcls"}
        Which feature to use as the representation.

    """
    def __init__(self, encoder : nn.Module, representation: str):
        super().__init__()
        self.encoder = encoder

        if representation == "cls+avg":
            self.n_last_blocks = 1
            self.avgpool_patchtokens = True
        elif representation == "4xcls":
            self.n_last_blocks = 4
            self.avgpool_patchtokens = False
        elif representation == "cls":
            self.n_last_blocks = 1
            self.avgpool_patchtokens = False
        else:
            raise ValueError(f"Unknown extract_mode={representation}")

    def forward(self, x: torch.Tensor):
        intermediate_output = self.encoder.get_intermediate_layers(x, self.n_last_blocks)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if self.avgpool_patchtokens:
            output = torch.cat((output.unsqueeze(-1),
                                torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)

        return output

def interpolate_pos_encoding(x, pos_embed):
    """Interpolated the position encoding to the input size. Should not be needed if using the same input size as trained on."""
    npatch = x.shape[1] - 1
    N = pos_embed.shape[1] - 1
    if npatch == N:
        return pos_embed
    class_emb = pos_embed[:, 0]
    pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    pos_embed = F.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=math.sqrt(npatch / N),
        mode='bicubic',
    )
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

def get_intermediate_layers(self, x, n=1):
    """Replicates https://github.com/facebookresearch/dino/blob/3247a0cacb4c0642270469e06facf96e895f56de
    /vision_transformer.py#L225 for TIMM ViT models.
    """

    ### prepare_tokens ###
    x = self.patch_embed(x)
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    if self.pos_embed is not None:
        pos_embed = interpolate_pos_encoding(x, self.pos_embed)
        x = self.pos_drop(x + pos_embed)
    ######################

    output = []
    for i, blk in enumerate(self.blocks):

        if hasattr(self, "rel_pos_bias") and self.rel_pos_bias is not None:
            # BEIT uses a relative positional embedding
            x = blk(x, self.rel_pos_bias)
        else:
            x = blk(x)

        if len(self.blocks) - i <= n:
            output.append(self.norm(x))

    return output

# from https://github.com/facebookresearch/vissl/blob/012f86f249158f00ac009a1cb7504352bcf3c6e6/vissl/utils/checkpoint.py
def replace_module_prefix(
    state_dict: Dict[str, Any], prefix: str, replace_with: str = ""
):
    """
    Remove prefixes in a state_dict needed when loading models that are not VISSL
    trained models.
    Specify the prefix in the keys that should be removed.
    """
    state_dict = {
        (key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict

def check_import(module: str, to_use: Optional[str] = None):
    """Check whether the given module is imported."""
    if module not in sys.modules:
        if to_use is None:
            error = '{} module not imported. Try "pip install {}".'.format(
                module, module
            )
            raise ImportError(error)
        else:
            error = 'You need {} to use {}. Try "pip install {}".'.format(
                module, to_use, module
            )
            raise ImportError(error)
