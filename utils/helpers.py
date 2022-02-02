from __future__ import annotations

import contextlib
import glob
import logging
import numbers
import os
import shutil
import warnings
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import pytorch_lightning as pl
import math
import torch
from torch import nn
import wandb
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union
import sys
from joblib import Parallel, dump, load
from sklearn.metrics import log_loss, accuracy_score
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np

from argparse import Namespace
from omegaconf import Container, OmegaConf
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader
from collections.abc import MutableMapping


try:
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)

def mean(array):
    """Take mean of array like."""
    return sum(array) / len(array)


def namespace2dict(namespace):
    """
    Converts recursively namespace to dictionary. Does not work if there is a namespace whose
    parent is not a namespace.
    """
    d = dict(**namespace)
    for k, v in d.items():
        if isinstance(v, NamespaceMap):
            d[k] = namespace2dict(v)
    return d

class NamespaceMap(Namespace, MutableMapping):
    """Namespace that can act like a dict."""

    def __init__(self, d):
        # has to take a single argument as input instead of a dictionary as namespace usually do
        # because from pytorch_lightning.utilities.apply_func import apply_to_collection doesn't work
        # with namespace (even though they think it does)
        super().__init__(**d)

    def select(self, k):
        """Allows selection using `.` in string."""
        to_return = self
        for subk in k.split("."):
            to_return = to_return[subk]
        return to_return

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __delitem__(self, k):
        del self.__dict__[k]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)


def omegaconf2namespace(cfg, is_allow_missing=False):
    """Converts omegaconf to namespace so that can use primitive types."""
    cfg = OmegaConf.to_container(cfg, resolve=True)  # primitive types
    return dict2namespace(cfg, is_allow_missing=is_allow_missing)


def dict2namespace(d, is_allow_missing=False, all_keys=""):
    """
    Converts recursively dictionary to namespace. Does not work if there is a dict whose
    parent is not a dict.
    """
    namespace = NamespaceMap(d)

    for k, v in d.items():
        if v == "???" and not is_allow_missing:
            raise ValueError(f"Missing value for {all_keys}.{k}.")
        elif isinstance(v, dict):
            namespace[k] = dict2namespace(v, f"{all_keys}.{k}")
    return namespace


def remove_rf(path: Union[str, Path], not_exist_ok: bool = False) -> None:
    """Remove a file or a folder"""
    path = Path(path)

    if not path.exists() and not_exist_ok:
        return

    if path.is_file():
        path.unlink()
    elif path.is_dir:
        shutil.rmtree(path)

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


class SklearnTrainer:
    """Wrapper around sklearn that mimics pytorch lightning trainer."""

    def __init__(self, hparams):
        self.model = None
        self.hparams = deepcopy(hparams)

    def fit(self, model, datamodule):
        data = datamodule.get_train_dataset()
        model.fit(data.X, data.Y)
        self.model = model

    def save_checkpoint(self, ckpt_path: Union[str, Path], weights_only: bool = False):
        dump(self.model, ckpt_path)

    def test(
        self, dataloaders: DataLoader, ckpt_path: Union[str, Path]
    ) -> list[dict[str, float]]:
        data = dataloaders.dataset
        if ckpt_path is not None and ckpt_path != "best":
            self.model = load(ckpt_path)

        model = self.model
        y = data.Y
        y_pred = model.predict(data.X)
        y_pred_proba = model.predict_proba(data.X)

        results = {}
        name = f"test/{self.hparams.component}"
        results[f"{name}/acc"] = accuracy_score(y, y_pred)
        results[f"{name}/err"] = 1 - results[f"{name}/acc"]
        results[f"{name}/loss"] = log_loss(y, y_pred_proba)

        logger.info(results)

        if wandb.run is not None:
            # log to wandb if its active
            wandb.run.log(results)

        # return a list of dict just like pl trainer (where usually the list is an element for each data loader)
        # here only works with one dataloader
        return [results]


def get_torch_trainer(cfg: NamespaceMap) -> pl.Trainer:
    """Instantiate pytorch lightning trainer."""
    if cfg.callbacks.is_log_wandb:
        check_import("wandb", "WandbLogger")

        # if wandb.run is not None:
        #     wandb.run.finish()  # finish previous run if still on

        try:
            pl_logger = WandbLogger(**cfg.callbacks.wandb_kwargs)
        except Exception:
            cfg.logger.kwargs.offline = True
            pl_logger = WandbLogger(**cfg.callbacks.wandb_kwargs)
    else:
        pl_logger = False

    callbacks = [ModelCheckpoint(**cfg.checkpoint.kwargs),
                 LearningRateMonitor(logging_interval="epoch")]

    trainer = pl.Trainer(
        logger=pl_logger,
        callbacks=callbacks,
        **cfg.trainer,
    )

    # lightning automatically detects slurm and tries to handle checkpointing but we want outside
    # so simply remove hpc save until  #6204 #5225 #6389
    # TODO change when #6389
    trainer.checkpoint_connector.hpc_save = lambda *args, **kwargs: None

    return trainer


def log_dict(trainer: pl.Trainer, to_log: dict, is_param: bool) -> None:
    """Safe logging of param or metrics."""
    try:
        if is_param:
            trainer.logger.log_hyperparams(to_log)
        else:
            trainer.logger.log_metrics(to_log)
    except:
        pass

def replace_str(s, old, new, is_prfx=False, is_sffx=False):
    """replace optionally only at the start or end"""
    assert not (is_prfx and is_sffx)

    if is_prfx:
        if s.startswith(old):
            s = new + s[len(old) :]
    elif is_sffx:
        if s.endswith(old):
            s = s[: -len(old)] + new
    else:
        s = s.replace(old, new)

    return s


def replace_keys(
    d: dict[str, ...], old: str, new: str, is_prfx: bool = False, is_sffx: bool = False
) -> dict[str, ...]:
    """replace keys in a dict."""
    return {
        replace_str(k, old, new, is_prfx=is_prfx, is_sffx=is_sffx): v
        for k, v in d.items()
    }

# taken from https://github.com/rwightman/pytorch-image-models/blob/d5ed58d623be27aada78035d2a19e2854f8b6437/timm/models/layers/weight_init.py
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='truncated_normal'):
    """Initialization by scaling the variance."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        nn.init.trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def init_std_modules(module: nn.Module, nonlinearity: str = "relu") -> bool:
    """Initialize standard layers and return whether was initialized."""
    # all standard layers
    if isinstance(module, nn.modules.conv._ConvNd):
        variance_scaling_(module.weight)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        try:
            nn.init.zeros_(module.bias)
        except AttributeError:
            pass

    elif isinstance(module, nn.modules.batchnorm._NormBase):
        if module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    else:
        return False

    return True



def weights_init(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    nonlinearity : str, optional
        Name of the nn.functional activation. Used for initialization.
    """
    init_std_modules(module)  # in case you gave a standard module

    # loop over direct children (not grand children)
    for m in module.children():

        if init_std_modules(m):
            pass
        elif hasattr(m, "reset_parameters"):
            # if has a specific reset
            # Imp: don't go in grand children because you might have specific weights you don't want to reset
            m.reset_parameters()
        else:
            weights_init(m, nonlinearity=nonlinearity)  # go to grand children

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_load_checkpoint(self, *args, **kwargs):
        super().on_load_checkpoint(*args, **kwargs)

        # trick to keep only one model because pytorch lightning by default doesn't save
        # best k_models, so when preempting they stack up. Open issue. This is only correct for k=1
        self.best_k_models = {}
        self.best_k_models[self.best_model_path] = self.best_model_score
        self.kth_best_model_path = self.best_model_path

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

#
# @contextlib.contextmanager
# def add_sys_path(path: Union[str, os.PathLike]) -> Iterator[None]:
#     """Temporarily add the given path to `sys.path`."""
#     path = os.fspath(path)
#     try:
#         sys.path.insert(0, path)
#         yield
#     finally:
#         sys.path.remove(path)

