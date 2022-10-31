from __future__ import annotations

import contextlib
import json
import logging
import numbers
import os
import random
import shutil
from copy import deepcopy
from functools import  wraps
import pytorch_lightning as pl
import math
import torch
from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch import nn
from pathlib import Path
from typing import Optional, Union
import sys
from joblib import dump, load
from sklearn.metrics import log_loss, accuracy_score
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
from torchvision.transforms import functional as F_trnsf
from torchvision.transforms import InterpolationMode

from argparse import Namespace
from omegaconf import OmegaConf
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader
from collections.abc import MutableMapping


try:
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)

def file_cache(filename):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator

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

    def save_checkpoint(self, ckpt_path: Union[str, Path], *args, **kwargs):
        dump(self.model, ckpt_path)

    def test(
        self, dataloaders: DataLoader, ckpt_path: Union[str, Path], model: torch.nn.Module=None
    ) -> list[dict[str, float]]:
        data = dataloaders.dataset

        if model is not  None:
            model = self.model

        if ckpt_path is not None and ckpt_path != "best":
            model = load(ckpt_path)

        y = data.Y
        y_pred = model.predict(data.X)
        y_pred_proba = model.predict_proba(data.X)

        results = {}
        name = f"test/{self.hparams.data.name}/{self.hparams.component}"
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
    if cfg.is_log_wandb:
        check_import("wandb", "WandbLogger")

        # if wandb.run is not None:
        #     wandb.run.finish()  # finish previous run if still on

        try:
            pl_logger = WandbLogger(**cfg.wandb_kwargs)
        except Exception:
            cfg.logger.kwargs.offline = True
            pl_logger = WandbLogger(**cfg.wandb_kwargs)
    else:
        pl_logger = False

    callbacks = [ModelCheckpoint(**cfg.checkpoint.kwargs),
                 LearningRateMonitor(logging_interval="epoch")]

    trainer = pl.Trainer(
        logger=pl_logger,
        callbacks=callbacks,
        plugins=[SLURMEnvironment(auto_requeue=False)],  # see lightning #6389. very annoying but it already tries to requeue
        **cfg.trainer,
    )

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

def int_or_ratio(alpha, n):
    """Return an integer for alpha. If float, it's seen as ratio of `n`."""
    if isinstance(alpha, int):
        return alpha
    return int(alpha * n)

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

def npimg_resize(np_imgs, size):
    """Batchwise resizing numpy images."""
    if np_imgs.ndim == 3:
        np_imgs = np_imgs[:, :, :, None]

    torch_imgs = torch.from_numpy(np_imgs.transpose((0, 3, 1, 2))).contiguous()
    torch_imgs = F_trnsf.resize(torch_imgs, size=size, interpolation=InterpolationMode.BICUBIC)
    np_imgs = to_numpy(torch_imgs).transpose((0, 2, 3, 1))
    return np_imgs


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

# modified from https://github.com/skorch-dev/skorch/blob/92ae54b/skorch/utils.py#L106
def to_numpy(X) -> np.array:
    """Convert tensors,list,tuples,dataframes to numpy arrays."""
    if isinstance(X, np.ndarray):
        return X

    # the sklearn way of determining pandas dataframe
    if hasattr(X, "iloc"):
        return X.values

    if isinstance(X, (tuple, list, numbers.Number)):
        return np.array(X)

    if not isinstance(X, (torch.Tensor, PackedSequence)):
        raise TypeError(f"Cannot convert {type(X)} to a numpy array.")

    if X.is_cuda:
        X = X.cpu()

    if X.requires_grad:
        X = X.detach()

    return X.numpy()

class LightningWrapper(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        out = self.encoder(x)
        return out

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x).cpu(), y.cpu()


def set_seed(seed: Optional[int]) -> None:
    """Set the random seed."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


@contextlib.contextmanager
def tmp_seed(seed: Optional[int], is_cuda: bool = torch.cuda.is_available()):
    """Context manager to use a temporary random seed with `with` statement."""
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    if is_cuda:
        torch_cuda_state = torch.cuda.get_rng_state()

    set_seed(seed)
    try:
        yield
    finally:
        if seed is not None:
            # if seed is None do as if no tmp_seed
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            if is_cuda:
                torch.cuda.set_rng_state(torch_cuda_state)

def max_num_workers():
    """Return the maximum number of workers on a machine."""
    try:
        max_num_workers = len(os.sched_getaffinity(0))
    except:
        max_num_workers = os.cpu_count()
    return max_num_workers

def min_max_scale(col):
    """Scale a column to [0, 1]."""
    return (col - col.min()) / (col.max() - col.min())


def update_prepending(to_update, new):
    """Update a dictionary with another. the difference with .update, is that it puts the new keys
    before the old ones (prepending)."""
    # makes sure don't update arguments
    to_update = to_update.copy()
    new = new.copy()

    # updated with the new values appended
    to_update.update(new)

    # remove all the new values => just updated old values
    to_update = {k: v for k, v in to_update.items() if k not in new}

    # keep only values that ought to be prepended
    new = {k: v for k, v in new.items() if k not in to_update}

    # update the new dict with old one => new values are at the beginning (prepended)
    new.update(to_update)

    return new


class StrFormatter:
    """String formatter that acts like some default dictionary `"formatted" == StrFormatter()["to_format"]`.

    Parameters
    ----------
    exact_match : dict, optional
        dictionary of strings that will be replaced by exact match.

    substring_replace : dict, optional
        dictionary of substring that will be replaced if no exact_match. Order matters.
        Everything is title case at this point.

    to_upper : list, optional
        Words that should be upper cased.
    """

    def __init__(self, exact_match={}, substring_replace={}, to_upper=[]):
        self.exact_match = exact_match
        self.substring_replace = substring_replace
        self.to_upper = to_upper

    def __getitem__(self, key):
        if not isinstance(key, str):
            return key

        if key in self.exact_match:
            return self.exact_match[key]

        key = key.title()

        for match, replace in self.substring_replace.items():
            key = key.replace(match, replace)

        for w in self.to_upper:
            key = key.replace(w, w.upper())

        return key

    def __call__(self, x):
        return self[x]

    def update(self, new_dict):
        """Update the substring replacer dictionary with a new one (missing keys will be prepended)."""
        self.substring_replace = update_prepending(self.substring_replace, new_dict)