from __future__ import annotations

import contextlib
import json
import logging
import numbers
import shutil
from tqdm import tqdm
import urllib.request
from copy import deepcopy
from functools import  wraps
import pytorch_lightning as pl
import math
import torch
from torch import nn
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union
import sys
from joblib import Parallel, dump, load
from sklearn.metrics import log_loss, accuracy_score
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from torchvision.transforms import functional as F_trnsf
from torchvision.transforms import InterpolationMode

from argparse import Namespace
from omegaconf import Container, OmegaConf
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader
from collections.abc import MutableMapping


try:
    import wandb
except ImportError:
    pass

try:
    import cv2
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

    def save_checkpoint(self, ckpt_path: Union[str, Path], weights_only: bool = False):
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

class ImgPil2LabTensor(torch.nn.Module):
    """
    Convert a PIL image to LAB tensor of shape C x H x W
    This transform was proposed in Colorization - https://arxiv.org/abs/1603.08511
    The input image is PIL Image. We first convert it to tensor
    HWC which has channel order RGB. We then convert the RGB to BGR
    and use OpenCV to convert the image to LAB. The LAB image is
    8-bit image in range > L [0, 255], A [0, 255], B [0, 255]. We
    rescale it to: L [0, 100], A [-128, 127], B [-128, 127]
    The output is image torch tensor.
    """

    def __init__(self, indices = []):
        super().__init__()
        check_import("cv2", "ImgPil2LabTensor")
        self.indices = indices

    def forward(self, image):
        img_tensor = np.array(image)
        # PIL image tensor is RGB. Convert to BGR
        img_bgr = img_tensor[:, :, ::-1]
        img_lab = self._convertbgr2lab(img_bgr.astype(np.uint8))
        # convert HWC -> CHW. The image is LAB.
        img_lab = np.transpose(img_lab, (2, 0, 1))
        # torch tensor output
        img_lab_tensor = torch.from_numpy(img_lab).float()

        return img_lab_tensor

    def _convertbgr2lab(self, img):

        # img is [0, 255] , HWC, BGR format, uint8 type
        assert len(img.shape) == 3, "Image should have dim H x W x 3"
        assert img.shape[2] == 3, "Image should have dim H x W x 3"
        assert img.dtype == np.uint8, "Image should be uint8 type"
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # 8-bit image range -> L [0, 255], A [0, 255], B [0, 255]. Rescale it to:
        # L [0, 100], A [-128, 127], B [-128, 127]
        img_lab = img_lab.astype(np.float32)
        img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
        img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
        return img_lab

class DownloadProgressBar(tqdm):
    """Progress bar for downloading files."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# Modified from https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
def download_url(url, save_dir, filename=None):
    """Download a url to `save_dir`."""
    if filename is None:
        filename = url.split("/")[-1]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:

        urllib.request.urlretrieve(
            url, filename=save_dir / filename, reporthook=t.update_to
        )