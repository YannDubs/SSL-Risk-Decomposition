"""
Base dataset.
Most code is reused from: https://github.com/YannDubs/lossyless/tree/main/utils/data
"""
from __future__ import annotations

import ast
import os
import random
import time
import warnings
from functools import partial
from os import path
import math
import abc
import logging
import torch
from pathlib import Path
from typing import Any, Optional, Union
from collections.abc import Callable

from pytorch_lightning.plugins.environments import SLURMEnvironment
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from toma import toma
from tqdm import tqdm
from PIL import Image

from torchvision.datasets import ImageNet, ImageFolder

from utils.helpers import file_cache, int_or_ratio, max_num_workers, npimg_resize, remove_rf, tmp_seed



EXIST_DATA = "data_exist.txt"
DIR = Path(__file__).parents[2].joinpath("data")
logger = logging.getLogger(__name__)

__all__ = ["get_Datamodule"]

def get_Datamodule(datamodule: str) -> type:
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "imagenet":
        return ImagenetFeaturizedDataModule
    else:
        raise ValueError(f"Unknown datamodule: {datamodule}")

### Base Dataset ###
class ImgDataset(abc.ABC):
    """Base class for Invariant SSL.

    Parameters
    -----------
    curr_split : str, optional
        Which data split you are considering.

    transform : callable, optional
        Preprocessor to apply to each image.

    seed : int, optional
        Pseudo random seed.
    """

    def __init__(
        self,
        *args,
        curr_split: str,
        transform: Optional[Callable] = None,
        seed: int = 123,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.curr_split = curr_split
        self.transform = transform
        self.seed = seed


### Base Datamodule ###
# cannot use abc because inheriting from lightning :(
class BaseDataModule(LightningDataModule):
    """Base class for data module for ISSL.

    Parameters
    -----------
    data_dir : str, optional
        Directory for saving/loading the dataset.

    num_workers : int, optional
        How many workers to use for loading data. If -1 uses all but one.

    batch_size : int, optional
        Number of example per batch for training.

    seed : int, optional
        Pseudo random seed.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.
    """

    def __init__(
        self,
        data_dir: Union[Path, str] = DIR,
        num_workers: int = -1,
        batch_size: int = 128,
        seed: int = 123,
        dataset_kwargs: dict = {},
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.num_workers = (max_num_workers()+num_workers) if num_workers < 0 else num_workers
        logger.info(f"Using num_workers={self.num_workers}, max={max_num_workers()}.")
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs

        # see #9943 pytorch lightning now calls setup each time
        self._already_setup = {}
        for stage in ("fit", "validate", "test", "predict"):
            self._already_setup[stage] = False

        self.setup()


        Y_train = self.train_dataset.Y
        self.label_set = np.unique(Y_train)
        self.n_labels = len(self.label_set)

        self.reset()

    def set_fit(self):
        pass

    def set_eval(self):
        pass

    @property
    def len_train(self):
        if hasattr(self, "total_train_size"):
            return self.total_train_size
        return len(self.train_dataset)

    def reset(self, is_train_on="train", is_test_on="test", label_size=None):
        self.is_train_on = is_train_on
        self.is_test_on = is_test_on

        if label_size is not None:
            n_selected_labels = int_or_ratio(label_size, len(self.label_set))
            with tmp_seed(self.seed):
                self.labels_to_keep = np.random.permutation(self.label_set)[:n_selected_labels]
        else:
            n_selected_labels = len(self.label_set)
            self.labels_to_keep = None

        logger.info(f"Set data to use is_train_on={self.is_train_on} and is_test_on={self.is_test_on} and label_size={n_selected_labels}.")

    def get_initial_test_dataset(self):
        return self.Dataset(
                    self.data_dir, curr_split="test", download=True, **self.dataset_kwargs
                )

    def get_initial_train_dataset(self):
        return self.Dataset(
                    self.data_dir, curr_split="train", download=True, **self.dataset_kwargs
                )

    def setup(self, stage: Optional[str] = None) -> None:

        if (not self._already_setup["test"]) and (stage == "test" or stage is None):
            self.test_dataset = self.get_initial_test_dataset()
            self._already_setup["test"] = True

        if (not self._already_setup["fit"]) and (stage == "fit" or stage is None):
            self.train_dataset = self.get_initial_train_dataset()
            logger.info("Representing the train set.")
            self._already_setup["fit"] = True

    def get_dataset(self, name):
        """Given a string of the form 'data-split-size' or 'data' return the correct dataset."""

        separator_data_split = "-"
        if separator_data_split in name:
            if name.count(separator_data_split) == 3:
                data, split, sizestr, seed = name.split(separator_data_split)
                with tmp_seed(int(seed)):
                    # sample some seed increment to be able to change subsets in a reproducible way
                    seed_add = random.randint(10, int(1e4))
            elif name.count(separator_data_split) == 2:
                seed_add = 0
                data, split, sizestr = name.split(separator_data_split)
            else:
                raise ValueError(f"Cannot split {name} using {separator_data_split}")
        else:
            data = name
            split = "all"

        if data == "train":
            dataset = self.train_dataset
        elif data == "test":
            dataset = self.test_dataset
        elif data == "union":
            dataset = concatenate_datasets([self.train_dataset, self.test_dataset])
        else:
            raise ValueError(f"Unknown data={data}")

        if self.labels_to_keep is not None:
            # subset labels if necessary
            dataset = label_subset(dataset, labels_to_keep=self.labels_to_keep)

        if split == "all":
            pass
        else:
            size = self.get_size(sizestr, data)

            if split == "balsbst":
                f_subset = balanced_subset
            elif split == "nperclass":
                f_subset = partial(balanced_subset, is_n_per_class=True)
            else:
                f_subset = partial(stratified_subset, split=split)

            dataset = f_subset(dataset, size=size, seed=self.seed+seed_add)

        return dataset

    def get_size(self, sizestr, data):
        if sizestr == "ntest":
            size = len(self.test_dataset)  # use exactly same number as test
        elif "ntrain" in sizestr and "." in sizestr:
            # uses percentage of ntrain regardless
            sffx = sizestr.replace("ntrain", "")
            size = int(ast.literal_eval(sffx) * self.len_train)
        else:
            size = ast.literal_eval(sizestr)

        return size

    def get_train_dataset(self):
        return self.get_dataset(self.is_train_on)

    def get_test_dataset(self):
        return self.get_dataset(self.is_test_on)

    def train_dataloader(self, dataset=None) -> DataLoader:
        """Return the training dataloader."""
        if dataset is None:
            dataset = self.get_train_dataset()
        return DataLoader(
            dataset,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            batch_size=self.batch_size
        )

    def test_dataloader(self, dataset=None) -> DataLoader:
        """Return the test dataloader."""
        if dataset is None:
            dataset = self.get_test_dataset()
        return DataLoader(
            dataset,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            batch_size=self.batch_size
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    @classmethod
    @property
    def Dataset(cls) -> Any:
        """Return the correct dataset."""
        raise NotImplementedError()


class FeaturizedDataModule(BaseDataModule):
    """Base class for datamodules that featurizes data before training.

    Parameters
    -----------
    representor : Callable

    representor_name : str
        Name of the representor

    is_save_features : bool, optional
        Whether to save the features to disk.

    features_basedir : str
        Directory to save the features. if `is_save_features` is True.

    is_avoid_raw_dataset : bool, optional
        Avoids using the raw dataset by only relying on the saved features.
        This is useful if the eaw dataset is not available anymore.

    subset_raw_dataset : float, optional
        Percentage of the raw dataset to use. This is useful if the raw dataset is large
        and you only end up need a small fraction of it.
    """

    def __init__(
        self,
        representor: Callable,
        representor_name: str,
        *args,
        features_basedir: Union[Path, str] = DIR,
        is_save_features: bool = True,
        is_avoid_raw_dataset: bool=False,
        subset_raw_dataset: Optional[float]=None,
        **kwargs
    ) -> None:

        self.is_save_features = is_save_features
        self.features_basedir = features_basedir
        self.representor = representor.eval()
        self.representor_name = representor_name
        self.is_avoid_raw_dataset = is_avoid_raw_dataset
        self.subset_raw_dataset = subset_raw_dataset
        logger.info(f"Representing data with {representor_name}")

        super().__init__(*args, **kwargs)

        self.z_dim = self.train_dataset.X.shape[1]
        logger.info(f"z_dim={self.z_dim}")

    def get_initial_dataset(self, split, subset=None):
        logger.info(f"Representing the {split} set.")
        if self.is_avoid_raw_dataset:
            dataset = SklearnDataset(*self.represent(None, split))
        else:
            assert not (self.is_save_features and subset is not None)  # don't compute and save subset of features
            dataset = self.Dataset(
                self.data_dir, curr_split=split, download=True, **self.dataset_kwargs
            )

        total_size = len(dataset)

        if subset is not None:
            n_select = int_or_ratio(subset, len(dataset))
            dataset = stratified_subset(dataset, size=n_select, seed=self.seed)

        if not self.is_avoid_raw_dataset:
            dataset = SklearnDataset(*self.represent(dataset, split))

        return dataset, total_size

    def get_initial_test_dataset(self, subset_dataset=None):
        return self.get_initial_dataset("test", subset_dataset)[0]

    def get_initial_train_dataset(self, subset_dataset=None):

        if subset_dataset is None:
            dataset, total_size = self.get_initial_dataset("train", self.subset_raw_dataset)

            if self.subset_raw_dataset is not None:
                self.total_train_size = total_size
        else:
            dataset, _ = self.get_initial_dataset("train", subset_dataset)

        return dataset

    def get_size(self, sizestr, data):
        size = super().get_size(sizestr, data)

        if data == "train" and self.subset_raw_dataset is not None:
            # you still want subsets to be computed on real size
            size = int_or_ratio(size, self.len_train)

        return size

    @property
    def features_path(self):
        return Path(self.features_basedir) / f"{self.Dataset.__name__}_{self.representor_name}"

    def represent(self, dataset, split, max_chunk_size=20000):
        if not self.is_avoid_raw_dataset:
            batch_size = get_max_batchsize(dataset, self.representor)
            torch.cuda.empty_cache()
            logger.info(f"Selected max batch size for inference: {batch_size}")

        if self.is_save_features:
            if self.is_avoid_raw_dataset:
                N = self.Dataset.split_lengths[split]
            else:
                N = len(dataset)
            n_chunks = math.ceil(N / max_chunk_size)
            Z_files, Y_files = [], []
            data_path = self.features_path / split
            data_path.mkdir(parents=True, exist_ok=True)
            for i, idcs in enumerate(np.array_split(np.arange(N), n_chunks)):
                Z_file = data_path / f"Z_{i}.npy"
                Y_file = data_path / f"Y_{i}.npy"
                Z_files.append(Z_file)
                Y_files.append(Y_file)

                if Z_file.exists() and Y_file.exists():
                    logger.info(f"Skipping chunk representation: found existing {Z_file}.")
                    continue

                assert not self.is_avoid_raw_dataset

                chunk = Subset(dataset, indices=idcs)
                Z_i, Y_i = self.represent_chunk(chunk, batch_size)
                np.save(Z_file, Z_i)
                np.save(Y_file, Y_i)
                logger.info(f"Saving chunk representation to {Z_file}.")

            Z = np.concatenate([np.load(Z_file, allow_pickle=True) for Z_file in Z_files], axis=0)
            Y = np.concatenate([np.load(Y_file, allow_pickle=True) for Y_file in Y_files], axis=0)

            return Z, Y
        else:
            return self.represent_chunk(dataset, batch_size)

    def represent_chunk(self, dataset, batch_size):

        if torch.cuda.is_available():
            accelerator, precision = "gpu", 16
        else:
            accelerator, precision = "cpu", 32

        trainer = pl.Trainer(accelerator=accelerator,
                            devices=1,
                             precision=precision,
                             logger=False,
                             plugins=[SLURMEnvironment(auto_requeue=False)],   # see lightning #6389. very annoying but it already tries to requeue
                            callbacks=TQDMProgressBar(refresh_rate=20))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        out = trainer.predict(
            model=self.representor,
            ckpt_path=None,  # use current model
            dataloaders=[dataloader],
        )

        Z, Y = zip(*out)
        Z = np.concatenate(Z, axis=0)
        Y = np.concatenate(Y, axis=0)
        assert Z.ndim == 2
        return Z, Y

### HELPERS ###
class AugmentedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

@toma.batch(initial_batchsize=128)  # try largest bach size possible #DEV
def get_max_batchsize(batchsize, dataset, representor):
    """Return the largest batchsize you can fit."""
    # cannot use multiple workers with toma batch size
    dataloader = DataLoader(dataset, batch_size=batchsize, pin_memory=True)
    if torch.cuda.is_available():
        accelerator, precision = "gpu", 16
    else:
        accelerator, precision = "cpu", 32

    trainer = pl.Trainer(accelerator=accelerator,
                         devices=1,
                         precision=precision,
                         logger=False,
                         plugins=[SLURMEnvironment(auto_requeue=False)],   # see lightning #6389. very annoying but it already tries to requeue
                         limit_predict_batches=2)
    _ = trainer.predict(model=representor, ckpt_path=None,  dataloaders=[dataloader])
    return batchsize

class SklearnDataset(Dataset):
    """Mapping between numpy (or sklearn) datasets to PyTorch datasets."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        super().__init__()
        self.X = X
        self.Y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]
        return x, y

def concatenate_datasets(datasets):
    if all([d for d in datasets if isinstance(d, SklearnDataset)]):
        return SklearnConcatDataset(datasets)
    return ConcatDataset(datasets)

class SklearnConcatDataset(ConcatDataset):
    @property
    def X(self):
        return np.concatenate([d.X for d in self.datasets], axis=0)

    @property
    def Y(self):
        return np.concatenate([d.Y for d in self.datasets], axis=0)

class BaseSubset(Subset):
    @property
    def X(self):
        return self.dataset.X[self.indices]

    @property
    def Y(self):
        return self.dataset.Y[self.indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def stratified_subset(dataset: Dataset,
                        size: float = 0.1,
                        seed: Optional[int] = 123,
                        split: str="sbst"):
    """Split the dataset into a subset with possibility of stratifying.

    Parameters
    ----------
    dataset : Dataset
        Dataset to subset.

    size : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of
            the dataset to retain. If int, represents the absolute number or examples.

    seed : int, optional
        Random seed.

    split : {"sbst","cmplmnt","sbstcmplmnt", "sbstY"}, optional
        Which split of the subsetted data to use. If "sbst" sues the subset, if "cmplment"
        uses the complement of the subset, if "sbstcmplmnt" uses a subset of the complement of size
        `size`.

    """

    Y = dataset.Y

    # subset by indices
    complement_idcs, subset_idcs = train_test_split(
        range(len(dataset)), stratify=Y, test_size=size, random_state=seed
    )

    if split == "sbst":
        idcs = subset_idcs
    elif split == "cmplmnt":
        idcs = complement_idcs
    elif split == "sbstcmplmnt":
        # further splits the complement
        _, idcs = train_test_split(complement_idcs, stratify=Y, test_size=size, random_state=seed)
    else:
        raise ValueError(f"Unknown split={split}.")

    return BaseSubset(dataset, idcs)



def balanced_subset(dataset,
                     size: float = 0.1,
                     seed: Optional[int] = 123,
                     is_n_per_class: bool = False):
    Y = dataset.Y

    n_classes = len(np.unique(Y))
    if is_n_per_class:
        n_per_class = size
        n_additional = 0
    else:
        size = int_or_ratio(size, len(dataset))
        n_per_class = size // n_classes
        n_additional = size % n_classes
    idcs = np.arange(len(dataset))

    with tmp_seed(seed):
        additional = np.random.choice(range(n_classes), size=n_additional, replace=False)

        selected = []
        for y in range(n_classes):
            i_y = (idcs[Y == y]).tolist()
            random.shuffle(i_y)
            is_add = int(y in additional)
            selected += i_y[:n_per_class + is_add]

        random.shuffle(selected)  # shouldn't be needed

    return BaseSubset(dataset, selected)


def label_subset(dataset, labels_to_keep):
    """Subset a dataset by labels.

    Parameters
    ----------
    dataset : Dataset
        Dataset to subset.

    labels_to_keep : array like
        Which labels to keep
    """
    idcs = np.isin(dataset.Y, labels_to_keep).nonzero()[0]
    return BaseSubset(dataset, idcs)



### DATA ###

# Imagenet #
class ImageNetDataset(ImgDataset, ImageNet):
    split_lengths = {"train": 1281167, "test": 50000}

    def __init__(
        self,
        root: str,
        *args,
        curr_split: str = "train",
        download=None,  # for compatibility
        transform=None,
        **kwargs,
    ) -> None:

        if os.path.isdir(path.join(root, "imagenet256")):
            # use 256 if already resized
            data_dir = path.join(root, "imagenet256")
        elif os.path.isdir(path.join(root, "imagenet")):
            data_dir = path.join(root, "imagenet")
        else:
            raise ValueError(
                f"Imagenet data folder (imagenet256 or imagenet) not found in {root}."
                "This has to be installed manually as download is not available anymore."
            )

        # imagenet test set is not available so it is standard to use the val split as test
        split = "val" if curr_split == "test" else curr_split

        super().__init__(
            data_dir,
            *args,
            curr_split=curr_split,  # goes to ISSLImgDataset
            split=split,  # goes to imagenet
            transform=transform,
            **kwargs,
        )

        if len(self) != self.split_lengths[curr_split]:
            logger.info(f"The length of the dataset is different than expected {len(self)}!={self.split_lengths[curr_split]}")

        self.Y = np.array(self.targets).copy()

    @file_cache(filename="cached_classes.json")
    def find_classes(self, directory: str, *args, **kwargs) -> tuple[list[str], dict[str, int]]:
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="cached_structure.json")
    def make_dataset(self, directory: str, *args, **kwargs) -> list[tuple[str, int]]:
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset


class ImagenetFeaturizedDataModule(FeaturizedDataModule):

    @classmethod
    @property
    def Dataset(cls) -> Any:
        return ImageNetDataset


