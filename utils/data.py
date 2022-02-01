"""
Base dataset.
Most code is reused from: https://github.com/YannDubs/lossyless/tree/main/utils/data
"""
from __future__ import annotations

import os
from os import path
import abc
import logging
import torch
from pathlib import Path
from typing import Any, Optional, Union
from collections.abc import Callable
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from torchvision.datasets import ImageNet

DIR = Path(__file__).parents[2].joinpath("data")
logger = logging.getLogger(__name__)

__all__ = ["get_Datamodule"]

def get_Datamodule(datamodule: str) -> type:
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "imagenet":
        return ImagenetDataModule
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
class ImgDataModule(LightningDataModule):
    """Base class for data module for ISSL.

    Parameters
    -----------
    representor : Callable

    data_dir : str, optional
        Directory for saving/loading the dataset.

    num_workers : int, optional
        How many workers to use for loading data

    batch_size : int, optional
        Number of example per batch for training.

    seed : int, optional
        Pseudo random seed.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.
    """

    def __init__(
        self,
        representor: Callable,
        data_dir: Union[Path, str] = DIR,
        num_workers: int = 8,
        batch_size: int = 128,
        seed: int = 123,
        dataset_kwargs: dict = {}
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        self.representor=representor
        self.reset()
        self.setup()

        self.z_dim = self.train_dataset.X.shape[1]
        self.n_labels = len(np.unique(self.train_dataset.Y))

    def reset(self, subset_train_size=None, is_train_on_test = False, is_test_on_train=False, is_test_nonsubset_train=False):
        self.subset_train_size = subset_train_size
        self.is_train_on_test = is_train_on_test
        self.is_test_on_train = is_test_on_train
        self.is_test_nonsubset_train = is_test_nonsubset_train

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "test" or stage is None:
            logger.info("Representing the test set.")
            test_dataset = self.Dataset(
                self.data_dir, curr_split="test", download=True, **self.dataset_kwargs
            )
            #test_dataset = Subset(test_dataset, indices=list(range(1000))) #DEV
            self.test_dataset = SklearnDataset(*self.represent(test_dataset))

        if stage == "fit" or stage is None:
            logger.info("Representing the train set.")
            train_dataset = self.Dataset( self.data_dir, curr_split="train", download=True, **self.dataset_kwargs )
            self.train_dataset = SklearnDataset(*self.represent(train_dataset))
            #self.train_dataset = self.test_dataset #DEV

    def get_train_dataset(self):
        if self.is_train_on_test:
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset

        if self.subset_train_size is not None:
            Y = dataset.Y
            dataset = BalancedSubset(dataset, stratify=Y, size=self.subset_train_size, seed=self.seed)

        return dataset

    def get_test_dataset(self):
        if self.is_test_on_train or self.is_test_nonsubset_train:
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset

        if self.is_test_nonsubset_train:
            Y = dataset.Y
            dataset = BalancedSubset(dataset, stratify=Y, size=self.subset_train_size, seed=self.seed, is_complement=True)

        return dataset

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.get_train_dataset(),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.get_test_dataset(),
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            batch_size=self.batch_size
        )

    @classmethod
    @property
    def Dataset(cls) -> Any:
        """Return the correct dataset."""
        raise NotImplementedError()

    def represent(self, dataset):
        trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, logger=False)
        dataloader = DataLoader(
            dataset,
            batch_size=128,
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

        X, Y = zip(*out)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        return X, Y

### HELPERS ###
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

class BalancedSubset(Subset):
    """Split the dataset into a subset with possibility of stratifying.

    Parameters
    ----------
    dataset : Dataset
        Dataset to subset.

    size : float or int, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of
            the dataset to retain. If int, represents the absolute number or examples.

    stratify : array-like, optional
        What to stratify on.

    seed : int, optional
        Random seed.

    is_complement : bool, optional
        Whether to use the complement of the data rather than the selected data.
    """

    def __init__(
        self,
        dataset: Dataset,
        size: float = 0.1,
        stratify: Any = None,
        seed: Optional[int] = 123,
        is_complement: bool=False
    ):
        complement_idcs, subset_idcs = train_test_split(
            range(len(dataset)), stratify=stratify, test_size=size, random_state=seed
        )
        idcs = complement_idcs if is_complement else subset_idcs
        super().__init__(dataset, idcs)

    @property
    def X(self):
        return self.dataset.X[self.indices]

    @property
    def Y(self):
        return self.dataset.Y[self.indices]

    ### DATA ###

# Imagenet #
class ImageNetDataset(ImgDataset, ImageNet):

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


class ImagenetDataModule(ImgDataModule):

    @classmethod
    @property
    def Dataset(cls) -> Any:
        return ImageNetDataset