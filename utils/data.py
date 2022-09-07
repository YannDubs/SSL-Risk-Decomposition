"""
Base dataset.
Most code is reused from: https://github.com/YannDubs/lossyless/tree/main/utils/data
"""
from __future__ import annotations

import ast
import os
from os import path
import math
import abc
import logging
import torch
from pathlib import Path
from typing import Any, Optional, Union
from collections.abc import Callable
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

from utils.helpers import check_import, file_cache, npimg_resize, remove_rf

try:
    # useful to avoid "too many open files" error. See : https://github.com/tensorflow/datasets/issues/1441
    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    import tensorflow_datasets as tfds  # only used for tfds data
except ImportError:
    pass


EXIST_DATA = "data_exist.txt"
DIR = Path(__file__).parents[2].joinpath("data")
logger = logging.getLogger(__name__)

__all__ = ["get_Datamodule"]

def get_Datamodule(datamodule: str) -> type:
    """Return the correct uninstantiated datamodule."""
    datamodule = datamodule.lower()
    if datamodule == "imagenet":
        return ImagenetDataModule
    elif datamodule == "imagenet_v2":
        return ImageNetV2DataModule
    elif datamodule == "imagenet_sketch":
        return ImageNetSketchDataModule
    elif datamodule == "imagenet_r":
        return ImageNetRDataModule
    elif datamodule == "imagenet_a":
        return ImageNetADataModule
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


### Tensorflow Datasets Modules ###
class TensorflowBaseDataset(ImgDataset, ImageFolder):
    """Base class for tensorflow-datasets.

    Parameters
    ----------
    root : str or Path
        Path to directory for saving data.

    split : str, optional
        Split to use, depends on data but usually ["train","test"]

    download : bool, optional
        Whether to download the data if it is not existing.

    kwargs :
        Additional arguments to `ImgDataset` and `ImageFolder`.

    class attributes
    ----------------
    min_size : int, optional
        Resizing of the smaller size of an edge to a certain value. If `None` does not resize.
        Recommended for images that will be always rescaled to a smaller version (for memory gains).
        Only used when downloading.
    """

    min_size = 256

    def __init__(
        self, root, curr_split="train", download=True,  **kwargs,
    ):
        check_import("tensorflow_datasets", "TensorflowBaseDataset")

        self.root = root
        self.curr_split = curr_split

        if download and not self.is_exist_data:
            self.download()

        super().__init__(
            root=self.get_dir(self.curr_split),
            curr_split=curr_split,
            **kwargs,
        )
        self.root = root  # overwirte root for which is currently split folder

    def get_dir(self, split=None):
        """Return the main directory or the one for a split."""
        main_dir = Path(self.root) / self.dataset_name
        if split is None:
            return main_dir
        else:
            return main_dir / split

    @property
    def is_exist_data(self):
        """Whether the data is available."""
        is_exist = True
        for split in self.get_available_splits():
            check_file = self.get_dir(split) / EXIST_DATA
            is_exist &= check_file.is_file()
        return is_exist

    def download(self):
        """Download the data."""
        tfds_splits = self.get_available_splits()
        tfds_datasets, metadata = tfds.load(
            name=self.dataset_name,
            batch_size=1,
            data_dir=self.root,
            as_supervised=True,
            split=tfds_splits,
            with_info=True,
        )
        np_datasets = tfds.as_numpy(tfds_datasets)
        metadata.write_to_directory(self.get_dir())

        for split, np_data in zip(tfds_splits, np_datasets):
            split_path = self.get_dir(split)
            remove_rf(split_path, not_exist_ok=True)
            split_path.mkdir()
            for i, (x, y) in enumerate(tqdm(np_data)):
                if self.min_size is not None:
                    x = npimg_resize(x, self.min_size)

                x = x.squeeze()  # given as batch of 1 (and squeeze if single channel)
                target = y.squeeze().item()

                label_name = metadata.features["label"].int2str(target)
                label_name = label_name.replace(" ", "_")
                label_name = label_name.replace("/", "")

                label_dir = split_path / label_name
                label_dir.mkdir(exist_ok=True)

                img_file = label_dir / f"{i}.jpeg"
                Image.fromarray(x).save(img_file)

        for split in self.get_available_splits():
            check_file = self.get_dir(split) / EXIST_DATA
            check_file.touch()

        # remove all downloading files
        remove_rf(Path(metadata.data_dir))

    @classmethod
    @property
    def dataset_name(cls) -> str:
        """Name of datasets to load, this should be the same as found at `www.tensorflow.org/datasets/catalog/`."""
        raise NotImplementedError()


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

    is_save_features : bool, optional
        Whether to save the features to disk.

    seed : int, optional
        Pseudo random seed.

    is_debug : bool, optional
        Whether debugging, if so uses test set for train to be quicker.

    train_dataset : Dataset, optional
        Training dataset. Useful for out_dist benchmarks that do not have a training set.

    dataset_kwargs : dict, optional
        Additional arguments for the dataset.
    """

    def __init__(
        self,
        representor: Callable,
        representor_name : str,
        data_dir: Union[Path, str] = DIR,
        features_basedir: Union[Path, str] = DIR,
        num_workers: int = 6,
        batch_size: int = 128,
        is_save_features: bool = True,
        seed: int = 123,
        is_debug: int=False,
        train_dataset: Optional[Dataset] = None,
        is_predict_on_test: bool = True,
        is_add_idx: bool=False,
        dataset_kwargs: dict = {}
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.features_basedir = features_basedir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.is_save_features = is_save_features
        self.seed = seed
        self.dataset_kwargs = dataset_kwargs
        self.representor = representor.eval()
        self.representor_name = representor_name
        self.is_debug = is_debug
        self.train_dataset = train_dataset
        self.is_add_idx = is_add_idx
        self.is_predict_on_test = is_predict_on_test
        logger.info(f"Representing data with {representor_name}")
        self.reset()
        self.setup()

        self.z_dim = self.train_dataset.X.shape[1]

        Y_train = self.train_dataset.Y
        if self.is_add_idx:
            Y_train = Y_train[:, 0]

        self.n_labels = len(np.unique(Y_train))

        logger.info(f"z_dim={self.z_dim}")

    def reset(self, is_train_on = "train", is_test_on= "test"):
        self.is_train_on = is_train_on
        self.is_test_on = is_test_on
        logger.info(f"Set model to use is_train_on={self.is_train_on} and is_test_on={self.is_test_on}.")

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "test" or stage is None:
            logger.info("Representing the test set.")
            test_dataset = self.Dataset(
                self.data_dir, curr_split="test", download=True, **self.dataset_kwargs
            )
            self.test_dataset = SklearnDataset(*self.represent(test_dataset, "test"), is_add_idx=self.is_add_idx)

        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                if self.is_debug:
                    logger.info("Using test for train during debug.")
                    self.train_dataset = self.test_dataset
                else:
                    logger.info("Representing the train set.")
                    train_dataset = self.Dataset( self.data_dir, curr_split="train", download=True, **self.dataset_kwargs )
                    self.train_dataset = SklearnDataset(*self.represent(train_dataset, "train"), is_add_idx=self.is_add_idx)

    def get_dataset(self, name):
        """Given a string of the form 'data-split-size' or 'data' return the correct dataset."""

        separator_data_split = "-"
        if separator_data_split in name:
            data, split, sizestr = name.split(separator_data_split)

            if split == "sbst":
                is_complement = False
            elif split == "cmplmnt":
                is_complement = True
            else:
                raise ValueError(f"Unknown split={split}.")

        else:
            data = name
            split = "all"

        if data == "train":
            dataset = self.train_dataset
        elif data == "test":
            dataset = self.test_dataset
        elif data == "union":
            dataset = SklearnConcatDataset([self.train_dataset, self.test_dataset])
        else:
            raise ValueError(f"Unknown data={data}")

        if split == "all":
            pass
        else:
            Y = dataset.Y

            if self.is_add_idx:
                Y = Y[:,0]  # for subseting use real label

            if sizestr == "ntest":
                size = len(self.test_dataset)  # use exactly same number as test
            else:
                size = ast.literal_eval(sizestr)

            dataset = BalancedSubset(dataset, stratify=Y, size=size, seed=self.seed, is_complement=is_complement)

        return dataset

    def get_train_dataset(self):
        return self.get_dataset(self.is_train_on)

    def get_test_dataset(self):
        return self.get_dataset(self.is_test_on)

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

    def predict_dataloader(self) -> DataLoader:
        if self.is_predict_on_test:
            return self.test_dataloader()
        else:
            return self.train_dataloader()

    @classmethod
    @property
    def Dataset(cls) -> Any:
        """Return the correct dataset."""
        raise NotImplementedError()

    @property
    def features_path(self):
        return Path(self.features_basedir) / f"{self.Dataset.__name__}_{self.representor_name}"

    def represent(self, dataset, split, max_chunk_size = 20000):
        batch_size = get_max_batchsize(dataset, self.representor)
        torch.cuda.empty_cache()
        logger.info(f"Selected max batch size for inference: {batch_size}")

        if self.is_save_features:
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
            gpus, precision = 1, 16
        else:
            gpus, precision = 0, 32

        trainer = pl.Trainer(gpus=gpus, precision=precision,
                             logger=False, callbacks=TQDMProgressBar(refresh_rate=20))
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
        return Z, Y


@toma.batch(initial_batchsize=128)  # try largest bach size possible #DEV
def get_max_batchsize(batchsize, dataset, representor):
    """Return the largest batchsize you can fit."""
    # cannot use multiple workers with toma batch size
    dataloader = DataLoader(dataset, batch_size=batchsize, pin_memory=True)
    if torch.cuda.is_available():
        gpus, precision = 1, 16
    else:
        gpus, precision = 0, 32
    trainer = pl.Trainer(gpus=gpus, precision=precision,
                         logger=False, limit_predict_batches=2)
    _ = trainer.predict(model=representor, ckpt_path=None,  dataloaders=[dataloader])
    return batchsize

### HELPERS ###
class SklearnDataset(Dataset):
    """Mapping between numpy (or sklearn) datasets to PyTorch datasets."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_add_idx: bool = False
    ) -> None:
        super().__init__()
        self.X = X
        self.Y = y
        self.is_add_idx = is_add_idx
        if self.is_add_idx:
            self.Y = np.stack([self.Y, np.arange(len(self.Y))], axis=-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]
        return x, y

class SklearnConcatDataset(ConcatDataset):
    @property
    def X(self):
        return np.concatenate([d.X for d in self.datasets], axis=0)

    @property
    def Y(self):
        return np.concatenate([d.Y for d in self.datasets], axis=0)

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

    def __getitem__(self, idx):
        if self.dataset.is_add_idx:
            X, Y = self.dataset[self.indices[idx]]
            Y[1] = idx
            return X, Y
        else:
            return self.dataset[self.indices[idx]]


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

    @file_cache(filename="cached_classes.json")
    def find_classes(self, directory: str, *args, **kwargs) -> tuple[list[str], dict[str, int]]:
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="cached_structure.json")
    def make_dataset(self, directory: str, *args, **kwargs) -> list[tuple[str, int]]:
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset


class ImagenetDataModule(ImgDataModule):

    @classmethod
    @property
    def Dataset(cls) -> Any:
        return ImageNetDataset

# ImageNetV2 #
class ImageNetV2Dataset(TensorflowBaseDataset):
    min_size = 256

    @classmethod
    @property
    def dataset_name(cls):
        return "imagenet_v2"

    @classmethod
    def get_available_splits(cls):
        return ["test"]


class ImageNetV2DataModule(ImgDataModule):

    @classmethod
    @property
    def Dataset(cls):
        return ImageNetV2Dataset

# ImageNetSketch #
class ImageNetSketchDataset(TensorflowBaseDataset):
    min_size = 256

    @classmethod
    @property
    def dataset_name(cls):
        return "imagenet_sketch"

    @classmethod
    def get_available_splits(cls):
        return ["test"]


class ImageNetSketchDataModule(ImgDataModule):

    @classmethod
    @property
    def Dataset(cls):
        return ImageNetSketchDataset

# ImageNetSketch #
class ImageNetRDataset(TensorflowBaseDataset):
    min_size = 256

    @classmethod
    @property
    def dataset_name(cls):
        return "imagenet_r"

    @classmethod
    def get_available_splits(cls):
        return ["test"]


class ImageNetRDataModule(ImgDataModule):

    @classmethod
    @property
    def Dataset(cls):
        return ImageNetRDataset


# ImageNetSketch #
class ImageNetADataset(TensorflowBaseDataset):
    min_size = 256

    @classmethod
    @property
    def dataset_name(cls):
        return "imagenet_a"

    @classmethod
    def get_available_splits(cls):
        return ["test"]


class ImageNetADataModule(ImgDataModule):

    @classmethod
    @property
    def Dataset(cls):
        return ImageNetADataset
