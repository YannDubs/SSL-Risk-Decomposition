"""Entry point to compute the loss decomposition for different models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations


import logging
import traceback
from pathlib import Path
import os
import sys

import numpy as np
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import Container

from main import instantiate_datamodule_
from main_augs import get_train_transform
from utils.cluster import nlp_cluster

from utils.helpers import LightningWrapper, tmp_seed
import hubconf

try:
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)
RESULTS_FILE = "results_{component}.csv"
LAST_CHECKPOINT = "last.ckpt"
FILE_END = "end.txt"

@hydra.main(config_name="main", config_path="config")
def main_except(cfg):

    if cfg.is_nlp_cluster:
        with nlp_cluster(cfg):
            main(cfg)
    else:
        main(cfg)

def main(cfg):
    logger.info(os.uname().nodename)

    ############## STARTUP ##############
    logger.info("Stage : Startup")
    begin(cfg)
    save_dir = Path(cfg.paths.statistics)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {save_dir}.")

    last_file = save_dir / f"trainrealaug_statistics.npz"
    if last_file.exists() and not cfg.statistics.force_recompute:
        logger.info(f"Results already exist at {last_file}, skipping.")
        return

    ############## COMPUTING STATISTICS WITH PRETRAINED FEATURES ##############
    logger.info(f"Representing data with {cfg.representor}")
    representor, preprocess = hubconf.__dict__[cfg.representor]()
    representor = LightningWrapper(representor)
    datamodule = instantiate_datamodule_(cfg, representor, preprocess)

    for data, datagetter in dict(test=datamodule.get_test_dataset, train=datamodule.get_train_dataset).items():
        dataset = datagetter()
        Z = torch.tensor(dataset.X).to(torch.float64)
        Y = torch.tensor(dataset.Y)
        if torch.cuda.is_available():
            Z = Z.cuda()
            Y = Y.cuda()
        rank, log_eigenv, uniformity = get_eff_dim(Z)
        logger.info(f"Effective dimension for {data}: {rank}. uniformity: {uniformity}")

        nc1, intra_var, inter_var, alignment = get_collapse(Z, Y)
        logger.info(f"For {data}. Intra {intra_var}, inter variance {inter_var}. NC1: {nc1}. Alignment: {alignment}")

        np.savez(save_dir/f"{data}_statistics.npz",
                 rank=rank, log_eigenv=log_eigenv, nc1=nc1,
                 intra_var=intra_var, inter_var=inter_var,
                 uniformity=uniformity, alignment=alignment)

    datamodule.is_avoid_raw_dataset = False
    datamodule.is_save_features = False

    # computes statistics for images that are always augmented using a standard and unique transformation
    train_transform = get_train_transform(preprocess)
    compute_augstatistics(datamodule,
                          train_transform,
                          save_dir,
                          filename="{data}aug_statistics.npz",
                          n_augs=cfg.statistics.n_augs)

    # computes statistics for images that are augmented with the actual transformation used for training
    _, train_transform = hubconf.__dict__[cfg.representor](is_train_transform=True)
    compute_augstatistics(datamodule,
                          train_transform,
                          save_dir,
                          filename="{data}realaug_statistics.npz",
                          n_augs=cfg.statistics.n_augs)

def compute_augstatistics(datamodule, train_transform, save_dir, filename="{data}aug_statistics.npz", n_augs=10):
    # use the same number of samples for estimating aug than for the train statistics (after augmentation)
    n_train_samples = int(datamodule.len_train * datamodule.subset_raw_dataset) // n_augs
    for data, datagetter in dict(train=datamodule.get_initial_train_dataset,
                                 test=datamodule.get_initial_test_dataset).items():
        if data == "train":
            n_samples = n_train_samples
        elif data == "test":
            n_samples = min(n_train_samples, len(datamodule.test_dataset))

        Z, Y = [], []
        for seed in range(n_augs):
            # ! DEV making sure that sampling is correct => remove train_transform and variance should be zero
            datamodule.dataset_kwargs["transform"] = train_transform
            with tmp_seed(seed):
                # note that we are not resetting the datamodule's seed => actually the examples will be in order
                # => can use their index as the class. But we are setting a different seed for augmentations
                dataset = datagetter(subset_dataset=n_samples)
            Z += [torch.tensor(dataset.X)]
            Y += [torch.arange(len(dataset.X))]

        Z = torch.concat(Z, dim=0).to(torch.float64)
        Y = torch.concat(Y, dim=0)
        if torch.cuda.is_available():
            Z = Z.cuda()
            Y = Y.cuda()

        filename = filename.format(data=data)
        nc1, intra_var, inter_var, alignment = get_collapse(Z, Y)
        logger.info(f"For {filename}. Intra {intra_var}, inter variance {inter_var}. NC1: {nc1}. Alignment: {alignment}")

        np.savez(save_dir / filename,
                 nc1=nc1, intra_var=intra_var, inter_var=inter_var, alignment=alignment)

def get_eff_dim(Z, t_uniformity=2):
    corr_coef = Z.T.corrcoef()
    # remove if all nan
    nan_cols = corr_coef.isnan().all(1)
    corr_coef = corr_coef[~nan_cols]
    corr_coef = corr_coef.T[~nan_cols].T
    rank = torch.linalg.matrix_rank(corr_coef, atol=1e-4, rtol=0.01, hermitian=True)
    log_eigenv = torch.linalg.eigvalsh(corr_coef).abs().log().sort(descending=True)[0]

    # for memory reasons, uniformity has to be computes on batches (can't compute pairwise distance of 1M examples)
    uniformity = 0
    n_batches = 1000
    for batch in Z[torch.randperm(len(Z))].chunk(n_batches):
        # uniformity is computed on normalized features
        batch = torch.nn.functional.normalize(batch, dim=-1)
        uniformity += torch.pdist(batch, p=2).pow(2).mul(-t_uniformity).exp().mean()
    uniformioty = (uniformity / n_batches).log()

    return rank.cpu().numpy(), log_eigenv.cpu().numpy(), uniformity.cpu().numpy()

def get_collapse(Z,Y):
    Z_Norm = torch.nn.functional.normalize(Z, dim=-1)

    Y_unique = Y.unique()
    means = torch.zeros((len(Y_unique), Z.shape[1])).to(torch.float64).to(Z.device)
    intra_cov = 0
    intra_var = 0
    alignment = 0
    for i, y in enumerate(Y_unique):
        Z_y = Z[Y==y]
        means[i,:] = Z_y.mean(0)
        intra_cov += Z_y.T.cov()
        intra_var += (Z_y - means[i]).pow(2).sum(1).mean()

        # alignment is computed on normalized features
        Z_norm_y = Z_Norm[Y == y]
        # E[||f(x)-f(x')||^2] = 2 * \sum_i Var(f_i(x)). Compute O(n) instead of O(n^2)
        alignment += 2 * (Z_norm_y - Z_norm_y.mean(0)).pow(2).sum(1).mean()

    inter_cov = means.T.cov()
    intra_cov /= len(Y_unique)
    intra_var /= len(Y_unique)
    inter_var = (means - means.mean(0)).pow(2).sum(1).mean()
    alignment /= len(Y_unique)

    nc1 = torch.trace(torch.linalg.pinv(inter_cov, atol=1e-4, rtol=0.01, hermitian=True) @ intra_cov)

    return nc1.cpu().numpy(), intra_var.cpu().numpy(), inter_var.cpu().numpy(), alignment.cpu().numpy()

def begin(cfg: Container) -> None:
    """Script initialization."""
    pl.seed_everything(cfg.seed)
    cfg.paths.work = str(Path.cwd())
    logger.info(f"Workdir : {cfg.paths.work}.")
    logger.info(f"Job id : {cfg.job_id}.")

if __name__ == "__main__":
    try:
        main_except()
    except:
        logger.exception("Failed this error:")
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
    finally:
        wandb.finish()