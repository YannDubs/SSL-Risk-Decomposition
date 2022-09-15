"""Entry point to compute the loss decomposition for different models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations


import copy
import logging
import traceback
from pathlib import Path
import os
import sys
from collections.abc import Callable
from typing import Union

import pandas as pd
import hydra
import torch
from hydra.utils import instantiate
import pytorch_lightning as pl
import omegaconf
from omegaconf import Container

from utils.cluster import nlp_cluster
from utils.data import get_Datamodule
from utils.helpers import (SklearnTrainer, check_import, get_torch_trainer, log_dict, namespace2dict,
                           omegaconf2namespace,
                           NamespaceMap, remove_rf)
from pretrained import load_representor
from utils.predictor import Predictor

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

    ############## REPRESENT DATA ##############
    logger.info("Stage : Representor")
    representor, preprocess = load_representor(cfg.representor.name, **cfg.representor.kwargs)
    datamodule = instantiate_datamodule_(cfg, representor, preprocess)

    ############## DOWNSTREAM PREDICTOR ##############
    results = dict()

    # those components have the same training setup so don't retrain
    components_same_train = {"train_train": ["train_test"],
                             "train-cmplmnt-ntest_train-sbst-ntest": ["train-cmplmnt-ntest_test"],
                             }

    if cfg.is_run_in_dist:

        if cfg.is_supervised:
            # only need train on train for supervised baselines (i.e. approx error) and train on test (agg risk)
            components = ["train_train", "train-cmplmnt-ntest_train-sbst-ntest"]
        else:
            components = ["train_train", "train-cmplmnt-ntest_train-sbst-ntest",
                          "union_test"]  # for other ~equivalent decomposition.

        for component in components:
            results = run_component_(component, datamodule, cfg, results, components_same_train)

        # save results
        results = pd.DataFrame.from_dict(results)

        if not cfg.is_supervised:
            # cannot compute decodability at theis point because need to estimate approximation error
            # which is easier by checking simply online train supervised performance
            results["pred_gen"] = results["train-cmplmnt-ntest_train-sbst-ntest"] - results["train_train"]
            results["enc_gen"] = results["train-cmplmnt-ntest_test"] - results["train-cmplmnt-ntest_train-sbst-ntest"]
            # this is for the other decomposition that nearly equivalent to the above. Choice is ~arbitrary.
            results["pred_gen_switched"] = results["union_test"] - results["train_train"]
            results["enc_gen_switched"] = results["train-cmplmnt-ntest_test"] - results["union_test"]

        save_results(cfg, results, "all")

    if cfg.is_run_out_dist:
        # Only if want out_dist results
        for rob_dataset in cfg.out_dist_datasets:
            run_out_dist_decomposition(rob_dataset, datamodule, cfg, representor, preprocess)

    # remove all saved features at the end
    # remove_rf(datamodule.features_path, not_exist_ok=True)


def begin(cfg: Container) -> None:
    """Script initialization."""
    pl.seed_everything(cfg.seed)
    cfg.paths.work = str(Path.cwd())

    if cfg.is_log_wandb:
        try:
            init_wandb(**cfg.wandb_kwargs)
        except Exception:
            init_wandb(offline=True, **cfg.wandb_kwargs)

    logger.info(f"Workdir : {cfg.paths.work}.")
    logger.info(f"Job id : {cfg.job_id}.")

def init_wandb(offline=False, **kwargs):
    """Initialize wandb while accepting param of pytorch lightning."""
    check_import("wandb", "wandb")

    if offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        resume="allow",
        **kwargs,
    )

def instantiate_datamodule_(cfg: Container, representor : Callable, preprocess: Callable, **kwargs) -> pl.LightningDataModule:
    """Instantiate dataset."""
    data_kwargs = omegaconf2namespace(cfg.data.kwargs)
    data_kwargs.dataset_kwargs.transform = preprocess
    Datamodule = get_Datamodule(cfg.data.name)
    datamodule = Datamodule(representor=representor, representor_name=cfg.representor.name, **data_kwargs, **kwargs)
    return datamodule

def run_component_(component : str, datamodule : pl.LightningDataModule, cfg : Container, results : dict,
                   components_same_train : dict ={}, Predictor=Predictor, **kwargs):
    logger.info(f"Stage : {component}")
    cfg_comp, datamodule = set_component_(datamodule, cfg, component)

    if cfg.predictor.is_sklearn:
        dict_cfgp = namespace2dict(cfg_comp.predictor)
        predictor = instantiate(dict_cfgp["model"])
        trainer = SklearnTrainer(cfg_comp)
    else:
        predictor = Predictor(cfg_comp, datamodule.z_dim, datamodule.n_labels)
        trainer = get_torch_trainer(cfg_comp)

    try:
        # try to load in case already precomputed
        results[component] = load_results(cfg_comp, component)
        for other_comp in components_same_train.get(component, []):
            results[other_comp] = load_results(cfg_comp, other_comp)
        logger.info(f"Skipping {component} as already computed ...")

    except FileNotFoundError:
        fit_(trainer, predictor, datamodule, cfg_comp)

        logger.info(f"Evaluate predictor for {component} ...")
        results[component] = evaluate(trainer, datamodule, cfg_comp, component, model=predictor, **kwargs)

        # evaluate component with same train
        for other_comp in components_same_train.get(component, []):
            logger.info(f"Evaluate predictor for {other_comp} without retraining ...")
            cfg_comp, datamodule = set_component_(datamodule, cfg, other_comp)
            trainer = set_component_trainer_(trainer, cfg_comp, other_comp, model=predictor)
            results[other_comp] = evaluate(trainer, datamodule, cfg_comp, other_comp, model=predictor, **kwargs)

    return results

def set_component_(datamodule : pl.LightningDataModule, cfg: Container, component: str) -> tuple[NamespaceMap, pl.LightningDataModule]:
    """Set the current component to evaluate."""
    cfg = copy.deepcopy(cfg)  # not inplace

    with omegaconf.open_dict(cfg):
        cfg.component = component

    file_end = Path(cfg.paths.logs) / f"{FILE_END}"
    if file_end.is_file():
        logger.info(f"Skipping most of {component} as {file_end} exists.")

    # make sure all paths exist
    for _, path in cfg.paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)

    separator_tr_te = "_"
    is_train_on, is_test_on = component.split(separator_tr_te)
    datamodule.reset(is_train_on=is_train_on, is_test_on=is_test_on)

    cfg.data.n_train = len(datamodule.get_train_dataset())

    if not cfg.predictor.is_sklearn:
        # if training on very small subset increase number of epochs (sqrt to avoid long++)
        orig_size = len(datamodule.train_dataset)
        factor = orig_size / cfg.data.n_train  # how many times larger is original
        if factor > 1:
            mult_factor = max(1, round(factor**0.5))
            cfg.trainer.max_epochs = cfg.trainer.max_epochs * mult_factor

    return omegaconf2namespace(cfg), datamodule

def set_component_trainer_(trainer: pl.Trainer, cfg: NamespaceMap, component: str, model: torch.nn.Module):
    hparams = copy.deepcopy(cfg)
    hparams.component = component

    if cfg.predictor.is_sklearn:
        trainer.hparams = hparams
    else:
        trainer.lightning_module.hparams.update(hparams)
        model.hparams.update(hparams)

    return trainer


def fit_(
    trainer: pl.Trainer,
    module: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    cfg: NamespaceMap,
):
    """Fit the module."""
    kwargs = dict()

    # Resume training ?
    last_checkpoint = Path(cfg.paths.checkpoint) / LAST_CHECKPOINT
    if last_checkpoint.exists():
        kwargs["ckpt_path"] = str(last_checkpoint)

    trainer.fit(module, datamodule=datamodule, **kwargs)

def evaluate(
    trainer: pl.Trainer,
    datamodule : pl.LightningDataModule,
    cfg: NamespaceMap,
    component: str,
    model : torch.nn.Module,
    is_per_task_results = False
) -> pd.Series:
    """Evaluate the trainer by logging all the metrics from the test set from the best model."""
    cfg = copy.deepcopy(cfg)
    cfg.component = component

    eval_dataloader = datamodule.test_dataloader()
    results = trainer.test(dataloaders=eval_dataloader, ckpt_path=None, model=model)[0]
    log_dict(trainer, results, is_param=False)
    # only keep the metric
    results = { k.split("/")[-1]: v for k, v in results.items() }

    if is_per_task_results:
        predict_dataloader = datamodule.predict_dataloader()
        per_task_results = trainer.predict(dataloaders=predict_dataloader, ckpt_path=None, model=model)
        per_task_results = torch.stack(per_task_results,dim=0).mean(0)
        for i in range(per_task_results.shape[0]):
            results[f"acc_{i}"] = per_task_results[i].item()

    results = pd.Series(results)
    save_results(cfg, results, component)

    return results

def load_results(cfg: NamespaceMap, component: str) -> Union[pd.Series,pd.DataFrame]:
    cfg = copy.deepcopy(cfg)
    cfg.component = component

    results_path = Path(cfg.paths.results)
    filename = RESULTS_FILE.format(component=cfg.component)
    path = results_path / filename
    return pd.read_csv(path, index_col=0).squeeze("columns")

def save_results(cfg : NamespaceMap, results : Union[pd.Series,pd.DataFrame], component: str):
    cfg = copy.deepcopy(cfg)
    cfg.component = component

    results_path = Path(cfg.paths.results)
    results_path.mkdir(parents=True, exist_ok=True)
    filename = RESULTS_FILE.format(component=cfg.component)
    path = results_path / filename
    results.to_csv(path, header=True, index=True)
    logger.info(f"Logging {component} results to {path}.")

def run_out_dist_decomposition(rob_dataset : str, old_datamodule: pl.LightningDataModule, cfg: Container,
                                 representor : Callable, preprocess: Callable):
    """Run the out_dist decomposition."""
    logger.info("Stage : Out of distribution")

    main_data = cfg.data.name
    cfg = copy.deepcopy(cfg)  # not inplace
    results = dict()

    # for loading test set use out_dist data only, then use old training set
    cfg.data.name = rob_dataset
    datamodule = instantiate_datamodule_(cfg, representor, preprocess, train_dataset=old_datamodule.train_dataset)
    cfg.data.name = f"{main_data}-{rob_dataset}"  # for saving you want to remember both datasets: train and test

    for component in ["test_test", "union_test", "train_test"]:
        results = run_component_(component, datamodule, cfg, results)

    # save results
    results = pd.DataFrame.from_dict(results)
    results["enc_gen"] = results["union_test"] - results["test_test"]
    results["pred_gen"] = results["train_test"] - results["union_test"]
    save_results(cfg, results, "all")

if __name__ == "__main__":
    try:
        main_except()
    except:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
    finally:
        wandb.finish()