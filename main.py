"""Entry point to compute the loss decomposition for differen models.

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
from hydra.utils import instantiate
import pytorch_lightning as pl
import omegaconf
from omegaconf import Container, OmegaConf

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
                             "train-sbst-ntest_train-cmplmnt-ntest": ["train-sbst-ntest_test"],
                             "train-sbst-0.1_train": ["train-sbst-0.1_test"]}

    if not cfg.is_only_robustness:
        for component in [ "train_train", "train-sbst-ntest_train-cmplmnt-ntest", "train-sbst-0.1_train"]:
            results = run_component_(component, datamodule, cfg, results, components_same_train)

        # save results
        results = pd.DataFrame.from_dict(results)
        # cannot compute decodability at theis point because need to estimate approximation error
        # which is easier by checking simply online train supervised performance
        results["pred_gen"] = results["train-sbst-ntest_train-cmplmnt-ntest"] - results["train_train"]
        results["enc_gen"] = results["train-sbst-ntest_test"] - results["train-sbst-ntest_train-cmplmnt-ntest"]
        results["pred_gen_01"] = results["train-sbst-0.1_train"] - results["train_train"]
        results["enc_gen_01"] = results["train-sbst-0.1_test"] - results["train-sbst-0.1_train"]
        save_results(cfg, results, "all")

    # Only if want robustness results
    for rob_dataset in cfg.robustness_datasets:
        run_robustness_decomposition(rob_dataset, datamodule, cfg, representor, preprocess)

    # remove all saved features at the end
    #remove_rf(datamodule.features_path, not_exist_ok=True)


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

def init_wandb(offline=False, **kwargs):
    """Initializae wandb while accepting param of pytorch lightning."""
    check_import("wandb", "wandb")

    if offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        resume="allow",
        **kwargs,
    )

def instantiate_datamodule_(cfg: Container, representor : Callable, preprocess: Callable, **kwargs) -> pl.LightningDataModule:
    """Instantiate dataset."""
    cfgd = omegaconf2namespace(cfg.data)
    cfgd.kwargs.dataset_kwargs.transform = preprocess
    Datamodule = get_Datamodule(cfgd.name)
    datamodule = Datamodule(representor=representor, representor_name=cfg.representor.name, **cfgd.kwargs, **kwargs)
    return datamodule

def run_component_(component : str, datamodule : pl.LightningDataModule, cfg : Container, results : dict,
                   components_same_train : dict ={}):
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
        results[component] = evaluate(trainer, datamodule, cfg_comp, component)

        # evaluate component with same train
        for other_comp in components_same_train.get(component, []):
            logger.info(f"Evaluate predictor for {other_comp} without retraining ...")
            cfg_comp, datamodule = set_component_(datamodule, cfg, other_comp)
            trainer = set_component_trainer_(trainer, cfg_comp, other_comp)
            results[other_comp] = evaluate(trainer, datamodule, cfg_comp, other_comp)

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

    if not cfg.predictor.is_sklearn:
        if len(datamodule.get_train_dataset()) < len(datamodule.train_dataset) // 5:
            # if training on very small subset multiply by 5 n epochs
            cfg.trainer.max_epochs = cfg.trainer.max_epochs * 5

    return omegaconf2namespace(cfg), datamodule

def set_component_trainer_(trainer: pl.Trainer, cfg: NamespaceMap, component: str):
    hparams = copy.deepcopy(cfg)
    hparams.component = component

    if cfg.predictor.is_sklearn:
        trainer.hparams = hparams
    else:
        trainer.lightning_module.hparams = hparams

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
    component: str
) -> pd.Series:
    """Evaluate the trainer by logging all the metrics from the test set from the best model."""
    cfg = copy.deepcopy(cfg)
    cfg.component = component

    eval_dataloader = datamodule.test_dataloader()
    results = trainer.test(dataloaders=eval_dataloader, ckpt_path=None)[0]
    log_dict(trainer, results, is_param=False)
    # only keep the metric
    results = { k.split("/")[-1]: v for k, v in results.items() }

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
    logger.info(f"Logging results to {path}.")

def run_robustness_decomposition(rob_dataset : str, old_datamodule: pl.LightningDataModule, cfg: Container,
                                 representor : Callable, preprocess: Callable):
    """Run the robustness decomposition."""
    logger.info("Stage : Robustness")

    main_data = cfg.data.name
    cfg = copy.deepcopy(cfg)  # not inplace
    results = dict()

    # for loading test set use robustness data only, then use old training set
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
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)