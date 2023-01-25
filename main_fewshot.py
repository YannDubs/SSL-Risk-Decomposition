"""Entry point to compute the loss decomposition for different models.

This should be called by `python main_fewshot.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main_fewshot.py -h`.
"""


from __future__ import annotations

try:
    from sklearnex import patch_sklearn

    patch_sklearn(["LogisticRegression"])
except:
    # tries to speedup sklearn if possible (has to be before import sklearn)
    pass

import copy
import logging
import traceback
from pathlib import Path
import os
import sys
from collections.abc import Callable
from typing import Union
from timeit import default_timer as timer

import pandas as pd
import hydra
import torch
import pytorch_lightning as pl
import omegaconf
from omegaconf import Container

from utils.cluster import nlp_cluster
from utils.data import get_Datamodule
from utils.helpers import (LightningWrapper, SklearnTrainer, check_import, get_torch_trainer, log_dict,
                           omegaconf2namespace,
                           NamespaceMap)
import hubconf
from utils.predictor import Predictor, get_sklearn_predictor
from utils.tune_hyperparam import tune_hyperparam_

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
    logger.info(f"Representing data with {cfg.representor}")
    representor, preprocess = hubconf.__dict__[cfg.representor]()
    representor = LightningWrapper(representor)
    datamodule = instantiate_datamodule_(cfg, representor, preprocess)

    if cfg.data.subset is not None:
        cfg.data.name = f"{cfg.data.name}-S{cfg.data.subset}"
        sffx = f"balsbst-ntrain{cfg.data.subset}"  # use percentage of ntrain even for ntest

    elif cfg.data.n_per_class is not None:
        cfg.data.name = f"{cfg.data.name}-N{cfg.data.n_per_class}"
        sffx = f"nperclass-{cfg.data.n_per_class}"

    else:
        raise ValueError("To use `main_fewshot.py` you should either set data.subset or data.n")

    n_sbst = len(datamodule.get_dataset("train-" + sffx))
    n_testUsbst = len(datamodule.test_dataset) + n_sbst

    ############## DOWNSTREAM PREDICTOR ##############
    results = dict()

    # those components have the same training setup so don't retrain
    if cfg.is_riskdec:
        components_same_train = {f"train-{sffx}_train-{sffx}": [f"train-{sffx}_test",
                                                                f"train-{sffx}_train-balsbst-ntest"],
                                 }
    else:
        components_same_train = {f"train-{sffx}_train-{sffx}": [f"train-{sffx}_test"]}

    if cfg.predictor.is_tune_hyperparam:
        # should be sklearn
        train = f"train-{sffx}"
        valid = train if cfg.predictor.hypopt.is_tune_on_train else f"train-balsbst-ntest-11"
        tune_hyperparam_(datamodule, cfg, train_on=train, validate_on=valid)

    if cfg.is_supervised:
        if cfg.is_riskdec:
            # only need train on train for supervised baselines (i.e. approx error) and train on test (agg risk)
            components = [f"train-{sffx}_train-{sffx}",
                          f"train-balsbst-{n_testUsbst}_train-balsbst-{n_testUsbst}"]
        else:
            components = [f"train-{sffx}_train-{sffx}"]

    else:
        if cfg.is_riskdec:
            components = [f"train-balsbst-{n_testUsbst}_train-balsbst-{n_testUsbst}",
                          f"train-{sffx}_train-{sffx}",
                          f"test-{sffx}_test-{sffx}"]
        else:
            components = [f"train-{sffx}_train-{sffx}"]

    for component in components:
        results = run_component_(component, datamodule, cfg, results, components_same_train)

    # save results
    results = pd.DataFrame.from_dict(results)

    save_results(cfg, results, "all")


    # remove all saved features at the end
    # remove_rf(datamodule.features_path, not_exist_ok=True)

def begin(cfg: Container) -> None:
    """Script initialization."""
    pl.seed_everything(cfg.seed)
    cfg.paths.work = str(Path.cwd())

    if cfg.is_log_wandb:
        cfg.wandb_kwargs.id = "" + cfg.wandb_kwargs.id
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


def instantiate_datamodule_nofeature_(cfg: Container,
                                        preprocess: Callable,
                                        **kwargs) -> pl.LightningDataModule:
    """Instantiate dataset."""
    data_kwargs = omegaconf2namespace(cfg.data.kwargs)
    data_kwargs.dataset_kwargs.transform = preprocess
    Datamodule = get_Datamodule(cfg.data.name)
    datamodule = Datamodule(**data_kwargs, **kwargs)

    if cfg.data.check_length is not None:
        check_length = cfg.data.check_length
        assert datamodule.len_train > check_length, f"Training set supposed to be at least {check_length} long but it is {datamodule.len_train} long."

    return datamodule


def instantiate_datamodule_(cfg: Container, representor : Callable, preprocess: Callable, **kwargs) -> pl.LightningDataModule:
    """Instantiate dataset."""
    return instantiate_datamodule_nofeature_(cfg, preprocess, representor=representor, representor_name=cfg.representor, **kwargs)


def run_component_(component : str, datamodule : pl.LightningDataModule, cfg : Container, results : dict,
                   components_same_train : dict ={}, Predictor=Predictor, results_path=None,
                   reset_kwargs : dict ={}, **kwargs):
    logger.info(f"Stage : {component}")

    cfg_comp, datamodule = set_component_(datamodule, cfg, component, **reset_kwargs)

    if cfg.predictor.is_sklearn:
        predictor = get_sklearn_predictor(cfg_comp)
        trainer = SklearnTrainer(cfg_comp)
    else:
        predictor = Predictor(cfg_comp, datamodule.z_dim, datamodule.n_labels)
        trainer = get_torch_trainer(cfg_comp)

    try:
        # try to load in case already precomputed
        try:
            results[component] = load_results(cfg_comp, component, results_path=results_path)
        except:  # DEV TEMPORARY TO REMOVE! tries to load models before adding hyp
            results[component] = load_results(cfg_comp, component, results_path=None)
            save_results(cfg_comp, results[component], component, results_path=results_path)

        for other_comp in components_same_train.get(component, []):
            results[other_comp] = load_results(cfg_comp, other_comp, results_path=results_path)
        logger.info(f"Skipping {component} as already computed ...")

    except FileNotFoundError:

        start = timer()
        fit_(trainer, predictor, datamodule, cfg_comp)
        end = timer()
        logger.info(f"Fitted {component} in {end - start} seconds.")

        logger.info(f"Evaluate predictor for {component} ...")
        results[component] = evaluate(trainer, datamodule, cfg_comp, component, model=predictor, **kwargs)

        # evaluate component with same train
        for other_comp in components_same_train.get(component, []):
            logger.info(f"Evaluate predictor for {other_comp} without retraining ...")
            cfg_comp, datamodule = set_component_(datamodule, cfg, other_comp)
            trainer = set_component_trainer_(trainer, cfg_comp, other_comp, model=predictor)
            results[other_comp] = evaluate(trainer, datamodule, cfg_comp, other_comp, model=predictor, results_path=results_path, **kwargs)

    return results

def set_component_(datamodule : pl.LightningDataModule,
                   cfg: Container,
                   component: str,
                   **reset_kwargs) -> tuple[NamespaceMap, pl.LightningDataModule]:
    """Set the current component to evaluate."""
    cfg = copy.deepcopy(cfg)  # not inplace

    with omegaconf.open_dict(cfg):
        cfg.component = component

    file_end = Path(cfg.paths.logs) / f"{FILE_END}"
    if file_end.is_file():
        logger.info(f"Skipping most of {component} as {file_end} exists.")

    # make sure all paths exist
    for name, path in cfg.paths.items():
        if (name == "data") and ("is_avoid_raw_dataset" in cfg.data.kwargs) and cfg.data.kwargs.is_avoid_raw_dataset:
            continue
        Path(path).mkdir(parents=True, exist_ok=True)

    separator_tr_te = "_"
    is_train_on, is_test_on = component.split(separator_tr_te)
    datamodule.reset(is_train_on=is_train_on, is_test_on=is_test_on, **reset_kwargs)

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
    datamodule: pl.LightningDataModule,
    cfg: NamespaceMap,
    component: str,
    model: torch.nn.Module,
    is_per_task_results=False,
    results_path=None,
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
    save_results(cfg, results, component, results_path=results_path)

    return results


def save_results(cfg : NamespaceMap, results : Union[pd.Series,pd.DataFrame], component: str, results_path=None):
    cfg = copy.deepcopy(cfg)
    cfg.component = component

    if results_path is None:
        results_path = cfg.paths.results

    results_path = Path(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    filename = RESULTS_FILE.format(component=cfg.component)
    path = results_path / filename
    results.to_csv(path, header=True, index=True)
    logger.info(f"Logging {component} results to {path}.")


def load_results(cfg: NamespaceMap, component: str, results_path=None) -> Union[pd.Series,pd.DataFrame]:
    cfg = copy.deepcopy(cfg)
    cfg.component = component

    if results_path is None:
        results_path = cfg.paths.results

    results_path = Path(results_path)
    filename = RESULTS_FILE.format(component=cfg.component)
    path = results_path / filename
    logger.info(f"Trying to load results from {path}")
    return pd.read_csv(path, index_col=0).squeeze("columns")



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