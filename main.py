"""Entry point to compute the loss decomposition for differen models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

import copy
import logging
from pathlib import Path
import os
from collections.abc import Callable

import pandas as pd
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
import omegaconf
from omegaconf import Container, OmegaConf

from utils.cluster import nlp_cluster
from utils.data import get_Datamodule
from utils.helpers import (SklearnTrainer, get_torch_trainer, log_dict, namespace2dict, omegaconf2namespace,
                           NamespaceMap, replace_keys)
from pretrained import load_representor
from utils.predictor import Predictor

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
    representor, preprocess = load_representor(**cfg.representor.kwargs)
    datamodule = instantiate_datamodule_(cfg, representor, preprocess)

    ############## DOWNSTREAM PREDICTOR ##############
    results = dict()
    for component in ["train_risk", "subset_risk_01", "subset_risk_001"]:
        logger.info(f"Stage : {component}")
        cfg_comp = set_component_(datamodule, cfg, component)

        if cfg.predictor.is_sklearn:
            dict_cfgp = namespace2dict(cfg_comp.predictor)
            predictor = instantiate(dict_cfgp["model"])
            trainer = SklearnTrainer(cfg_comp)
        else:
            predictor = Predictor(cfg_comp, datamodule.z_dim, datamodule.n_labels)
            trainer = get_torch_trainer(cfg_comp)

        fit_(trainer, predictor, datamodule, cfg_comp)

        logger.info(f"Evaluate predictor for {component} ...")
        results[component] = evaluate(trainer, datamodule, cfg_comp)

        if component == "train_risk":
            # also add the standard risk
            logger.info(f"Evaluate predictor for std_risk ...")
            _ = set_component_(datamodule, cfg, "std_risk")
            results["std_risk"] = evaluate(trainer, datamodule, cfg_comp)

        elif component == "subset_risk_01":
            # also evaluate on test
            logger.info(f"Evaluate predictor for subset_test_risk_01 ...")
            _ = set_component_(datamodule, cfg, "subset_test_risk_01")
            results["subset_test_risk_01"] = evaluate(trainer, datamodule, cfg_comp)

        elif component == "subset_risk_001":
            logger.info(f"Evaluate predictor for subset_test_risk_001 ...")
            _ = set_component_(datamodule, cfg, "subset_test_risk_001")
            results["subset_test_risk_001"] = evaluate(trainer, datamodule, cfg_comp)

    # save results
    results = pd.DataFrame.from_dict(results)
    # cannot compute decodability at theis point because need to estimate approximation error
    # which is easier by checking simply online train supervised performance
    results["pred_gen_01"] = results["subset_risk_01"] - results["train_risk"]
    results["pred_gen_001"] = results["subset_risk_001"] - results["train_risk"]
    results["enc_gen_01"] = results["subset_test_risk_01"] - results["subset_risk_01"]
    results["enc_gen_001"] = results["subset_test_risk_001"] - results["subset_risk_001"]
    cfg.component = "all"
    save_results(cfg, results)

def begin(cfg: Container) -> None:
    """Script initialization."""
    pl.seed_everything(cfg.seed)
    cfg.paths.work = str(Path.cwd())
    logger.info(f"Workdir : {cfg.paths.work}.")

def instantiate_datamodule_(cfg: Container, representor : Callable, preprocess: Callable ) -> pl.LightningDataModule:
    """Instantiate dataset."""
    cfgd = omegaconf2namespace(cfg.data)
    cfgd.kwargs.dataset_kwargs.transform = preprocess
    Datamodule = get_Datamodule(cfgd.name)
    datamodule = Datamodule(representor=representor, representor_name=cfg.representor.name, **cfgd.kwargs)
    return datamodule

def set_component_(datamodule : pl.LightningDataModule, cfg: Container, component: str) -> NamespaceMap:
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

    # you don't compute approximation error here. For many benchmarks and architecture can find those online
    # + full supervised training on imagenet is slow
    if component == "train_risk":
        datamodule.reset(is_test_on_train=True)

    elif component == "subset_risk_01":
        datamodule.reset(is_test_nonsubset_train=True, subset_train_size=0.1)

    elif component == "subset_risk_001":
        datamodule.reset(is_test_nonsubset_train=True, subset_train_size=0.01) # DEV

    elif component == "subset_test_risk_01":
        datamodule.reset(is_test_nonsubset_train=False, subset_train_size=0.1)

    elif component == "subset_test_risk_001":
        datamodule.reset(is_test_nonsubset_train=False, subset_train_size=0.01) # DEV

    elif component == "test_risk":
        datamodule.reset(is_train_on_test=True)

    elif component == "std_risk":
        datamodule.reset(is_train_on_test=True)

    else:
        raise ValueError(f"Unknown component={component}.")

    return omegaconf2namespace(cfg)


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
) -> pd.Series:
    """Evaluate the trainer by logging all the metrics from the test set from the best model."""
    eval_dataloader = datamodule.test_dataloader()
    test_res = trainer.test(dataloaders=eval_dataloader, ckpt_path="best")[0]
    log_dict(trainer, test_res, is_param=False)
    results = replace_keys(test_res, f"test/{cfg.component}/", "", is_prfx=True)

    results = pd.Series(results)
    save_results(cfg, results)

    return results

def save_results(cfg, results):
    results_path = Path(cfg.paths.results)
    results_path.mkdir(parents=True, exist_ok=True)
    filename = RESULTS_FILE.format(component=cfg.component)
    path = results_path / filename
    results.to_csv(path, header=True, index=True)
    logger.info(f"Logging results to {path}.")

if __name__ == "__main__":
    main_except()
