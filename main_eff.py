"""Entry point to compute the loss decomposition for different models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

try:
    from sklearnex import patch_sklearn

    patch_sklearn(["LogisticRegression"])
except:
    # tries to speedup sklearn if possible (has to be before import sklearn)
    pass

import logging
import traceback
import os
import sys

import pandas as pd
import hydra

from utils.cluster import nlp_cluster
from utils.helpers import LightningWrapper
import hubconf
from utils.tune_hyperparam import tune_hyperparam_
from main import begin, instantiate_datamodule_, run_component_, save_results

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
        raise ValueError("To use main_eff you should either set data.subset or data.n")

    n_sbst = len(datamodule.get_dataset("train-" + sffx))
    n_testUsbst = len(datamodule.test_dataset) + n_sbst

    ############## DOWNSTREAM PREDICTOR ##############
    results = dict()

    # those components have the same training setup so don't retrain
    components_same_train = {f"train-{sffx}_train-{sffx}": [f"train-{sffx}_test",
                                                            f"train-{sffx}_train-balsbst-ntest"],
                             }

    if cfg.predictor.is_tune_hyperparam:
        # should be sklearn
        train = f"train-{sffx}"
        valid = train if cfg.predictor.hypopt.is_tune_on_train else f"train-balsbst-ntest-11"
        tune_hyperparam_(datamodule, cfg, train_on=train, validate_on=valid)

    if cfg.is_supervised:
        # only need train on train for supervised baselines (i.e. approx error) and train on test (agg risk)
        components = [f"train-{sffx}_train-{sffx}",
                      f"train-balsbst-{n_testUsbst}_train-balsbst-{n_testUsbst}"
                      ]
    else:
        components = [f"train-balsbst-{n_testUsbst}_train-balsbst-{n_testUsbst}",
                      f"train-{sffx}_train-{sffx}",
                      f"test-{sffx}_test-{sffx}"]

    for component in components:
        results = run_component_(component, datamodule, cfg, results, components_same_train)

    # save results
    results = pd.DataFrame.from_dict(results)

    save_results(cfg, results, "all")


    # remove all saved features at the end
    # remove_rf(datamodule.features_path, not_exist_ok=True)



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