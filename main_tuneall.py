"""Entry point to compute the loss decomposition for different models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

import pdb

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

    ############## DOWNSTREAM PREDICTOR ##############
    results = dict()

    assert cfg.predictor.is_tune_hyperparam
    # those components can have the same hyperparameters
    components2hypopt = {"train_train": dict(train_on="train-sbst-0.5",  validate_on="train-sbst-0.5", label_size=0.2),
                         "train-cmplmnt-ntest_train-sbst-ntest": dict(train_on="train-sbst-0.5", validate_on="train-cmplmnt-0.5", label_size=0.2),
                         "train_test": dict(train_on="train-sbst-0.5", validate_on="test", label_size=0.2),   # valdiation should be done on test-sbst-0.1
                         "train_test-cmplmnt-0.1": dict(train_on="train", validate_on="test-sbst-0.1"),
                         "union_test": dict(train_on="train-sbst-0.5", validate_on="train-sbst-0.5", label_size=0.2),
                         }

    # those components have the same training setup so don't retrain
    components_same_train = {}

    if cfg.is_supervised:
        # only need train on train for supervised baselines (i.e. approx error) and train on test (agg risk)
        components = ["train_train",
                      "train_test"
                      #"train_test-cmplmnt-0.1",
                      ]
    else:
        # test should be replaced by test-cmplmnt-0.1
        components = ["train_train",
                      "train_test",
                      "train-cmplmnt-ntest_train-sbst-ntest"]

        if cfg.is_alternative_decomposition:
            components += ["union_test"]
    for component in components:

        sffx_hypopt = "hyp_{train_on}_{validate_on}_{label_size}".format(**components2hypopt[component])
        tune_hyperparam_(datamodule, cfg,
                         tuning_path=cfg.paths.tuning + sffx_hypopt,
                         **components2hypopt[component])

        results = run_component_(component, datamodule, cfg, results, components_same_train,
                                 results_path=cfg.paths.results + sffx_hypopt)

    # save results
    results = pd.DataFrame.from_dict(results)

    save_results(cfg, results, "all")

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