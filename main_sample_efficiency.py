"""Entry point to compute the loss decomposition for differen models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations


import logging
import traceback
import os
import sys

import pandas as pd
import torch
from torch.nn import functional as F
import hydra
from pathlib import Path

from utils.cluster import nlp_cluster
from pretrained import load_representor
from utils.predictor import Predictor
from utils.architectures import get_Architecture

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[1]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)

from main import begin, instantiate_datamodule_, run_component_, save_results  # isort:skip

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
    components_same_train = {"train-sbst-0.9_test": ["train-sbst-0.9_train-cmplmnt-0.9"],
                             "train-sbst-0.3_test": ["train-sbst-0.3_train-cmplmnt-0.3"],
                             "train-sbst-0.1_test": ["train-sbst-0.1_train-cmplmnt-0.1"],
                             "train-sbst-0.03_test": ["train-sbst-0.03_train-cmplmnt-0.03"],
                             "train-sbst-0.01_test": ["train-sbst-0.01_train-cmplmnt-0.01"],
                             "train-sbst-0.007_test": ["train-sbst-0.005_train-cmplmnt-0.007"],
                             "train-sbst-0.005_test": ["train-sbst-0.007_train-cmplmnt-0.005"],
                             "train-sbst-0.003_test": ["train-sbst-0.003_train-cmplmnt-0.003"],
                             "train-sbst-0.002_test": ["train-sbst-0.002_train-cmplmnt-0.002"],
                             "train-sbst-0.001_test": ["train-sbst-0.001_train-cmplmnt-0.001"]
                             }
    components = [ "train_test",
                  "train-sbst-0.9_test",
                  "train-sbst-0.3_test",
                  "train-sbst-0.1_test",
                  "train-sbst-0.03_test",
                  "train-sbst-0.01_test",
                  "train-sbst-0.007_test",
                  "train-sbst-0.005_test",
                  "train-sbst-0.003_test",
                 "train-sbst-0.002_test",
                  "train-sbst-0.001_test",
                  ]

    for component in components:
        results = run_component_(component, datamodule, cfg, results, components_same_train)

    # save results
    results = pd.DataFrame.from_dict(results)

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