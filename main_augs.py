"""Entry point to compute the loss decomposition for different models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

import pdb
from functools import partial

from utils.predictor import RepresentorPredictor

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
import hubconf
from utils.tune_hyperparam import tune_hyperparam_
from main import begin, instantiate_datamodule_nofeature_, run_component_, save_results
from torchvision import transforms

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

    ############## PREPARE ##############
    representor, preprocess = hubconf.__dict__[cfg.representor]()
    Predictor = partial(RepresentorPredictor, representor=representor)

    datamodule = instantiate_datamodule_nofeature_(cfg, None,
                                                   train_transform=get_train_transform(preprocess),
                                                   test_transform=preprocess)
    datamodule.z_dim = hubconf.metadata_dict()[cfg.representor]["representation"]["z_dim"]

    ############## DOWNSTREAM PREDICTOR ##############
    results = dict()

    assert cfg.predictor.is_tune_hyperparam
    # those components can have the same hyperparameters
    components2hypopt = {"train-cmplmnt-ntest_train-sbst-ntest": dict(train_on="train-sbst-0.5", validate_on="train-cmplmnt-0.5", label_size=0.2),
                         "train_test": dict(train_on="train-sbst-0.5", validate_on="train-cmplmnt-0.5", label_size=0.2),
                         }

    # those components have the same training setup so don't retrain
    components_same_train = {}

    # TODO for train_train should simply copy from torch no aug

    if cfg.is_supervised:
        # only need train on train for supervised baselines (i.e. approx error) and train on test (agg risk)
        components = ["train_test"]
    else:
        # test should be replaced by test-cmplmnt-0.1
        components = ["train_test",
                      "train-cmplmnt-ntest_train-sbst-ntest",
                    ]

    for component in components:

        sffx_hypopt = "hyp_{train_on}_{validate_on}_{label_size}".format(**components2hypopt[component])
        tune_hyperparam_(datamodule, cfg,
                         tuning_path=cfg.paths.tuning + sffx_hypopt,
                         Predictor=Predictor,
                         **components2hypopt[component])

        results = run_component_(component, datamodule, cfg, results, components_same_train,
                                 results_path=cfg.paths.results + sffx_hypopt,
                                 Predictor=Predictor)

    # save results
    results = pd.DataFrame.from_dict(results)

    save_results(cfg, results, "all")


def get_train_transform(preprocess, is_fixed=True):
    # use the same normalization and interpolation as for eval
    # same augmentation for all models. We use cropping as this is used for all models
    normalization = [t for t in preprocess.transforms if isinstance(t, transforms.Normalize)][0]
    resize = [t for t in preprocess.transforms if isinstance(t, transforms.Resize)][0]
    return transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=resize.interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization,
    ])


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