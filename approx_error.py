"""Compute the supervised performance for approximation error."""

from __future__ import annotations

import logging
import os

import pandas as pd
import numpy as np
import hydra
import scipy
from sklearn.metrics import log_loss, accuracy_score

from main import instantiate_datamodule_, begin, save_results
from utils.cluster import nlp_cluster
from pretrained import load_representor
from utils.helpers import remove_rf

logger = logging.getLogger(__name__)

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

    results = dict()
    datasets = dict(train=datamodule.get_train_dataset(), test=datamodule.get_test_dataset())
    for mode, data in datasets.items():
        # in the supervised case the representation is the logits
        y_pred = scipy.special.softmax(data.X.astype(np.double), axis=1)
        y = data.Y

        results[mode] = dict()
        results[mode]["acc"] = accuracy_score(y, np.argmax(y_pred, axis=-1))
        results[mode]["err"] = 1 - results[mode]["acc"]
        results[mode]["loss"] = log_loss(y, y_pred)

    # save results
    cfg.component = "all"
    results = pd.DataFrame.from_dict(results)
    save_results(cfg, results)

    # remove all saved features at the end
    #remove_rf(datamodule.features_path, not_exist_ok=True)


if __name__ == "__main__":
    main_except()
