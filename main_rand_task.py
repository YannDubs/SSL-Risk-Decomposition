"""Entry point to compute the loss decomposition for differen models.

This should be called by `python main.py <conf>` where <conf> sets all configs from the cli, see
the file `config/main.yaml` for details about the configs. or use `python main.py -h`.
"""


from __future__ import annotations

import copy
import logging
import time
import traceback
import os
import sys
import gc

import pandas as pd
import torch
from einops import einops
from torch.nn import functional as F
import hydra
from pathlib import Path
import pytorch_lightning as pl

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
    for i in range(cfg.rand_task.n_runs):
        cfg.seed = cfg.seed + 1
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
        components_same_train = {"train_train": ["train_test"]}

        for component in cfg.rand_task.components:
            results = run_component_(component,
                                     datamodule,
                                     cfg,
                                     results,
                                     components_same_train,
                                     Predictor=RandTaskPredictor,
                                     is_per_task_results=True)

        # save results
        results = pd.DataFrame.from_dict(results)

        save_results(cfg, results, "all")

        del datamodule
        del representor
        pl.utilities.memory.garbage_collection_cuda()


class RandTaskPredictor(Predictor):
    def __init__( self, hparams, z_dim, n_labels):
        super().__init__(hparams, z_dim, n_labels)

        Architecture = get_Architecture(**self.hparams.predictor.arch_kwargs)
        n_rand_tasks = self.hparams.rand_task.n_rand_tasks
        kway = self.hparams.rand_task.n_label_per_task
        if kway == 2:
            n_logits = n_rand_tasks
        else:
            n_logits = n_rand_tasks * kway

        self.predictor = Architecture(self.z_dim, self.n_labels + n_logits)

        n_Mx = self.hparams.data.n_train if self.hparams.data.kwargs.is_add_idx else self.n_labels
        logger.info(f"n_Mx={n_Mx}.")
        self.register_buffer("agg_tgt_mapper", torch.randint(high=kway, size=(n_Mx, n_rand_tasks)).to(torch.int16 ))

        if kway == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss()

    def loss(self, Y_hat, Mx, return_acc_per_task=False):
        """Compute the MSE or cross entropy loss."""
        n_rand_tasks = self.hparams.rand_task.n_rand_tasks
        kway = self.hparams.rand_task.n_label_per_task

        if self.hparams.data.kwargs.is_add_idx:
            y_rand = self.agg_tgt_mapper[Mx[:,1]]
            y = Mx[:,0]
        else:
            y = Mx
            y_rand = self.agg_tgt_mapper[Mx]

        Mx_logits = Y_hat[:, :self.n_labels]
        rand_logits = Y_hat[:, self.n_labels :]

        loss_Mx = F.cross_entropy(Mx_logits, y.squeeze().long())

        if kway == 2:
            loss_rand = self.criterion(rand_logits, y_rand.float())
            y_rand_hat = (rand_logits > 0)

        else:
            rand_logits = einops.rearrange(rand_logits, "b (l k) -> (b l) k", l=n_rand_tasks, k=kway)
            y_rand = einops.rearrange(y_rand, "b l -> (b l)")
            loss_rand = F.cross_entropy(rand_logits, y_rand.long())

            y_rand_hat = rand_logits.argmax(dim=-1)
            y_rand_hat = einops.rearrange(y_rand_hat, "(b l) -> b l", l=n_rand_tasks)
            y_rand = einops.rearrange(y_rand, "(b l) -> b l", l=n_rand_tasks)


        if self.hparams.rand_task.is_sum_tasks:
            # use sum over tasks so that gradient magnitude independent of number of outputs
            loss_rand = loss_rand * n_rand_tasks

        loss = loss_Mx + loss_rand

        logs = dict()
        logs["acc_Mx"] = (Mx_logits.argmax(dim=-1) == y).float().mean()
        logs["loss_Mx"] = loss_Mx

        acc_per_task = (y_rand_hat.float() == y_rand).float().mean(0)
        logs["acc_rand"] = acc_per_task.mean()
        logs["loss_rand"] = loss_rand

        other = dict(acc_per_task = acc_per_task.detach().cpu())

        return loss, logs, other

    def shared_step(self, batch, mode, *args, return_acc_per_task=False, **kwargs):
        loss, logs, other = self.step(batch)

        name = f"{mode}/{self.hparams.data.name}/{self.hparams.component}"
        self.log_dict(
            {
                f"{name}/{k}": v
                for k, v in logs.items()
            },
        )

        if return_acc_per_task:
            return loss, other["acc_per_task"]
        else:
            return loss

    def predict_step(self, batch, *args, **kwargs):
        loss, logs, other = self.step(batch)  # don't use shared_step becausec can't log in predict step
        return other["acc_per_task"].cpu()

    def test_step(self, batch, *args, **kwargs):
        loss, return_acc_per_task = self.shared_step(batch, "test", *args, return_acc_per_task=True, **kwargs)
        return loss, return_acc_per_task

    def test_epoch_end(self, outputs):
        _, acc_per_task = zip(*outputs)
        acc_rand_worst = (sum(acc_per_task) / len(acc_per_task)).min()
        self.log("test/acc_rand_worst", acc_rand_worst)

if __name__ == "__main__":
    try:
        main_except()
    except:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)
    finally:
        wandb.finish()