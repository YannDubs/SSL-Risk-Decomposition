from optuna._callbacks import MaxTrialsCallback
from optuna.trial import TrialState

try:
    from sklearnex import patch_sklearn

    patch_sklearn(["LogisticRegression"])
except:
    # tries to speedup sklearn if possible (has to be before import sklearn)
    pass


import copy
from functools import partial
from pathlib import Path
import json

import numpy as np
import pytorch_lightning as pl
import logging

from hydra.utils import instantiate
from matplotlib import pyplot as plt
from pytorch_lightning.plugins.environments import SLURMEnvironment
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import namespace2dict, omegaconf2namespace
from utils.predictor import Predictor, get_sklearn_predictor
from utils.plotting import save_fig


try:
    import optuna
    from optuna.visualization.matplotlib import plot_param_importances, plot_parallel_coordinate, plot_optimization_history
except:
    pass

try:
    import wandb
except ImportError:
    pass

logger = logging.getLogger(__name__)
BEST_HPARAMS = "best_hparams.json"

def tune_hyperparam_(datamodule, cfg, train_on="train-sbst-0.5", validate_on="train-cmplmnt-0.5", label_size=None, tuning_path=None):
    logger.info(f"Hyperparameter tuning")

    if tuning_path is None:
        tuning_path = cfg.paths.tuning

    path_hypopt = Path(tuning_path)
    path_hypopt.mkdir(parents=True, exist_ok=True)
    path_best_hparams = path_hypopt / BEST_HPARAMS

    if path_best_hparams.is_file():
        logger.info(f"Hyperparameters already found at {path_best_hparams}")
        with open(path_best_hparams, 'r') as f:
            hparams = json.load(f)
            set_hparams_(cfg, datamodule, hparams)
        logger.info("Finished tuning")
        return

    datamodule.reset(is_train_on=train_on, is_test_on=validate_on, label_size=label_size)
    cfg.data.n_train = len(datamodule.get_train_dataset())

    if cfg.predictor.is_sklearn:
        hparams = tune_hyperparam_sklearn_(datamodule, cfg, path_hypopt)
    else:
        hparams = tune_hyperparam_torch_(datamodule, cfg, path_hypopt)

    # reset to default splits
    datamodule.reset()
    cfg.data.n_train = len(datamodule.get_train_dataset())

    set_hparams_(cfg, datamodule, hparams)

    logger.info(f"Saving best hparams at {path_best_hparams}")
    with open(path_best_hparams, 'w') as f:
        json.dump(hparams, f)

    logger.info("Finished tuning")


def tune_hyperparam_sklearn_(datamodule, cfg, path_hypopt):
    """Tune the hyperparameters for the sklearn probe and then loads them."""


    assert cfg.predictor.hypopt.metric == "err"

    cfgh = cfg.predictor.hypopt
    predictor = get_sklearn_predictor(cfg)

    # when pipeline (feature scaling) needs to change params to `clf__*`
    prfx = "clf__" if cfg.predictor.is_scale_features else ""

    param_space = dict()
    param_space[f"{prfx}C"] = loguniform(1e-6, 100)  # same param for svm and logistic

    train_dataset = datamodule.get_train_dataset()
    val_dataset = datamodule.get_test_dataset()

    X = np.concatenate((train_dataset.X, val_dataset.X))
    Y = np.concatenate((train_dataset.Y, val_dataset.Y))

    clf = RandomizedSearchCV(
        predictor,
        param_space,
        scoring=make_scorer(accuracy_score),
        n_jobs=None,  # avoids huge memory cost
        cv=PredefinedSplit(
            [-1] * len(train_dataset.X) + [0] * len(val_dataset.X)
        ),
        n_iter=cfgh.n_hyper,
        verbose=1,
        random_state=cfg.seed,
        refit=False
    )

    logging.info(f"Fitting sklearn on {len(train_dataset.X)} examples, validating on .{len(val_dataset.X)}. Clf={clf}.")
    clf = clf.fit(X, Y)
    return clf.best_params_



def tune_hyperparam_torch_(datamodule, cfg, path_hypopt):
    """Tune the hyperparameters for the torch probe and then loads them."""


    cfgh = cfg.predictor.hypopt
    Sampler = optuna.samplers.__dict__[cfgh.sampler]
    sampler = Sampler(**cfgh.kwargs_sampler)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(sampler=sampler,
                                direction="minimize",
                                storage=f"sqlite:///{(path_hypopt/ 'optuna.db').resolve()}",
                                study_name="main",
                                load_if_exists=True)

    for _, trial in cfgh.to_eval_first.items():
        # runs standard hparam to start with
        study.enqueue_trial(dict(**trial))  # need to be a dict

    study.optimize(partial(objective, cfg=cfg, datamodule=datamodule),
                   #n_trials=cfgh.n_hyper,
                   callbacks=[MaxTrialsCallback(cfgh.n_hyper, states=(TrialState.COMPLETE,))],
                   gc_after_trial=True)  # ensures memory not adding

    logger.info(f"Tuning duration: {study.trials_dataframe().duration.astype('timedelta64[m]').sum()} minutes")

    try:
        summarize_study(study, path_hypopt, is_wandb=cfg.is_log_wandb)
    except:
        logger.exception("Couldn't summarize study due to this error:")

    return study.best_trial.params


def objective(trial, cfg, datamodule):
    """Objective function to hyperparameter tune."""
    cfg = copy.deepcopy(cfg)
    cfg.trainer.enable_checkpointing = False  # no checkpointing during hypopt
    cfg.component = "hypertune"

    sp = cfg.predictor.hypopt.search_space

    hparams = dict()
    if "batch_size" in sp.to_tune:
        hparams["batch_size"] = trial.suggest_int("batch_size", sp.batch_size.min, sp.batch_size.max)
    if "lr" in sp.to_tune:
        hparams["lr"] = trial.suggest_float("lr", sp.lr.min, sp.lr.max, log=True)
    if "weight_decay" in sp.to_tune:
        hparams["weight_decay"] = trial.suggest_float("weight_decay", sp.weight_decay.min, sp.weight_decay.max, log=True)
    if "optim" in sp.to_tune:
        hparams["optim"] = trial.suggest_categorical("optim", sp.optim)
    if "scheduler" in sp.to_tune:
        hparams["scheduler"] = trial.suggest_categorical("scheduler", sp.scheduler)
    if "is_batchnorm" in sp.to_tune:
        hparams["is_batchnorm"] = trial.suggest_categorical("is_batchnorm", sp.is_batchnorm)

    set_hparams_(cfg, datamodule, hparams)

    predictor = Predictor(cfg, datamodule.z_dim, datamodule.n_labels)
    trainer = pl.Trainer(logger=False,
                         plugins=[SLURMEnvironment(auto_requeue=False)],   # see lightning #6389. very annoying but it already tries to requeue
                         **cfg.trainer)
    trainer.fit(predictor, datamodule=datamodule)

    # evaluates
    eval_dataloader = datamodule.test_dataloader()
    results = trainer.test(dataloaders=eval_dataloader, ckpt_path=None, model=predictor)[0]
    results = {k.split("/")[-1]: v for k, v in results.items()}
    return results[cfg.predictor.hypopt.metric]

def set_hparams_(cfg, datamodule, hparams):
    """Set the hyperparameters."""

    if cfg.predictor.is_sklearn:
        for k, v in hparams.items():
            k = k.replace("clf__", "")  # in case you had a Pipeline
            if isinstance(v, np.float64):
                v = float(v)
            cfg.predictor.model[k] = v

    else:
        cfgo = cfg.predictor.opt_kwargs
        cfga = cfg.predictor.arch_kwargs

        if "batch_size" in hparams:
            datamodule.batch_size = 2 ** hparams["batch_size"]
            cfg.data.kwargs.batch_size = datamodule.batch_size
        if "lr" in hparams:
            cfgo.lr = hparams["lr"]
        if "weight_decay" in hparams:
            cfgo.weight_decay = hparams["weight_decay"]
        if "optim" in hparams:
            cfgo.optim = hparams["optim"]
        if "scheduler" in hparams:
            cfgo.scheduler = hparams["scheduler"]
        if "is_batchnorm" in hparams:
            cfga.is_normalize = hparams["is_batchnorm"]


def summarize_study(study, path, is_wandb=False):
    optim_history = plot_optimization_history(study)
    param_importances = plot_param_importances(study)
    time_importances = plot_param_importances(study, target_name="duration",
                                              target=lambda t: t.duration.total_seconds())
    parallel_coordinate = plot_parallel_coordinate(study)
    plt.tight_layout()

    plots = dict(optimization_history=optim_history,
                 parallel_coordinate=parallel_coordinate,
                 param_importances=param_importances,
                 time_importances=time_importances
                 )

    for name, plot in plots.items():
        save_fig(plot, path / name)

    study.trials_dataframe().to_csv(path / "trials_dataframe.csv", header=True, index=False)

    if is_wandb:
        # log to wandb if its active
        for name, plot in plots.items():
            wandb.log({name: [wandb.Image(plot)]})