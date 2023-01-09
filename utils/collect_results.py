import os

import joblib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pdb
import types
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.inspection import permutation_importance
import shap
from IPython.display import display

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
import copy
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
from lmfit import Model, Parameter
import inspect
from pandas.api.types import is_numeric_dtype

import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, plot_parallel_coordinate

import hubconf
from utils.helpers import ols_clean_df_, powerset
from utils.pretty_renamer import PRETTY_RENAMER

COMPONENTS_ONLY = ["approx", "usability", "probe_gen", "enc_gen"]
COMPONENTS_ONLY_IMP = [ "usability", "probe_gen", "enc_gen", "approx"]
COMPONENTS = COMPONENTS_ONLY + ["agg_risk"]
CORE_METRICS = ["train_test", 'train-balsbst-ntrain0.01_test',
                ]

def tune_std_xgb(X, y, seed=123, n_trials=50, verbose=False, **kwargs):
    """Tune standard xgboost and return final model + study"""
    sampler = TPESampler(seed=seed,
                         n_startup_trials=n_trials // 2,
                         multivariate=True,
                         group=True,
                         consider_endpoints=True)

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(sampler=sampler, direction="minimize",
                                load_if_exists=True)  # continues from previous if exist

    study.optimize(lambda t: std_xgb_objective(t, X, y, seed=seed, **kwargs),
                   n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)  # ensures memory not adding

    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    params = dict(**study.best_trial.params)
    num_boost_round = params.pop("num_boost_round")
    xgb_cv = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    nfold=30)
    rmse = xgb_cv.iloc[-1]["test-rmse-mean"]
    print(f"Final 30-fold cv rmse={rmse}")

    best_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
    return best_model, study, rmse

def plot_optuna(study):
    """Visually summarizes an optuna hypopt study."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        plot_optimization_history(study)
        plt.show()

        plot_param_importances(study)
        plt.show()

        plot_param_importances(study, target_name="duration",
                                target=lambda t: t.duration.total_seconds())
        plt.show()

        plot_parallel_coordinate(study)
        plt.show()

def tune_sk_xgb(X, y, seed=123, n_trials=50, verbose=False, **kwargs):
    """Tune xgboost sklearn and return final model + study"""
    sampler = TPESampler(seed=seed,
                         n_startup_trials=n_trials // 2,
                         multivariate=True,
                         group=True,
                         consider_endpoints=True)

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(sampler=sampler, direction="minimize",
                                load_if_exists=True)

    study.optimize(lambda t: sk_xgb_objective(t, X, y, seed=seed, **kwargs),
                   n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)  # ensures memory not adding

    best_model = xgb.XGBRegressor(enable_categorical=True, **study.best_trial.params)
    rmse = cross_validate_pd(best_model, X, y, kfold=30, verbose=False)

    print(f"Final 30-fold cv rmse={rmse}")

    return best_model, study

def std_xgb_objective(trial, X, y, is_force_exact=False, seed=123, kfold=10):
    """Hyperparameter tuning objective with standard xgboost."""

    num_boost_round = trial.suggest_categorical('num_boost_round', [300, 1000, 2000])
    params = {'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
              'alpha': trial.suggest_float('alpha', 1e-8, 1, log=True),
              'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 1.0),
              'subsample': trial.suggest_float("subsample", 0.6, 1.0),
              'eta': trial.suggest_float("eta", 3e-2, 1.0, log=True),
              'max_depth': trial.suggest_int("max_depth", 3, 10),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 9, step=2),
              "objective": 'reg:squarederror'}

    if is_force_exact:
        params['tree_method'] = "exact"
    else:
        params['tree_method'] = trial.suggest_categorical('tree_method',
                                                          ["approx", "hist", "exact"])  # could also use gpu_hist

    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

    xgb_cv = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    nfold=kfold,
                    seed=seed)

    return xgb_cv.iloc[-1]["test-rmse-mean"]


def sk_xgb_objective(trial, X, y, **kwargs):
    """Hyperparameter tuning objective with sklearn API of xgboost."""

    params = {
        'tree_method': trial.suggest_categorical('tree_method', ["approx", "hist"]),  # could also use gpu_hist
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1, log=True),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 1.0),
        'subsample': trial.suggest_float("subsample", 0.6, 1.0),  # very little data => don't subsaple too much
        'learning_rate': trial.suggest_float("learning_rate", 3e-2, 1.0, log=True),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 300, 1000]),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 9, step=2),
    }

    model = xgb.XGBRegressor(enable_categorical=True, **params)

    score = cross_validate_pd(model, X, y, verbose=False, **kwargs)
    return score


def cross_validate_pd(model, X, y, kfold=10, seed=123, **kwargs):
    """Performs cross validation for pandas data."""
    if kfold == -1:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)

    scores = []
    for train_index, test_index in kf.split(X):
        curr_model = copy.deepcopy(model)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        curr_model.fit(X_train, y_train, **kwargs)
        rmse = mean_squared_error(y_test, curr_model.predict(X_test), squared=False)
        scores.append(rmse)
    return sum(scores) / len(scores)

def path2params(f):
    """Converts a path to a dictionary of parameters."""
    return {f.split("_", 1)[0].lower(): f.split("_", 1)[1].lower()
            for f in str(f.resolve()).split("results")[1].split("/")[1:-1]}

def load_single_file(f, skip_ifin=dict(), skip_ifneq=dict(), metric="err", is_skip_test=True):
    """Load results from a single file. Skip is a dictionary of values to skip for certain params."""

    params = path2params(f)

    if is_skip_test:
        if "test" in params["exp"]:
            return None, None

    for k, values in skip_ifin.items():
        for v in values:
            if v in params[k]:
                return None, None

    for k, v in skip_ifneq.items():
        if params[k] != v:
            return None, None

    metrics = pd.read_csv(f, index_col=0).T[metric]
    if "hyp" in params:
        key = (params["ssl"], params["pred"], params["seed"], params["hyp"])
    else:
        key = (params["ssl"], params["pred"], params["seed"])

    return key, metrics

def load_all_results(results_dir=Path() / "results/",
                     pattern="**/results_all.csv",
                     metric="err",
                     skip_ifin=dict(exp=["dev", "inv", "shat"], ssl=["colorization"]),
                     skip_ifneq=dict(data="imagenet-N5")
                     ):
    """Load all ssl and supervised results."""
    #
    files = list(results_dir.glob(pattern))
    print(f"Found {len(files)} result files to load.")
    # files = [f for f in files if "imagenet-N5" in str(f)]

    results = []
    keys = []

    for f in files:

        key, metrics = load_single_file(f,
                                        skip_ifin=skip_ifin,
                                        skip_ifneq=skip_ifneq,
                                        metric=metric)

        if metrics is None:
            continue

        results.append(metrics)
        keys.append(key)

    df_results = pd.concat(results, keys=keys, axis=1).T
    if len(df_results.index.names) == 4:
        df_results.index.set_names(["enc", "pred", "seed", "hyp"], inplace=True)
    else:
        df_results.index.set_names(["enc", "pred", "seed"], inplace=True)

    if metric == "err":
        df_results = df_results * 100

    return df_results


def add_approximation_results_(ssl, sup, is_allow_nan=True, **sbst_kwargs):
    """Uses the supervised results to add approximation error to the SSL models."""

    sffx = get_sffx_data(**sbst_kwargs)
    if sffx == "":
        tr_tr_sffx = ""
    else:
        tr_tr_sffx = "-balsbst-62811"

    ind_sup = sup.index.droplevel("enc")
    dupli = sup[ind_sup.duplicated()]
    if len(dupli) > 0:
        raise ValueError(f"Assuming that supervised results only vary in architecture, probe, seed. But duplicates "
                         f"found {dupli}.")

    # replace model name by arch
    sup.index = ind_sup
    ssl_to_sup = ssl.index.droplevel(0)

    missing = np.array([tuple(k) not in sup.index for k in ssl_to_sup.values])
    if sum(missing) > 0:
        assert is_allow_nan
        warnings.warn(f"Found missing supervised models for:")
        display(ssl_to_sup[missing].to_frame(index=False))
        ssl[f"sup_train{tr_tr_sffx}_train{tr_tr_sffx}"] = np.nan
        ssl[f"sup_train{sffx}_test"] = np.nan

    ssl.loc[~missing, f"sup_train{tr_tr_sffx}_train{tr_tr_sffx}"] = sup.loc[ssl_to_sup[~missing], f"train{tr_tr_sffx}_train{tr_tr_sffx}"].values
    ssl.loc[~missing, f"sup_train{sffx}_test"] = sup.loc[ssl_to_sup[~missing], f"train{sffx}_test"].values

    #ssl.loc[~missing, f"sup_train{tr_tr_sffx}_train{tr_tr_sffx}"]

def format_approx_results(results, metadata, f_replace_arch=None, is_keep_sup=True, **kwargs):
    """Separates the supervised results into the approximation error column."""

    arch = np.array([metadata.loc[i, "architecture_exact"] for i in results.index.get_level_values(0)])
    results["arch"] = arch

    is_supervised = np.array([metadata.loc[i, "ssl_mode"] for i in results.index.get_level_values(0)]) == "supervised"
    results_sup = results[is_supervised].copy()

    if is_keep_sup:
        results_ssl = results.copy()
    else:
        results_ssl = results[~is_supervised].copy()

    results_sup = results_sup.set_index("arch", append=True)
    if f_replace_arch is not None:
        # replaces architectures if needed for the supervised models (in the case where couldn't find correct one)
        results_ssl.loc[:, "arch"] = results_ssl.loc[:, "arch"].apply(f_replace_arch)
    results_ssl = results_ssl.set_index("arch", append=True)

    # Placeholder that gives 0 to whatever you want. Used to set zero for approx error of large models
    results_sup.loc[("zero", "torch_linear_delta_hypopt", "123", "zero")] = 0

    add_approximation_results_(results_ssl, results_sup, **kwargs)

    results_ssl.index = results_ssl.index.droplevel(["arch"])

    return results_ssl


def check_missing(results, metadata):
    """Flags if results or meta data are missing."""
    results_models = set(results.index.get_level_values(0))
    metadata_models = set(metadata.index.get_level_values(0))
    no_metadata = results_models.difference(metadata_models)
    no_results = metadata_models.difference(results_models)

    if len(no_metadata) > 0:
        txt = '\n'.join(no_metadata)
        warnings.warn(f"Missing metadata for: \n {txt}")

    if len(no_results) > 0:
        txt = '\n'.join(no_results)
        warnings.warn(f"Missing results for: {txt}")


def f_replace_arch(arch):
    """
    Maps architectures of the ssl models to supervised ones if they are very similar.
    """
    arch = arch.replace("clipresnet", "resnet")
    arch = arch.replace("beit", "vit")
    arch = arch.replace("vitl14", "vitl16")  # close enough
    arch = arch.replace("vitl7", "vitl16")  # close enough
    arch = arch.replace("vitb4", "vitb8")  # close enough
    arch = arch.replace("vitb32 pred", "vitb32 cls")  # close enough although dimensionality decreases from 768 -> 512

    if "resnet50d" in arch:
        # if increasing last dimension the architecture does not change much
        return "resnet50"

    if arch == "resnet50w2":
        # surprisingly I couldn't find any pytorch pretrained resnet50x2
        # so i' using wide resnet which is somewhat comparable although not ideal
        return "wide_resnet50_2"

    # all the following are assumed to have zero approximation error given that they are upperbounded
    # by resnet50w which achieve 0.8% and those are much better with larger dim
    # and are much better models
    if arch in ["resnet50w4", "resnet50w16", "resnet50w64", "vith14 cls", "convnextxl"]:
        arch = "zero"
    #arch = arch.replace("vith14 cls", "zero")

    # Those values seem are suspiciously large. A vit-S would be able to fit the TRAINING data by more than 92%
    # => remove until you rerun
    if arch in ["vits16 cls+avg", "vits16 cls"]:
        arch = "missing"


    return arch

def get_sffx_data(subset=None, n_per_class=None):
    """Return the suffix for the data."""
    if subset is not None:
        sffx = f"-balsbst-ntrain{subset}"
    elif n_per_class is not None:
        sffx = f"-nperclass-{n_per_class}"
    else:
        sffx=""
    return sffx

def make_risk_decomposition(results, traverse_path=["down", "right", "down"],
                            is_print=False, **sbst_kwargs):
    """
    Make the risk decomposition depending on how you want to traverse the decomposition table.
    traverse_path=`None` returns decomposition table.
    """
    results = results.copy()
    sffx = get_sffx_data(**sbst_kwargs)

    if sffx == "":  # no subsetting
        enc_gen = 'union_test'
        probe_gen = 'train-cmplmnt-ntest_train-sbst-ntest'
        tr_tr_sffx = ""
    else:
        enc_gen = f"test{sffx}_test{sffx}"
        probe_gen = f"train{sffx}_train-balsbst-ntest"
        tr_tr_sffx = "-balsbst-62811"

    # all the values from the decomposition table
    dec_table = np.array([[f'sup_train{tr_tr_sffx}_train{tr_tr_sffx}', f'sup_train{sffx}_test'],
                          [f'train{tr_tr_sffx}_train{tr_tr_sffx}', probe_gen],
                          [enc_gen, f'train{sffx}_test']])
    results["agg_risk"] = results[f'train{sffx}_test']

    if traverse_path is None:
        selected_columns = list(dec_table.flatten()) + ["agg_risk"]

    else:
        results["approx"] = results[f'sup_train{tr_tr_sffx}_train{tr_tr_sffx}']
        selected_columns = COMPONENTS

        # traverses the decomposition table
        ind = [0, 0]
        component_names = dict(down=["enc_gen", "usability"], right=["probe_gen"])  # last will be taken first
        for action in traverse_path:
            curr_val = results[dec_table[tuple(ind)]]

            if is_print:
                old_ind = list(ind)

            if action.lower() == "down":
                ind[0] += 1
            elif action.lower() == "right":
                ind[1] += 1
            else:
                raise ValueError(f"Unknown action={action} in traverse_path={traverse_path}.")

            if is_print:
                print(f"[{component_names[action][-1]}] = [{dec_table[tuple(ind)]}] - [{dec_table[tuple(old_ind)]}]")

            results[component_names[action].pop()] = results[dec_table[tuple(ind)]] - curr_val

    return results#[selected_columns]


def add_statistics_(metadata):
    """Add precomputed statistics to the metadata."""
    results_dir = Path() / "results/exp_statistics"
    pattern = "data_*/ssl_*/"
    files = list(results_dir.glob(pattern))

    all_data = ["train", "test", "trainaug", "testaug", "trainrealaug", "testrealaug"]
    for f in files:
        params = path2params(f/"placeholder")

        for d in all_data:
            f_stats = f / f"{d}_statistics.npz"
            if f_stats.exists():
                with np.load(f  / f"{d}_statistics.npz") as statistics:
                    for k in statistics.keys():
                        if statistics[k].size == 1:
                            metadata.loc[params["ssl"], f"{d}_{k}"] = statistics[k]

    for d in all_data:
        try:

            metadata[f"{d}_vars"] = metadata[f"{d}_intra_var"] / metadata[f"{d}_inter_var"]
        except KeyError:
            pass

    # nc1 computes the trace => sum over non zero dimensions => let's normalize (sqrt is to have std instead of var)
    for d in all_data:
        try:
            prfx = "train" if d.startswith("train") else "test"
            metadata[f"{d}_nc1norm"] = metadata[f"{d}_nc1"] / metadata[f"{prfx}_rank"]**0.5
        except KeyError:
            pass

def clean_results(results,
                  metadata,
                  predictor=None,  # predictor to select
                  is_avg_seed=True,  # avg over seed
                  is_positive=True,  # force pos
                  is_add_metadata=True  # adds additional metadata that are funciton of others
                  ):
    """Clean the risk decomposition results."""

    if predictor is not None:
        results = results.loc[results.index.get_level_values("pred") == predictor]
        results.index = results.index.droplevel("pred")

    if is_avg_seed:
        idcs_no_seed = [n for n in results.index.names if n != "seed"]
        results = results.reset_index().groupby(idcs_no_seed).mean(numeric_only=True)

    metadata = metadata.loc[results.index.get_level_values("enc")]

    if is_positive:
        idx_neg = (results < 0).any(axis=1)
        if idx_neg.sum() > 0:
            warnings.warn(f"Found negative values:")
            display(results[idx_neg].round(3))
        results = results.clip(lower=0)

    for c in metadata.columns:
        # Int64 still experimental / has issues (eg working with seaborn) so avoid if can
        if metadata[c].dtype == pd.Int64Dtype():
            try:
                metadata[c] = metadata[c].astype("int")
            except:
                pass

    metadata = metadata.replace(dict(ssl_mode={"hierarchical contrastive": "hierarchical"}))

    if is_add_metadata:
        add_metadata_(metadata)


    return results, metadata

def validate_results(results, metadata,
                     threshold_bad_ifnew=50,  # used to flag bad or good models
                     threshold_bad_ifold=90,
                     new_old_threshold=2020,
                     threshold_delta=10
                     ):
    """Validates the risk decomposition results"""
    bad_new = results[(results["agg_risk"] > threshold_bad_ifnew) & (metadata["year"] >= new_old_threshold)]
    bad_old = results[(results["agg_risk"] > threshold_bad_ifold) & (metadata["year"] < new_old_threshold)]
    bad = pd.concat([bad_new, bad_old], axis=0)
    if bad.shape[0]:
        warnings.warn(f"The following results seem suspicously bad:")
        display(bad[['agg_risk']])

    delta = results["agg_risk"] - (100 - metadata["top1acc_in1k_official"])
    is_large_delta = delta.abs() > threshold_delta
    if is_large_delta.sum() > 0:
        warnings.warn(f"The following models have very different original and evalauted performance:")
        display(delta[is_large_delta])

    is_nan = results[COMPONENTS_ONLY].isna().any(axis=1)
    if is_nan.sum() > 0:
        warnings.warn(f"The following results have some nan:")
        display(results.loc[is_nan,COMPONENTS_ONLY].round(3))

def add_metadata_(metadata):
    """Adds additional metadata that are function of others"""
    idx_ssl = ~metadata["ssl_mode"].isin(["initialized","supervised"])

    metadata.loc[idx_ssl,"nviews"] = metadata.loc[idx_ssl,"views"].apply(lambda s: count_views(s)).astype(pd.Int64Dtype())

    # cleanup date (chooses day = 1 arbitrarily)
    metadata.loc[idx_ssl,"date_published"] = pd.to_datetime(dict(year=metadata[idx_ssl].year,
                                                                     month=metadata[idx_ssl].month,
                                                                     day=metadata[idx_ssl].month * 0 + 1),
                                                                errors='coerce')

    metadata.loc[idx_ssl,"n_augmentations"] = metadata.loc[idx_ssl,"augmentations"].apply(lambda s: len(s))

    metadata.loc[idx_ssl, "projection_nparameters_hidden"] = metadata.loc[idx_ssl, "projection_nparameters"] - (metadata.loc[idx_ssl, "z_dim"] * metadata.loc[idx_ssl, "projection_hid_width"])


def count_views(s):
    if pd.isnull(s):
        return 0
    count = 0
    while "x" in s:
        splitted = s.split("x", maxsplit=1)
        count += int(splitted[0][-1])
        s = splitted[1]
    return max(count,1)

def preprocess_features(df,
                        round_dict=dict(n_parameters=int(1e7), n_classes=100, projection_nparameters=int(1e7), epochs=100),
                        pow_dict=dict(batch_size=2, z_dim=2, patch_size=2,learning_rate=10,
                                      weight_decay=10, pred_dim=2, img_size=2, n_negatives=2,
                                      projection_hid_width=2, n_classes=2, n_augmentations=2,
                                      nviews=2),
                        len_cols=[]):
    """Preprocesses the features to make it more amenable for ML."""

    df = df.copy()

    for c in len_cols:
        df["n_"+c] = df[c].apply(lambda s: len(s))

    for c, round_to in round_dict.items():
        notna = (~df[c].isna())
        df.loc[notna,c] = (df.loc[notna,c] / round_to).round() * round_to

    for c, base in pow_dict.items():
        logable = (~df[c].isna()) & (df[c] > 0)
        powered = base**(np.log(df.loc[logable,c])//np.log(base))
        if df[c].dtype in [int, pd.Int64Dtype()]:
            powered = powered.round().astype(df[c].dtype)
        df.loc[logable,c] = powered
    return df


def prepare_sklearn(df,
                    features_to_del=["notes", "where", "top1acc_in1k_official", "n_pus", "pu_type", "time_hours",
                                     "license", "month"],
                    features_to_keep=None,
                    components=COMPONENTS,
                    min_count_onehot=3,
                    features_onehot=[],
                    target="agg_risk"):
    X = df.copy()
    y = df[target].copy()

    X = X[~y.isna()]
    y = y[~y.isna()]

    X = X.drop(components, axis=1)

    if features_to_keep is not None:
        features_to_del = [c for c in X.columns if c not in features_to_keep]

    X = X.drop(features_to_del, axis=1)

    all_str_cols = []
    for c in X.columns:
        # convert categorical
        if X[c].dtype == "string" or isinstance(X[c].dtype, pd.StringDtype):
            X[c] = X[c].astype("category")
            all_str_cols += [c]
        elif isinstance(X[c].dtype, pd.Int64Dtype):
            try:
                X[c] = X[c].astype(int)
            except:
                X[c] = X[c].astype(float)
        elif isinstance(X[c].dtype, pd.Float64Dtype):
            X[c] = X[c].astype(float)
        elif isinstance(X[c].dtype, pd.BooleanDtype) or X[c].dtype == bool:
            try:
                X[c] = X[c].astype(int)
            except:
                X[c] = X[c].astype(float)

    if features_onehot == "all":
        features_onehot = all_str_cols

    for o in features_onehot:
        counts = X[o].value_counts()
        idx_threshold = counts > min_count_onehot
        if not idx_threshold.all():
            X[o] = pd.Categorical(
                X[o],
                categories=list(counts[idx_threshold].index) + ["other"]
            ).fillna('other')

    X = pd.get_dummies(X,
                       prefix=features_onehot,
                       columns=features_onehot,
                       drop_first=True)

    date_col = X.select_dtypes(include=['datetime64']).columns
    X = X.drop(date_col, axis="columns")

    return X, y


def report_xgboost(reg, X, y, is_std_featimp=True, is_shap_interactive=False,
                   is_perm_importance=True, is_shap=True, n_feat=15,
                   examples=["swav_rn50_ep400", "dino_vitS16", "dino_rn50", "dissl_resnet50_d8192_e400_m6",
                             "dissl_resnet50_dNone_e400_m6", "dissl_resnet50_dNone_e400_m2",
                             "dissl_resnet50_dNone_e100_m2"]
                   ):
    sns.set_style("white")

    if isinstance(reg, xgb.core.Booster):
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True, feature_names=X.columns)
        inp_pred = dtrain
    else:
        inp_pred = X

    regression_report(y, reg.predict(inp_pred))

    if is_std_featimp:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # gain: gain in accuracy. for each tree in the model
        xgb.plot_importance(reg,
                            importance_type='gain',
                            max_num_features=n_feat,
                            show_values=False,
                            grid=False,
                            ax=axes[0],
                            title='',
                            xlabel='gain')
        # cover: metric means the relative number of observations related to this feature.
        xgb.plot_importance(reg,
                            importance_type='cover',
                            max_num_features=n_feat,
                            show_values=False,
                            grid=False,
                            ax=axes[1],
                            title='',
                            ylabel='',
                            xlabel='cover', )
        plt.suptitle("XGB Feature importance")
        plt.tight_layout()
        plt.show()

    if is_perm_importance and not isinstance(reg, xgb.core.Booster):
        perm_importance = permutation_importance(reg, X, y)
        sorted_idx = perm_importance.importances_mean.argsort()[-n_feat:]
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.3))
        ax.barh(X.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.tight_layout()
        plt.show()

    if is_shap and isinstance(reg, xgb.core.Booster):
        explainer = shap.TreeExplainer(reg, feature_names=X.columns)
        shap_values = explainer.shap_values(dtrain)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        shap.summary_plot(shap_values, X, show=False, plot_size=None, max_display=n_feat, color_bar=False)
        plt.gca().set_xlabel("SHAP: impact on model output", fontsize=12)
        plt.subplot(1, 2, 2)
        shap.summary_plot(shap_values,
                          X,
                          show=False,
                          plot_size=None,
                          plot_type="bar",
                          max_display=n_feat,
                          color_bar=False,
                          feature_names=["" for _ in X.columns])
        plt.gca().set_xlabel("mean(|SHAP|)", fontsize=12)
        plt.tight_layout()
        plt.show()

        for c in examples:
            print(c)
            shap.force_plot(explainer.expected_value, shap_values[X.index.get_loc(c.lower()), :],
                            feature_names=X.columns, matplotlib=True)

            # plt.gca().set_title("c", fontsize=12)

        if is_shap_interactive:
            display(shap.plots.force(explainer.expected_value, shap_values, feature_names=X.columns))


def get_only_vary(df, varying_keys, all_keys, drop_cols=[]):
    """Return all results that only vary on some given features."""
    non_vary = [k for k in all_keys if k not in varying_keys]
    nunique = df[all_keys].reset_index().groupby(non_vary, dropna=False).nunique()
    indices = nunique.loc[(nunique[varying_keys] > 1).values].index
    selected = df.set_index(non_vary).loc[indices].reset_index()
    non_vary = [c for c in non_vary if selected[c].nunique() > 1 and c not in drop_cols]
    selected["non_vary"] = selected[non_vary].astype(pd.StringDtype()).fillna("NA").agg(' '.join, axis=1)
    return selected


def melt(df, components=COMPONENTS, var_name="component", **kwargs):
    """Melts components"""
    return pd.melt(df,
                   value_vars=components,
                   id_vars=[c for c in df.columns if c not in components],
                   var_name=var_name,
                   **kwargs)

def clean_model_name(name, pretty_renamer=PRETTY_RENAMER):
    name = name.replace(" ", "_")
    if name.count("_") >= 2:
        *model_arch, rest = name.split("_", 2)
        model_arch = "_".join(model_arch)
    else:
        model_arch = name
        rest = ""
    model_arch = pretty_renamer[model_arch]
    return model_arch, rest


def filter_by_quantile(df, col="agg_risk", is_year=True, quantile=0.1):
    """Select results that are the best (quantile) in total or for their year"""
    df = df.copy()
    if is_year:
        quantiles = df.groupby("year").quantile(quantile, numeric_only=True)[col]
        quantiles = [quantiles[y] for y in df.year]
    else:
        quantiles = df[col].quantile(quantile)

    return df[df[col].le(quantiles)]


def load_df(is_read_files = True,
            DATA = "imagenet",
            subset = None,
            pred = 'torch_linear_delta_hypopt',
            threshold_kwargs = dict(),
            traverse_path=["down", "right", "down"],
            is_zero_approx=False):

    metadata_df = hubconf.metadata_df(is_multiindex=False)

    if is_read_files:
        results = load_all_results(pattern=f"**/data_{DATA}/**/pred_{pred}/seed_*/results_all.csv",
                                   skip_ifneq=dict(data=DATA.lower()),
                                   skip_ifin=dict(ssl=["swav_rn50w5", "selav2_rn50_ep400_2x160_4x96"]),
                                   )

        check_missing(results, metadata_df)
        results = format_approx_results(results, metadata_df, f_replace_arch=f_replace_arch,
                                        subset=subset)

        if is_zero_approx:
            assert subset is None
            results[f"sup_train_train"] = 0

        results = make_risk_decomposition(results,
                                          traverse_path=traverse_path,
                                          is_print=True,
                                          subset=subset)

        results, metadata_df = clean_results(results, metadata_df, predictor=pred)
        add_statistics_(metadata_df)
        validate_results(results, metadata_df, threshold_delta=5, **threshold_kwargs)

        results.to_csv(f"notebooks/saved/results_{DATA}_{pred}.csv")

        # filter out values that are suspiciously bad
        to_del = []
        to_keep = [i for i in results.index.get_level_values("enc") if i not in to_del]
        results = results.loc[to_keep]
    else:
        results = pd.read_csv(f"notebooks/saved/results_{DATA}_{pred}.csv", index_col=0)

    metadata_df = metadata_df.loc[to_keep]
    df = pd.concat([results, metadata_df], axis=1)
    return df, metadata_df

def compute_correlations_log(x,y):
    """Compute correlations between x and y under possible log transformation"""
    for is_logx in [True, False]:
        for is_logy in [True, False]:
            print(f"logx={is_logx}, logy={is_logy}")
            compute_correlations(x,y, is_logx=is_logx, is_logy= is_logy)


def compute_correlations(x, y, is_logx=False, is_logy=False, correlations=["Pearson", "Spearman", "Kendall"]):
    """Compute different correlation coefficient."""

    notna = ~(x.isna() | y.isna())
    x = x[notna]
    y = y[notna]


    if is_logx:
        selectx = x > 0
        x = np.log(x[selectx])
    else:
        selectx = True

    if is_logy:
        selecty = y > 0
        y = np.log(y[selecty])
    else:
        selecty = True

    if is_logx or is_logy:
        select = selectx & selecty
        x = x[select]
        y = y[select]

    for k, f in dict(Pearsons=pearsonr,  # linear
                     Spearman=spearmanr,  # monotonic / rank ordering
                     Kendall=kendalltau,  # monotonic / rank ordering
                     ).items():
        if k  in correlations:
            corr, pval = f(x, y)
            print(f'{k} correlation: {corr:.3f} pvalue = {pval:.2e}')

def loglike_undolog(self, params, scale=None):
    nobs2 = self.nobs / 2.0
    nobs = float(self.nobs)
    resid = np.exp(self.endog) - np.exp(np.dot(self.exog, params))
    assert not hasattr(self, 'offset'), "cannot yet deal with offset"

    ssr = np.sum(resid ** 2)
    if scale is None:
        # profile log likelihood
        llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(ssr / nobs) - nobs2
    else:
        # log-likelihood
        llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2 * scale)
    return llf

def fit_best_ols(data, potential_features,
                 model_selection = "bic", target = "train_test", log_target = [False]):


    best_loss = float("inf")

    for is_log_target in log_target:
        if is_log_target:
            ftarget = f"np.log({target})"
            f = np.exp
        else:
            f = lambda x: x
            ftarget = target

        for features in powerset(potential_features):
            formula = f"{ftarget} ~ " + " + ".join(features)
            model = smf.ols(formula=formula, data=data)

            if is_log_target:
                model.loglike = types.MethodType(loglike_undolog, model)

            model = model.fit()

            loss = getattr(model, model_selection)

            if loss < best_loss:

                best_loss = loss
                best_model = model
                best_features = features
                best_log_target = is_log_target
                rmse = ((data[target] - f(best_model.predict(data)))**2).mean()**0.5


    print(f"best features: {best_features}. {model_selection}: {best_loss}. R2 = "
          f"{best_model.rsquared:.3f}. best_log_target: {best_log_target}. best rmse: {rmse}")
    return best_model

def get_sample_size(s):
    if s =="train_test":
        return 1281167
    elif "train-nperclass-" in s:
        return int(s.split("_")[0][len("train-nperclass-"):]) * 1000
    elif "train-balsbst-ntrain" in s:
        return int(float(s.split("_")[0][len("train-balsbst-ntrain"):]) * 1281167)

def fit_scaling_law(data, features, target="train_test", n_epsilons=20,
                    is_raw_r2=True, eps_col=None, min_eps=0,
                    no_intercept=False, is_log_target=True):
    best_loss = float("inf")

    data = data.copy()
    ols_clean_df_(data, features+[target])

    if eps_col is not None:
        eps = data[eps_col]
    else:
        eps = 0

    if is_log_target:
        def f_inv_tgt(pred_tgt, eps):
            return np.exp(pred_tgt) + eps

        def f_tgt(tgt, eps):
            return np.log(tgt - eps)
    else:
        def f_inv_tgt(pred_tgt, eps):
            return pred_tgt + eps

        def f_tgt(tgt, eps):
            return tgt - eps

    for delta_eps in np.linspace(min_eps, (data[target] - eps).min(), num=n_epsilons, endpoint=False):

        data_eps = data.copy()
        new_eps = eps + delta_eps

        data_eps[target] = f_tgt(data_eps[target], new_eps)
        formula = f"{target} ~ " + " + ".join(features)
        if no_intercept:
            formula += " - 1"
        model = smf.ols(formula=formula, data=data_eps).fit()
        pred_tgt = f_inv_tgt(model.predict(data_eps), new_eps)
        rmse = ((data[target] - pred_tgt) ** 2).mean() ** 0.5

        if is_raw_r2:
            loss = 1 - model.rsquared
        else:
            loss = rmse

        if loss < best_loss:
            best_loss = loss
            best_model = model
            best_delta_eps = delta_eps
            best_rmse = rmse

    print(f"R2 = {best_model.rsquared:.3f}. best rmse: {best_rmse:.3f}. best delta eps: {best_delta_eps:.3f}")
    return best_model, best_rmse


def regression_report(Y,Y_hat, sffx=""):
    rmse = mean_squared_error(Y,Y_hat, squared=False)
    r2 = r2_score(Y,Y_hat)
    print(f"{sffx}RMSE: {rmse:.4f}. R2: {r2:.4f}")


def f_pred(params, data, model_var):
    return (params["Irr"] + params["C"] / (params["n_samples"] ** params["alpha"])).clip(0,100)

def f_pred_param(params, data, model_var):
    std = f_pred(params, data, model_var)
    return (std + params["K"] / (params["n_params_probe"] ** params["beta"])).clip(0,100)


def scalinglaw(data,
               independent_vars,
               model_col=None,
               f_pred=f_pred,
               coldep_report=["metrics"],
               model_dep=[],
               possible_params=["Irr", "C", "alpha"],
               all_pkwargs={"Irr": dict(value=10, min=0, max=100, user_data="Irr"),
                            "alpha": dict(value=0.2, min=0, max=0.5, user_data="alpha"),
                            "C": dict(value=100, min=0, max=1000, user_data="C")},
               max_nfev=100000,
               optim="least_squares",
               seed=123,
               test_mask=None,
               test_size=None,
               stratify="metrics",
               is_return_results=False
               ):
    def f_pred_params(model_var=None, data=None, **kwargs):
        params = {k: np.array([kwargs[f"{k}_{m}"] for m in model_var])
        if k in model_dep else kwargs[k]
                  for k in possible_params + independent_vars}
        return f_pred(params, data, model_var)

    all_params = {}
    for k in possible_params:
        if k in all_pkwargs:
            pkwargs = all_pkwargs[k]
        elif len(k) == 1 and k.isupper():
            # single upper letter for pos constants
            pkwargs = dict(value=100, min=0, max=1000, user_data=k)
        elif len(k) > 1 and k.islower():
            # greek letter for power
            pkwargs = dict(value=1, min=0, max=2, user_data=k)
        else:
            raise ValueError(f"Missing pkwargs for {k}")

        if k in model_dep:
            for m in data[model_col].unique():
                all_params[f"{k}_{m}"] = Parameter(f"{k}_{m}", **pkwargs)
        else:
            all_params[k] = Parameter(k, **pkwargs)

    f_pred_params.__signature__ = inspect.Signature([inspect.Parameter(param,
                                                                       kind=inspect.Parameter.POSITIONAL_OR_KEYWORD) for
                                                     param in independent_vars + list(all_params.keys())] +
                                                    [inspect.Parameter(param,
                                                                       default=None,
                                                                       kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                     for param in ["model_var", "data"]])

    if test_mask is not None:
        data_train = data[~test_mask]
        data_test = data[test_mask]

    elif test_size is not None:
        if stratify is not None:
            stratify = data[stratify]
        data_train, data_test = train_test_split(data, test_size=test_size, random_state=seed, stratify=stratify)

    else:
        data_train, data_test = data, data

    if model_col is not None and (len(data[model_col].unique()) != len(data_train[model_col].unique())):
        raise ValueError("Not all model_var are in train data")

    model = Model(f_pred_params,
                  independent_vars=independent_vars,
                  model_var=data_train[model_col] if model_col is not None else None,
                  data=data_train)

    indep_v = {i: data_train[i] for i in independent_vars}
    result = model.fit(data_train["value"],
                       nan_policy='omit',
                       max_nfev=max_nfev,
                       method=optim,
                       **indep_v,
                       **all_params,
                       )

    def predict(data):
        indep_v = {i: data[i] for i in independent_vars}
        return result.eval(**indep_v,
                           data=data,
                           model_var=data[model_col] if model_col is not None else None)

    Y_train = data_train["value"]
    Y_test = data_test["value"]
    Yh_train = predict(data_train)
    Yh_test = predict(data_test)

    regression_report(Y_train, Yh_train, sffx="*Train* ")
    regression_report(Y_test, Yh_test, sffx="*Test* ")

    def res_df(Y, Yh):
        rmse = mean_squared_error(Y, Yh, squared=False)
        r2 = r2_score(Y, Yh)
        return pd.Series(dict(rmse=rmse, r2=r2, mse=rmse**2/Y.var()))

    for c in coldep_report:
        df = pd.DataFrame({c: data_test[c], "y": Y_test, "y_pred": Yh_test})
        out = df.groupby(c).apply(lambda df: res_df(df["y"], df["y_pred"])).T
        display(out)

    fitted_params = pd.DataFrame.from_dict({k: dict(value=v.value, param=v.user_data) for k, v in
                                            result.params.items()}).T
    display(fitted_params.groupby("param").agg(['mean', 'sem']).T)

    print(f"N param: {len(result.params)}")

    if is_return_results:
        return result, res_df(Y_test, Yh_test)
    else:
        return result


def get_all_xgb(objectives, df, features_to_keep, prfx="", is_train=True, folder="notebooks/saved/", **kwargs):
    xgbs = dict()
    studys = dict()
    rmses = dict()
    Xs = dict()
    ys = dict()
    for o in objectives:
        Xs[o], ys[o] = prepare_sklearn(df,
                                       features_to_keep=features_to_keep,
                                       target=o)

        if is_train:
            xgbs[o], studys[o], rmses[o] = tune_std_xgb(Xs[o], ys[o], is_force_exact=True, **kwargs)
            joblib.dump(studys[o], f"{folder}{prfx}_study_{o}.pkl")
            xgbs[o].save_model(f"{folder}{prfx}_xgb_{o}.json")
            np.save(f"{folder}{prfx}_rmses_{o}.npy", rmses[o])
        else:
            xgbs[o] = xgb.Booster()
            xgbs[o].load_model(f"{folder}{prfx}_xgb_{o}.json")
            studys[o] = joblib.load(f"{folder}{prfx}_study_{o}.pkl")
            rmses[o] = np.load(f"{folder}{prfx}_rmses_{o}.npy")

    return xgbs, studys, Xs, ys, rmses


def get_df_shap(model, X, y):
    dtrain = xgb.DMatrix(X, label=y,
                         enable_categorical=True,
                         feature_names=X.columns)

    explainer = shap.TreeExplainer(model, feature_names=X.columns)
    shap_values = explainer(dtrain.get_data())
    shap_values.data = shap_values.data.toarray()

    joined = X.join(pd.DataFrame(shap_values.values,
                                 columns=[f"shap_{c}" for c in X.columns],
                                 index=X.index))
    return joined, shap_values





def prettify_df(df, pretty_renamer=PRETTY_RENAMER):
    df = df.copy()
    df = df.rename(columns=pretty_renamer)
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            df[c] = df[c].apply(lambda x: PRETTY_RENAMER[x])
    return df
