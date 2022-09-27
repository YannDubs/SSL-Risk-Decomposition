import warnings

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
from IPython.display import display

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
import copy
import seaborn as sns


import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, plot_parallel_coordinate


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

    study = optuna.create_study(sampler=sampler, direction="minimize")

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
    return best_model, study

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

    study = optuna.create_study(sampler=sampler, direction="minimize")

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

def load_single_file(f, skip_ifin=dict(), skip_ifneq=dict(), metric="err"):
    """Load results from a single file. Skip is a dictionary of values to skip for certain params."""

    params = {f.split("_", 1)[0].lower(): f.split("_", 1)[1].lower()
              for f in str(f.resolve()).split("results")[1].split("/")[1:-1]}

    for k, values in skip_ifin.items():
        for v in values:
            if v in params[k]:
                return None, None

    for k, v in skip_ifneq.items():
        if params[k] != v:
            return None, None

    metrics = pd.read_csv(f, index_col=0).T[metric]
    key = (params["ssl"], params["pred"], params["seed"])

    return key, metrics


def load_all_results(results_dir=Path() / "results/",
                     pattern="**/data_imagenet/**/results_all.csv",
                     metric="err"):
    """Load all ssl and supervised results."""
    files = list(results_dir.glob(pattern))
    print(f"Found {len(files)} result files to load.")

    results = dict()

    for f in files:

        key, metrics = load_single_file(f,
                                        skip_ifneq=dict(data="imagenet"),
                                        skip_ifin=dict(exp=["dev", "inv", "shat"],
                                                       ssl=["colorization"]),
                                        metric=metric)

        if metrics is None:
            continue

        results[key] = metrics

    df_results = pd.concat(results, axis=1).T
    df_results.index.set_names(["enc", "pred", "seed"], inplace=True)

    if metric == "err":
        df_results = df_results * 100

    return df_results


def add_approximation_results_(ssl, sup, is_allow_nan=True):
    """Uses the supervised results to add approximation error to the SSL models."""

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
        ssl["sup_train_train"] = np.nan
        ssl["sup_train_test"] = np.nan

    ssl.loc[~missing, "sup_train_train"] = sup.loc[ssl_to_sup[~missing], "train_train"].values
    ssl.loc[~missing, "sup_train_test"] = sup.loc[ssl_to_sup[~missing], "train_test"].values


def format_approx_results(results, metadata, f_replace_arch=None):
    """Separates the supervised results into the approximation error column."""

    arch = np.array([metadata.loc[i, "architecture_exact"] for i in results.index.get_level_values(0)])
    results["arch"] = arch

    is_supervised = np.array([metadata.loc[i, "ssl_mode"] for i in results.index.get_level_values(0)]) == "supervised"
    results_sup = results[is_supervised].copy()
    results_ssl = results[~is_supervised].copy()

    results_sup = results_sup.set_index("arch", append=True)
    if f_replace_arch is not None:
        # replaces architectures if needed for the supervised models (in the case where couldn't find correct one)
        results_ssl.loc[:, "arch"] = results_ssl.loc[:, "arch"].apply(f_replace_arch)
    results_ssl = results_ssl.set_index("arch", append=True)

    add_approximation_results_(results_ssl, results_sup)

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

    if "resnet50d" in arch:
        # if increasing last dimension the architecture does not change much
        return "resnet50"

    if arch == "resnet50w2":
        # surprisingly I couldn't find any pytorch pretrained resnet50x2
        # so i' using wide resnet which is somewhat comparable although not ideal
        return "wide_resnet50_2"

    return arch


def make_risk_decomposition(results, traverse_path=["down", "right", "down"], is_print=False):
    """
    Make the risk decomposition depending on how you wannt to traverse the decomposition table.
    traverse_path=`None` returns decomposition table.
    """
    # all the values from the decomposition table
    dec_table = np.array([['sup_train_train', 'sup_train_test'],
                          ['train_train', 'train-cmplmnt-ntest_train-sbst-ntest'],
                          ['union_test', 'train_test']])
    results["agg_risk"] = results['train_test']

    if traverse_path is None:
        selected_columns = list(dec_table.flatten()) + ["agg_risk"]

    else:
        results["approx"] = results['sup_train_train']
        selected_columns = ["agg_risk", "usability", "enc_gen", "probe_gen", "approx", 'train_train']

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

    return results[selected_columns]


def clean_results(results,
                  metadata,
                  predictor=None,  # predictor to select
                  is_avg_seed=True,  # avg over seed
                  is_positive=True,  # force pos
                  is_add_metadata=True # adds additional metadata that are funciton of others
                  ):
    """Clean the risk decomposition results."""

    if predictor is not None:
        results = results.loc[results.index.get_level_values("pred") == predictor]
        results.index = results.index.droplevel("pred")

    if is_avg_seed:
        idcs_no_seed = [n for n in results.index.names if n != "seed"]
        results = results.reset_index().groupby(idcs_no_seed).mean()

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

    is_nan = results.isna().any(axis=1)
    if is_nan.sum() > 0:
        warnings.warn(f"The following results have some nan:")
        display(results.loc[is_nan].round(3))

def add_metadata_(metadata):
    """Adds additional metadata that are function of others"""
    metadata["nviews"] = metadata["views"].apply(lambda s: count_views(s))

    # cleanup date (chooses day = 1 arbitrarily_
    metadata["date_published"] = pd.to_datetime(dict(year=metadata.year,
                                                     month=metadata.month,
                                                     day=metadata.month * 0 + 1))


def count_views(s):
    count = 0
    while "x" in s:
        splitted = s.split("x", maxsplit=1)
        count += int(splitted[0][-1])
        s = splitted[1]
    return max(count,1)

def prepare_sklearn(results,
                    metadata_df,
                    features_to_del=["notes", "where", "top1acc_in1k_official", "n_pus", "pu_type", "time_hours",
                                     "license", "month"],
                    features_to_keep=None,
                    round_dict=dict(),
                    target="agg_risk"):
    X = metadata_df.copy()
    y = results[target].copy()

    # binarize augmentations
    binarizer = MultiLabelBinarizer().fit(X["augmentations"])
    X = pd.concat([X, pd.DataFrame(binarizer.transform(X["augmentations"]),
                                   columns=["aug_" + c for c in binarizer.classes_],
                                   index=X.index)], axis=1
                  ).drop("augmentations", axis=1)

    if features_to_keep is not None:
        features_to_del = [c for c in X.columns if c not in features_to_keep]

    X = X.drop(features_to_del, axis=1)

    for c, round_to in round_dict.items():
        X[c] = X[c] // round_to * round_to

    # convert categorical
    for c in X.columns:
        if X[c].dtype == "string":
            X[c] = X[c].astype("category")

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
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
        inp_pred = dtrain
    else:
        inp_pred = X

    rmse = mean_squared_error(y, reg.predict(inp_pred), squared=False)
    r2 = r2_score(y, reg.predict(inp_pred))
    print(f"R2: {r2}. RMSE: {rmse}")

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
        explainer = shap.TreeExplainer(reg)
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

        for i, c in enumerate(examples):
            print(c)
            display(shap.force_plot(explainer.expected_value, shap_values[X.index.get_loc(c.lower()), :],
                                    feature_names=X.columns, matplotlib=True, figsize=(12, 3)))
            # plt.gca().set_title("c", fontsize=12)

        if is_shap_interactive:
            display(shap.plots.force(explainer.expected_value, shap_values, feature_names=X.columns))

