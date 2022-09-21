import warnings

import numpy as np
import pandas as pd
from pathlib import Path

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
        warnings.warn(f"Found missing supervised models for: \n {ssl_to_sup[missing].to_frame(index=False)}")
        ssl["sup_train_train"] = np.nan
        ssl["sup_train_test"] = np.nan

    ssl.loc[~missing, "sup_train_train"] = sup.loc[ssl_to_sup[~missing], "train_train"].values
    ssl.loc[~missing, "sup_train_test"] = sup.loc[ssl_to_sup[~missing], "train_test"].values


def format_approx_results(results, metadata, f_replace_arch=None):
    """Separates the supervised results into the approximation error column."""

    arch = np.array([metadata.loc[i, "architecture_exact"] for i in results.index.get_level_values(0)])
    results["arch"] = arch

    is_supervised = np.array([metadata.loc[i, "ssl_mode"] for i in results.index.get_level_values(0)]) == "supervised"
    results_sup = results[is_supervised]
    results_ssl = results[~is_supervised]

    results_sup = results_sup.set_index("arch", append=True)
    if f_replace_arch is not None:
        # replaces architectures if needeed for the supervised models (in the case wehre couldn't find correct one)
        results_ssl["arch"] = results_ssl["arch"].apply(f_replace_arch)
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
    arch = arch.replace("beit", "vit")

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
        selected_columns = ["agg_risk", "usability", "enc_gen", "probe_gen", "approx"]

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
                  threshold_bad_ifnew=50,  # used to flag bad or good models
                  threshold_bad_ifold=90,
                  new_old_threshold=2020,
                  is_positive=True  # force pos
                  ):
    """Clean and validates the risk decomposition results."""

    if predictor is not None:
        results = results.loc[results.index.get_level_values("pred") == predictor]
        results.index = results.index.droplevel("pred")

    if is_avg_seed:
        idcs_no_seed = [n for n in results.index.names if n != "seed"]
        results = results.reset_index().groupby(idcs_no_seed).mean()

    metadata = metadata.loc[results.index.get_level_values("enc")]

    bad_new = results[(results["agg_risk"] > threshold_bad_ifnew) & (metadata["year"] >= new_old_threshold)]
    bad_old = results[(results["agg_risk"] > threshold_bad_ifold) & (metadata["year"] < new_old_threshold)]
    bad = pd.concat([bad_new, bad_old], axis=0)
    warnings.warn(f"The following results seem suspicously bad: \n {bad[['agg_risk']]}")

    if is_positive:
        idx_neg = (results < 0).any(axis=1)
        if idx_neg.sum() > 0:
            warnings.warn(f"Found negative values: \n {results[idx_neg].round(3)}")
        results = results.clip(lower=0)

    return results, metadata
