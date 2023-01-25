import pdb

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from IPython.core.display import display

from utils.collect_results import COMPONENTS

from statsmodels.tools.eval_measures import rmse

from utils.helpers import ols_clean_df_
import graphviz as gr


def ols_summary(df,
                treatment,
                alpha=0.05,
                outcome="value",
                condition=["non_vary", "{treatment}"],
                objectives=COMPONENTS,
                f_outcome="",  # "log", "delta_log"
                n_tune=50,
                is_short=False,
                ):
    df = df.copy()


    condition = [c.format(treatment=treatment) for c in condition]


    ols_clean_df_(df, condition + [outcome])

    if "log" in f_outcome:
        y = f"np.log({outcome})"
        f = np.exp
    else:
        y = outcome
        f = lambda x: x

    for o in objectives:
        best_delta = 0
        formula = f"{y} ~  {' + '.join(condition)}"
        curr_df = df.loc[df.component == o]
        if "delta" not in f_outcome:
            best_model = smf.ols(formula=formula, data=df[df.component == o]).fit()
            best_rmse = rmse(curr_df[outcome], f(best_model.predict(curr_df)))
        else:
            best_rmse = float("inf")
            best_model = None
            epsilon_deltas = 10.0 ** np.arange(-5, 0)
            add_deltas = list(epsilon_deltas) + list(-epsilon_deltas) + [0]
            for delta in list(np.linspace(1,
                                          curr_df[outcome].min(),
                                          num=n_tune - len(add_deltas))) + add_deltas:
                try:
                    delta_df = curr_df.copy()
                    delta_df[outcome] = delta_df[outcome] - delta
                    with np.errstate(divide='ignore', invalid='ignore'):
                        model = smf.ols(formula=formula, data=delta_df).fit()
                    curr_rmse = rmse(delta_df[outcome], f(model.predict(delta_df)))
                    if curr_rmse < best_rmse:
                        best_model = model
                        best_rmse = curr_rmse
                        best_delta = delta
                except:
                    pass

        if best_model is None:
            print(f"Could not find a model for {o}")
            continue

        p_values = best_model.summary2().tables[1]["P>|t|"]
        treat_cols = [i for i in p_values.index if treatment in i]

        if (p_values[treat_cols] < alpha).any():
            summary = best_model.summary(title=o)
            if is_short:
                print(o)
                print(f"rmse: {best_rmse}, delta: {best_delta}")
                display(best_model.summary2().tables[1].loc[treat_cols])

            else:
                print(summary.tables[0])
                print(f"rmse: {best_rmse}, delta: {best_delta}")
                print(summary.tables[1])

            print()


def nodes2feat(node):
    replace = dict(data="pretraining_data", bs="batch_size", obj="objective", aug="augmentations", n_views="nviews",
                   arch="architecture", n_param='n_parameters', n_aug="n_augmentations")

    if node in replace:
        return replace[node]

    return node


TO_LOG = ["nviews", "z_dim", "batch_size", "n_parameters", "epochs"]


def causal_graph(treatment, return_to_condition=True, is_log=True):
    g = gr.Digraph()

    top_hypopt = ["data"]
    core_hypopt = ["epochs", "batch_size", "nviews", "n_augmentations", "projection2_arch"]

    if treatment == "objective":
        top_hypopt += ["objective"]
    else:
        top_hypopt += ["ssl_mode"]

    if treatment == "architecture":
        core_hypopt += ["architecture"]
    else:
        core_hypopt += ["n_parameters", "family", "z_dim", "patch_size"]

    for i in ["year", "is_industry"]:
        for j in core_hypopt + top_hypopt + ["outcome"]:
            g.edge(i, j)
    g.edge("is_official", j)

    for i in top_hypopt:
        g.edge(i, "outcome")
        for j in core_hypopt:
            g.edge(i, j)
            g.edge(j, "outcome")

    g.node(treatment, treatment, color="blue")

    if return_to_condition:
        to_condition = [nodes2feat(n) for n in top_hypopt + core_hypopt + ["year", "is_industry", "is_official"]]
        to_condition = [c for c in to_condition if c != treatment]
        if is_log:
            to_condition = [f"np.log({c})" if c in TO_LOG else c for c in to_condition ]

        return g, to_condition

    return g

