import pdb

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

from utils.collect_results import COMPONENTS

from statsmodels.tools.eval_measures import rmse



def ols_summary(df,
                treatment,
                alpha=0.05,
                outcome="value",
                condition=["non_vary", "{treatment}"],
                objectives=COMPONENTS,
                f_outcome="",  # "log", "delta_log"
                n_tune=50,
                ):
    df = df.copy()


    condition = [c.format(treatment=treatment) for c in condition]

    for c in condition + [outcome]:
        if "(" in c and ")" in c:
            # take column without function applied
            c = c[c.find("(") + 1:c.find(")")]
        if ":" in c:
            continue

        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
        elif isinstance(df[c].dtype, pd.Int64Dtype):
            df[c] = df[c].astype(int)
        elif isinstance(df[c].dtype, pd.BooleanDtype):
            df[c] = df[c].astype(bool)

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
                delta_df = curr_df.copy()
                delta_df[outcome] = delta_df[outcome] - delta
                with np.errstate(divide='ignore', invalid='ignore'):
                    model = smf.ols(formula=formula, data=delta_df).fit()
                curr_rmse = rmse(delta_df[outcome], f(model.predict(delta_df)))
                if curr_rmse < best_rmse:
                    best_model = model
                    best_rmse = curr_rmse
                    best_delta = delta

        p_values = best_model.summary2().tables[1]["P>|t|"]
        treat_cols = [i for i in p_values.index if treatment in i]

        if (p_values[treat_cols] < alpha).any():
            summary = best_model.summary(title=o)
            print(summary.tables[0])
            print(f"rmse: {best_rmse}, delta: {best_delta}")
            print(summary.tables[1])
            print()