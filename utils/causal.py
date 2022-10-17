import statsmodels.formula.api as smf
import numpy as np

from utils.collect_results import COMPONENTS


def ols_summary(df, treatment, alpha=0.05, outcome="value", condition="{treatment}"):
    df = df.copy()
    for c in COMPONENTS:
        if df[treatment].dtype == "string[python]":
            df[treatment] = df[treatment].astype("object")

        condition = condition.format(treatment=treatment)
        model = smf.ols(formula=f"{outcome} ~ non_vary + {condition}", data=df[df.component == c]).fit()
        p_values = model.summary2().tables[1]["P>|t|"]
        treat_cols = [i for i in p_values.index if treatment in i]

        if (p_values[treat_cols] < alpha).any():
            summary = model.summary(title=c)
            print(summary.tables[0])
            print(summary.tables[1])
            print()