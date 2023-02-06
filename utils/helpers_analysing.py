from itertools import chain, combinations

import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype


def _prettify_df(df, pretty_renamer, index=None):
    df = df.copy()
    df = df.rename(columns=pretty_renamer)
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            df[c] = df[c].apply(lambda x: pretty_renamer[x])
    if index == "join":
        df = df.rename(index=lambda x: " ".join(clean_model_name(x, pretty_renamer)))
    elif index == "model_arch":
        df = df.rename(index=lambda x: clean_model_name(x, pretty_renamer)[0])
    return df

def ols_clean_df_(df, columns):
    for c in columns:

        for s in ["*", ":", "+"]:
            if s in c:
                ols_clean_df_(df, c.split(s))

        if "(" in c and ")" in c:
            # take column without function applied
            c = c[c.find("(") + 1:c.find(")")]

        if c not in df:
            continue

        # to numeric if possible
        df[c] = pd.to_numeric(df[c], errors='ignore')

        if isinstance(df[c].dtype, pd.StringDtype):
            df[c] = df[c].astype("object")
        elif isinstance(df[c].dtype, pd.Int64Dtype):
            df[c] = df[c].astype(int)
        elif isinstance(df[c].dtype, pd.BooleanDtype):
            df[c] = df[c].astype(bool)

def clean_model_name(name, pretty_renamer):
    name = name.replace(" ", "_")
    if name.count("_") >= 2:
        *model_arch, rest = name.split("_", 2)
        model_arch = "_".join(model_arch)
    else:
        model_arch = name
        rest = ""
    model_arch = pretty_renamer[model_arch]
    return model_arch, rest


def min_max_scale(col):
    """Scale a column to [0, 1]."""
    return (col - col.min()) / (col.max() - col.min())


def powerset(iterable, rm_empty=True):
    "Compute power set of list"
    s = list(iterable)
    out = [s for s in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]
    if rm_empty:
        out = out [1:]
    return out

class StrFormatter:
    """String formatter that acts like some default dictionary `"formatted" == StrFormatter()["to_format"]`.

    Parameters
    ----------
    exact_match : dict, optional
        dictionary of strings that will be replaced by exact match.

    substring_replace : dict, optional
        dictionary of substring that will be replaced if no exact_match. Order matters.
        Everything is title case at this point.

    to_upper : list, optional
        Words that should be upper cased.
    """

    def __init__(self, exact_match={}, substring_replace={}, to_upper=[]):
        self.exact_match = exact_match
        self.substring_replace = substring_replace
        self.to_upper = to_upper

    def __getitem__(self, key):
        if not isinstance(key, str):
            return key

        if key in self.exact_match:
            return self.exact_match[key]

        key = key.title()

        for match, replace in self.substring_replace.items():
            key = key.replace(match, replace)

        for w in self.to_upper:
            key = key.replace(w, w.upper())

        return key

    def __call__(self, x):
        return self[x]

    def update(self, new_dict):
        """Update the substring replacer dictionary with a new one (missing keys will be prepended)."""
        self.substring_replace = update_prepending(self.substring_replace, new_dict)

def update_prepending(to_update, new):
    """Update a dictionary with another. the difference with .update, is that it puts the new keys
    before the old ones (prepending)."""
    # makes sure don't update arguments
    to_update = to_update.copy()
    new = new.copy()

    # updated with the new values appended
    to_update.update(new)

    # remove all the new values => just updated old values
    to_update = {k: v for k, v in to_update.items() if k not in new}

    # keep only values that ought to be prepended
    new = {k: v for k, v in new.items() if k not in to_update}

    # update the new dict with old one => new values are at the beginning (prepended)
    new.update(to_update)

    return new
