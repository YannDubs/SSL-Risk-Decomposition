"""Entry point to aggregate a series of results obtained using `main.py` in a nice plot / table.

This should be called by `python utils/aggregate.py <conf>` where <conf> sets all configs from the cli, see
the file `config/aggregate.yaml` for details about the configs. or use `python utils/aggregate.py -h`.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import hydra
import torch
from matplotlib import MatplotlibDeprecationWarning
from omegaconf import OmegaConf

MAIN_DIR = os.path.abspath(str(Path(__file__).parents[1]))
CURR_DIR = os.path.abspath(str(Path(__file__).parents[0]))
sys.path.append(MAIN_DIR)
sys.path.append(CURR_DIR)

from utils.helpers import (  # isort:skip
    omegaconf2namespace, to_numpy
)


logger = logging.getLogger(__name__)


@hydra.main(config_path=f"{MAIN_DIR}/config", config_name="aggregate")
def main_cli(cfg):
    # uses main_cli sot that `main` can be called from notebooks.
    return main(cfg)


def main(cfg):


    begin(cfg)

    # make sure you are using primitive types from now on because omegaconf does not always work
    cfg = omegaconf2namespace(cfg)

    aggregator = ResultAggregator(pretty_renamer=PRETTY_RENAMER, **cfg.kwargs)

    base_path = aggregator.base_dir / Path(cfg.base_path)

    logger.info(f"Collecting the data ..")
    for name, pattern in cfg.patterns.items():
        if pattern is not None:
            aggregator.collect_data(
                pattern=pattern, table_name=name, **cfg.collect_data
            )

    for f in cfg.agg_mode:

        logger.info(f"Mode {f} ...")

        if f is None:
            continue

        if f in cfg:
            kwargs = cfg[f]
        else:
            kwargs = {}

        getattr(aggregator, f)(**kwargs)

    logger.info("Finished.")


def begin(cfg):
    """Script initialization."""
    OmegaConf.set_struct(cfg, False)  # allow pop
    PRETTY_RENAMER.update(cfg.kwargs.pop("pretty_renamer"))
    OmegaConf.set_struct(cfg, True)

    logger.info(f"Aggregating {cfg.experiment} ...")

# DECORATORS AND CONSTANTS
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


PRETTY_RENAMER = StrFormatter(
    exact_match={},
    substring_replace={
        # Math stuff
        "beta": r"$\beta$",
        "calFissl": r"$\mathcal{F}_{issl}$",
        "calFpred": r"$\mathcal{F}_{pred}$",
        "calF --": r"$\mathcal{F}^{-}$",
        "calF ++": r"$\mathcal{F}^{+}$",
        "calF": r"$\mathcal{F}$",
        " --": r"⁻",
        " ++": r"⁺",
        # General
        "_": " ",
        "true": "True",
        "false": "False",
        "Resnet": "ResNet",
        "Lr": "Learning Rate",
        "Test/Pred/": "",
        "Train/Pred/": "Train ",
        "Zdim": r"$\mathcal{Z}$ dim.",
        "Pred": "Downstream Pred.",
        "Repr": "ISSL",
        # Project specific
        "Acc ": "Acc. ",
        "Accuracy Score": "Acc.",
        "Std Gen Smallz": "Std. Gen. ISSL",
        "Cntr Stda": "Our Cont. ISSL",
        "Acc. Agg Min": "Worst Acc.",
        # "Acc": "Accuracy",
        "K Labels": "Max. # Labels",
        "N Tasks": "# Tasks",
        "N Samples": "# Samples",
        "2.0": "Binary",
    },
    to_upper=["Cifar10", "Mnist", "Mlp", "Adam"],
)


def get_default_args(func: Callable) -> dict:
    """Return the default arguments of a function.
    credit : https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def assert_sns_vary_only_param(
    data: pd.DataFrame, sns_kwargs: dict, param_vary_only: Optional[list]
) -> None:
    """
    Make sure that the only multi indices that have not been conditioned over for plotting and has non
    unique values are in `param_vary_only`.
    """
    if param_vary_only is not None:
        multi_idcs = data.index
        issues = []
        for idx in multi_idcs.levels:
            is_varying = len(idx.values) != 1
            is_conditioned = idx.name in sns_kwargs.values()
            is_can_vary = idx.name in param_vary_only
            if is_varying and not is_conditioned and not is_can_vary:
                issues.append(idx.name)

        if len(issues) > 0:
            raise ValueError(
                f"Not only varying {param_vary_only}. Also varying {issues}."
            )


def aggregate(
    table: Union[pd.DataFrame, pd.Series],
    cols_to_agg: list[str] = [],
    aggregates: list[str] = ["mean", "sem"],
) -> Union[pd.DataFrame, pd.Series]:
    """Aggregate values of pandas dataframe over some columns.

    Parameters
    ----------
    table : pd.DataFrame or pd.Series
        Table to aggregate.

    cols_to_agg : list of str
        list of columns over which to aggregate. E.g. `["seed"]`.

    aggregates : list of str
        list of functions to use for aggregation. The aggregated columns will be called `{col}_{aggregate}`.
    """
    if len(cols_to_agg) == 0:
        return table

    if isinstance(table, pd.Series):
        table = table.to_frame()

    new_idcs = [c for c in table.index.names if c not in cols_to_agg]
    table_agg = table.reset_index().groupby(by=new_idcs, dropna=False).agg(aggregates)
    table_agg.columns = ["_".join(col).rstrip("_") for col in table_agg.columns.values]
    return table_agg


def save_fig(
    fig: Any, filename: Union[str, bytes, os.PathLike], dpi: int, is_tight: bool = True
) -> None:
    """General function for many different types of figures."""

    # order matters ! and don't use elif!
    if isinstance(fig, sns.FacetGrid):
        fig = fig.fig

    if isinstance(fig, torch.Tensor):
        x = fig.permute(1, 2, 0)
        if x.size(2) == 1:
            fig = plt.imshow(to_numpy(x.squeeze()), cmap="gray")
        else:
            fig = plt.imshow(to_numpy(x))
        plt.axis("off")

    if isinstance(fig, plt.Artist):  # any type of axes
        fig = fig.get_figure()

    if isinstance(fig, plt.Figure):

        plt_kwargs = {}
        if is_tight:
            plt_kwargs["bbox_inches"] = "tight"

        fig.savefig(filename, dpi=dpi, **plt_kwargs)
        plt.close(fig)
    else:
        raise ValueError(f"Unknown figure type {type(fig)}")


def filename_format(arg_format: list):
    """Allows formatting of kwargs"""

    def real_decorator(fn):
        dflt_kwargs = get_default_args(fn)

        @functools.wraps(fn)
        def helper(self, filename=dflt_kwargs["filename"], **kwargs):
            dflt_kwargs.update(kwargs)

            for arg in arg_format:
                val = dflt_kwargs[arg]
                if isinstance(val, Sequence) and not isinstance(val, str):
                    val = "-".join(val)

                if isinstance(val, str) and "/" in val:
                    val = val.split("/")[-1]

                filename = filename.replace("{" + f"{arg}" + "}", str(val))

            return fn(self, filename=filename, **kwargs)

        return helper

    return real_decorator


def data_getter(fn):
    """Get the correct data."""
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self, data=dflt_kwargs["data"], filename=dflt_kwargs["filename"], **kwargs
    ):
        if data is None:
            # if None run all tables
            return [
                helper(self, data=k, filename=filename, **kwargs)
                for k in self.tables.keys()
            ]

        if isinstance(data, str):
            # cannot use format because might be other other patterns (format cannot do partial format)
            filename = filename.replace("{table}", data)
            data = self.tables[data]

        data = data.copy()

        return fn(self, data=data, filename=filename, **kwargs)

    return helper


def table_summarizer(fn):
    """Get the data and save the summarized output to a csv if needed.."""
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self, data=dflt_kwargs["data"], filename=dflt_kwargs["filename"], **kwargs
    ):

        summary = fn(self, data=data, **kwargs)

        if self.is_return_plots:
            return summary
        else:
            summary.to_csv(self.save_dir / f"{self.prfx}{filename}.csv")

    return helper


def folder_split(fn):
    """Split the dataset by the values in folder_col and call fn on each subfolder."""
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self,
        *args,
        data=dflt_kwargs["data"],
        folder_col=dflt_kwargs["folder_col"],
        filename=dflt_kwargs["filename"],
        **kwargs,
    ):
        kws = ["folder_col"]
        for kw in kws:
            kwargs[kw] = eval(kw)

        if folder_col is None:
            processed_filename = self.save_dir / f"{self.prfx}{filename}"
            return fn(self, *args, data=data, filename=processed_filename, **kwargs)

        else:
            out = []
            flat = data.reset_index(drop=False)
            for curr_folder in flat[folder_col].unique():
                curr_data = flat[flat[folder_col] == curr_folder]

                sub_dir = self.save_dir / f"{folder_col}_{curr_folder}"
                sub_dir.mkdir(parents=True, exist_ok=True)

                processed_filename = sub_dir / f"{self.prfx}{filename}"

                out.append(
                    fn(
                        self,
                        *args,
                        data=curr_data.set_index(data.index.names),
                        filename=processed_filename,
                        **kwargs,
                    )
                )
            return out

    return helper


def single_plot(fn):
    """
    Wraps any of the aggregator function to produce a single figure. THis enables setting
    general seaborn and matplotlib parameters, saving the figure if needed, and aggregating the
    data over desired indices.
    """
    dflt_kwargs = get_default_args(fn)

    @functools.wraps(fn)
    def helper(
        self,
        x,
        y,
        *args,
        data=dflt_kwargs["data"],
        folder_col=dflt_kwargs["folder_col"],
        filename=dflt_kwargs["filename"],
        cols_vary_only=dflt_kwargs["cols_vary_only"],
        cols_to_agg=dflt_kwargs["cols_to_agg"],
        aggregates=dflt_kwargs["aggregates"],
        plot_config_kwargs=dflt_kwargs["plot_config_kwargs"],
        row_title=dflt_kwargs["row_title"],
        col_title=dflt_kwargs["col_title"],
        x_rotate=dflt_kwargs["x_rotate"],
        legend_out=dflt_kwargs["legend_out"],
        is_no_legend_title=dflt_kwargs["is_no_legend_title"],
        set_kwargs=dflt_kwargs["set_kwargs"],
        **kwargs,
    ):
        filename = Path(str(filename).format(x=x, y=y))

        kws = [
            "folder_col",
            "filename",
            "cols_vary_only",
            "cols_to_agg",
            "aggregates",
            "plot_config_kwargs",
            "row_title",
            "col_title",
            "x_rotate",
            "legend_out",
            "is_no_legend_title",
            "set_kwargs",
        ]
        for kw in kws:
            kwargs[kw] = eval(kw)  # put back in kwargs

        kwargs["x"] = x
        kwargs["y"] = y

        assert_sns_vary_only_param(data, kwargs, cols_vary_only)

        data = aggregate(data, cols_to_agg, aggregates)
        pretty_data = self.prettify_(data)
        pretty_kwargs = self.prettify_kwargs(pretty_data, **kwargs)
        used_plot_config = dict(self.plot_config_kwargs, **plot_config_kwargs)

        with plot_config(**used_plot_config):
            sns_plot = fn(self, *args, data=pretty_data, **pretty_kwargs)

            # THIS WAS MESSING UP WITH heatmap (removing text). But could double write text now ? TODO: check
            # for ax in sns_plot.axes.flat:
            #    plt.setp(ax.texts, text="")
            sns_plot.set_titles(row_template=row_title, col_template=col_title)

            if x_rotate != 0:
                # calling directly `set_xticklabels` on FacetGrid removes the labels sometimes
                for axes in sns_plot.axes.flat:
                    axes.set_xticklabels(axes.get_xticklabels(), rotation=x_rotate)

            if is_no_legend_title:
                #! not going to work well if is_legend_out (double legend)
                for ax in sns_plot.fig.axes:
                    handles, labels = ax.get_legend_handles_labels()
                    if len(handles) > 1:
                        ax.legend(handles=handles[1:], labels=labels[1:])

            sns_plot.set(**set_kwargs)

            if not legend_out:
                plt.legend()

        if self.is_return_plots:
            return sns_plot
        else:
            save_fig(sns_plot, f"{filename}.pdf", dpi=self.dpi)

    return helper


# MAIN CLASS
class ResultAggregator:
    """Aggregates batches of results (multirun)

    Parameters
    ----------
    save_dir : str or Path
        Where to save all results.

    base_dir : str or Path
        Base folder from which all paths start.

    kwargs :
        Additional arguments to `PostPlotter`.
    """

    def __init__(self, save_dir, base_dir=Path(__file__).parent, is_return_plots=False,
                 prfx="", pretty_renamer=PRETTY_RENAMER, dpi=300, plot_config_kwargs={}):
        self.is_return_plots = is_return_plots
        self.prfx = prfx
        self.pretty_renamer = pretty_renamer
        self.dpi = dpi
        self.plot_config_kwargs = plot_config_kwargs
        self.base_dir = Path(base_dir)
        self.save_dir = self.base_dir / Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.table = dict()



    def prettify_(self, table):
        """Make the name and values in a dataframe prettier / human readable (inplace)."""
        idcs = table.index.names
        table = table.reset_index()  # also want to modify multi index so tmp flatten
        table.columns = [self.pretty_renamer[c] for c in table.columns]
        table = table.applymap(self.pretty_renamer)
        table = table.set_index([self.pretty_renamer[c] for c in idcs])

        # replace `None` with "None" for string columns such that can see those
        str_col = table.select_dtypes(include=object).columns
        table[str_col] = table[str_col].fillna(value="None")

        return table

    def prettify_kwargs(self, table, **kwargs):
        """Change the kwargs of plotting function such that usable with `prettify(table)`."""
        cols_and_idcs = list(table.columns) + list(table.index.names)
        return {
            # only prettify if part of the columns (not arguments to seaborn)
            k: self.pretty_renamer[v]
            if isinstance(v, str) and self.pretty_renamer[v] in cols_and_idcs
            else v
            for k, v in kwargs.items()
        }


    def collect_data(
        self,
        pattern : str=f"results/**/results_all.csv",
        metrics : list[str] = ["err", "loss"]
    ):
        """Collect all the data.

        Parameters
        ----------
        pattern : str
            Pattern for globbing data.

        metrics : list of str
            Metrics to store.
        """
        paths = list(self.base_dir.glob(pattern))
        if len(paths) == 0:
            raise ValueError(f"No files found for your pattern={pattern}")

        results = dict()
        for path in paths:

            # select everything from "exp_"
            path_clean = "exp_" + str(path.parent.resolve()).split("/exp_")[-1]

            # make dict of params
            params = path_to_params(path_clean)

            # looks like : DataFrame(param1:...,param2:..., param3:...)
            df_params = pd.DataFrame.from_dict(params, orient="index").T
            # looks like : dict(acc={comp1:..., comp2:...}, loss={comp1:..., comp2:..., err={comp1:..., comp2:...})
            df_metrics = pd.read_csv(path, index_col=0).T

            for k in df_metrics.columns:
                if k not in metrics:
                    continue
                result = pd.concat([df_params, df_metrics[k]], axis=1)
                result = result.apply(pd.to_numeric, errors="ignore")
                results[k] = results.get(k, []).append(result)

        self.param_name = list(params.keys())

        for k, v in results.items():
            self.tables[k] = pd.concat(v, axis=0).set_index(self.param_name)

    @data_getter
    @folder_split
    @table_summarizer
    def summarize_metrics(
        self,
        data=None,
        cols_to_agg=["seed"],
        aggregates=["mean", "sem"],
        filename="summarized_metrics_{table}",
        folder_col=None,
    ):
        """Aggregate all the metrics and save them.

        Parameters
        ----------
        data : pd.DataFrame or str, optional
                Dataframe to summarize. If str will use one of self.tables. If `None` uses all data
                in self.tables.

        cols_to_agg : list of str
            List of columns over which to aggregate. E.g. `["seed"]`.

        aggregates : list of str
            List of functions to use for aggregation. The aggregated columns will be called `{col}_{aggregate}`.

        filename : str, optional
                Name of the file for saving the metrics. Can interpolate {table} if from self.tables.

        folder_col : str, optional
            Name of a column that will be used to separate the tables into multiple subfolders.
        """
        return aggregate(data, cols_to_agg, aggregates)


    @data_getter
    @folder_split
    @single_plot
    def plot_scatter_lines(
        self,
        x,
        y,
        data=None,
        filename="{table}_lines_{y}_vs_{x}",
        mode="relplot",
        folder_col=None,
        logbase_x=1,
        logbase_y=1,
        sharex=True,
        sharey=False,
        legend_out=True,
        is_no_legend_title=False,
        set_kwargs={},
        x_rotate=0,
        cols_vary_only=None,
        cols_to_agg=[],
        aggregates=["mean", "sem"],
        is_x_errorbar=False,
        is_y_errorbar=False,
        row_title="{row_name}",
        col_title="{col_name}",
        plot_config_kwargs={},
        xlabel="",
        ylabel="",
        hue_order=None,
        style_order=None,
        **kwargs,
    ):
        """Plotting all combinations of scatter and line plots.

        Parameters
        ----------
        x : str
            Column name of x axis.

        y : str
            Column name for the y axis.

        data : pd.DataFrame or str, optional
            Dataframe used for plotting. If str will use one of self.tables. If `None` runs all tables.

        filename : str or Path, optional
            Path to the file to which to save the results to. Will start at `base_dir`.
            Can interpolate {x} and {y}.

        mode : {"relplot","lmplot"}, optional
            Underlying function to use from seaborn. `lmplot` can also plot the estimated regression
            line.

        folder_col : str, optional
            Name of a column that will be used to separate the plot into multiple subfolders.

        logbase_x, logbase_y : int, optional
            Base of the x (resp. y) axis. If 1 no logscale. if `None` will automatically chose.

        sharex,sharey : bool, optional
            Wether to share x (resp. y) axis.

        legend_out : bool, optional
            Whether to put the legend outside of the figure.

        is_no_legend_title : bool, optional
            Whether to remove the legend title. If `legend_out` then will actually duplicate the
            legend :/, the best in that case is to remove the text of the legend column .

        set_kwargs : dict, optional
            Additional arguments to `FacetGrid.set`. E.g.
            dict(xlim=(0,None),xticks=[0,1],xticklabels=["a","b"]).

        x_rotate : int, optional
            By how much to rotate the x labels.

        cols_vary_only : list of str, optional
            Name of the columns that can vary when plotting (e.g. over which to compute bootstrap CI).
            This ensures that you are not you are not taking averages over values that you don't want.
            If `None` does not check. This is especially useful for

        cols_to_agg : list of str
            List of columns over which to aggregate. E.g. `["seed"]`. In case the underlying data
            are given at uniform intervals X, this is probably not needed as seaborn's line plot will
            compute the bootstrap CI for you.

        aggregates : list of str
            List of functions to use for aggregation. The aggregated columns will be called
            `{col}_{aggregate}`.

        is_x_errorbar,is_y_errorbar : bool, optional
            Whether to standard error (over the aggregation of cols_to_agg) as error bar . If `True`,
            `cols_to_agg` should not be empty and `"sem"` should be in `aggregates`.

        row_title,col_title : str, optional
            Template for the titles of the Facetgrid. Can use `{row_name}` and `{col_name}`
            respectively.

        hue_order, style_order : list of str, optional
            Specify the order of processing and plotting for categorical levels of the hue / style semantic.

        plot_config_kwargs : dict, optional
            General config for plotting, e.g. arguments to matplotlib.rc, sns.plotting_context,
            matplotlib.set ...

        kwargs :
            Additional arguments to underlying seaborn plotting function. E.g. `col`, `row`, `hue`,
            `style`, `size` ...
        """

        kwargs["x"] = x
        kwargs["y"] = y

        if hue_order is not None:
            # prettify hue order => can give in not prettified version
            kwargs["hue_order"] = [self.pretty_renamer[h] for h in hue_order]

        if style_order is not None:
            # prettify hue order => can give in not prettified version
            kwargs["style_order"] = [self.pretty_renamer[h] for h in style_order]

        if is_x_errorbar or is_y_errorbar:
            if (len(cols_to_agg) == 0) or ("sem" not in aggregates):
                logger.warning(
                    f"Not plotting errorbars due to empty cols_to_agg={cols_to_agg} or 'sem' not in aggregates={aggregates}."
                )
                is_x_errorbar, is_y_errorbar = False, False

        if mode == "relplot":
            used_kwargs = dict(
                legend="full",
                kind="line",
                markers=True,
                facet_kws={
                    "sharey": sharey,
                    "sharex": sharex,
                    "legend_out": legend_out,
                },
                style=kwargs.get("hue", None),
            )
            used_kwargs.update(kwargs)

            sns_plot = sns.relplot(data=data, **used_kwargs)

        elif mode == "lmplot":
            used_kwargs = dict(
                legend="full", sharey=sharey, sharex=sharex, legend_out=legend_out,
            )
            used_kwargs.update(kwargs)

            sns_plot = sns.lmplot(data=data, **used_kwargs)

        else:
            raise ValueError(f"Unknown mode={mode}.")

        if is_x_errorbar or is_y_errorbar:
            xerr, yerr = None, None
            if is_x_errorbar:
                x_sem = x.rsplit(" ", maxsplit=1)[0] + " Sem"  # _mean -> _sem
                xerr = data[x_sem]

            if is_y_errorbar:
                y_sem = y.rsplit(" ", maxsplit=1)[0] + " Sem"  # _mean -> _sem
                yerr = data[y_sem]

            sns_plot.map_dataframe(add_errorbars, yerr=yerr, xerr=xerr)

        if logbase_x != 1 or logbase_y != 1:
            sns_plot.map_dataframe(
                set_log_scale, basex=logbase_x, basey=logbase_y, **kwargs
            )

        # TODO remove when waiting for https://github.com/mwaskom/seaborn/issues/2456
        if xlabel != "":
            for ax in sns_plot.fig.axes:
                ax.set_xlabel(xlabel)

        if ylabel != "":
            for ax in sns_plot.fig.axes:
                ax.set_ylabel(ylabel)

        sns_plot.tight_layout()

        return sns_plot



# HELPERS

def path_to_params(path):
    """Take a path name of the form `param1_value1/param2_value2/...` and returns a dictionary."""
    params = {}

    for name in path.split("/"):
        if "_" in name:
            k, v = name.split("_", maxsplit=1)
            params[k] = v

    return params


def get_param_in_kwargs(data, **kwargs):
    """
    Return all arguments that are names of the multiindex (i.e. param) of the data. I.e. for plotting
    this means that you most probably conditioned over them.
    """
    return {
        n: col
        for n, col in kwargs.items()
        if (isinstance(col, str) and col in data.index.names)
    }


def add_errorbars(data, yerr, xerr, **kwargs):
    """Add errorbar to each sns.facetplot."""
    datas = [data]
    if xerr is not None:
        datas += [xerr.rename("xerr")]
    if yerr is not None:
        datas += [yerr.rename("yerr")]

    df = pd.concat(datas, axis=1).set_index(["hue", "style"])

    for idx in df.index.unique():
        # error bars will be different for different hue and style
        df_curr = df.loc[idx, :] if len(df.index.unique()) > 1 else df
        errs = dict()
        if xerr is not None:
            errs["xerr"] = df_curr["xerr"]
        if yerr is not None:
            errs["yerr"] = df_curr["yerr"]

        plt.errorbar(
            df_curr["x"].values,
            df_curr["y"].values,
            fmt="none",
            ecolor="lightgray",
            **errs,
        )


def set_log_scale(data, basex, basey, **kwargs):
    """Set the log scales as desired."""
    x_data = data[kwargs["x"]].unique()
    y_data = data[kwargs["y"]].unique()
    plt.xscale(**kwargs_log_scale(x_data, base=basex))
    plt.yscale(**kwargs_log_scale(y_data, base=basey))


def kwargs_log_scale(unique_val, mode="equidistant", base=None):
    """Return arguments to set log_scale as one would wish.

    Parameters
    ----------
    unique_val : np.array
        All unique values that will be plotted on the axis that should be put in log scale.

    axis : {"x","y"}
        Axis for which to use log_scales.

    mode : ["smooth","equidistant"], optional
        How to deal with the zero, which cannot be dealt with by default as log would give  -infinity.
        The key is that we will use a region close to zero which is linear instead of log.
        In the case of `equidistant` we use ensure that the large tick at zero is at the same distance
        of other ticks than if there was no linear. The problem is that this will give rise to
        nonexistent kinks when the plot goes from linear to log scale. `Smooth` tries to deal
        with that by smoothly varying between linear and log. For examples see
        https://github.com/matplotlib/matplotlib/issues/7008.

    base : int, optional
        Base to use for the log plot. If `None` automatically tries to find it. If `1` doesn't use
        any log scale.
    """
    unique_val.sort()

    # automatically compute base
    if base is None:
        # take avg multiplier between each consecutive elements as base i.e 2,8,32 would be 4
        # but 0.1,1,10 would be 10
        diffs = unique_val[unique_val > 0][1:] / unique_val[unique_val > 0][:-1]
        base = int(diffs.mean().round())

    # if constant diff don't use logscale
    if base == 1 or np.diff(unique_val).var() == 0:
        return dict(value="linear")

    # only need to use symlog if there are negative values (i.e. need some linear region)
    if (unique_val <= 0).any():
        min_nnz = np.abs(unique_val[unique_val != 0]).min()
        if mode == "smooth":
            linscale = np.log(np.e) / np.log(base) * (1 - (1 / base))
        elif mode == "equidistant":
            linscale = 1 - (1 / base)
        else:
            raise ValueError(f"Unkown mode={mode}")

        return {
            "value": "symlog",
            "linthresh": min_nnz,
            "base": base,
            "subs": list(range(base)),
            "linscale": linscale,
        }
    else:
        return {
            "value": "log",
            "base": base,
            "subs": list(range(base)),
        }

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


@contextlib.contextmanager
def plot_config(
    style="ticks",
    context="notebook",
    palette="colorblind",
    font_scale=1.5,
    font="sans-serif",
    is_ax_off=False,
    is_rm_xticks=False,
    is_rm_yticks=False,
    rc={"lines.linewidth": 2},
    set_kwargs=dict(),
    despine_kwargs=dict(),
    # pretty_renamer=dict(), #TODO
):
    """Temporary seaborn and matplotlib figure style / context / limits / ....

    Parameters
    ----------
    style : dict, None, or one of {darkgrid, whitegrid, dark, white, ticks}
        A dictionary of parameters or the name of a preconfigured set.

    context : dict, None, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.

    palette : string or sequence
        Color palette, see :func:`color_palette`

    font : string
        Font family, see matplotlib font manager.

    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.

    is_ax_off : bool, optional
        Whether to turn off all axes.

    is_rm_xticks, is_rm_yticks : bool, optional
        Whether to remove the ticks and labels from y or x axis.

    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        style dictionaries.

    set_kwargs : dict, optional
        kwargs for matplotlib axes. Such as xlim, ylim, ...

    despine_kwargs : dict, optional
        Arguments to `sns.despine`.
    """
    defaults = plt.rcParams.copy()

    try:
        rc["font.family"] = font
        plt.rcParams.update(rc)

        with sns.axes_style(style=style, rc=rc), sns.plotting_context(
            context=context, font_scale=font_scale, rc=rc
        ), sns.color_palette(palette):
            yield
            last_fig = plt.gcf()
            for i, ax in enumerate(last_fig.axes):
                ax.set(**set_kwargs)

                if is_ax_off:
                    ax.axis("off")

                if is_rm_yticks:
                    ax.axes.yaxis.set_ticks([])

                if is_rm_xticks:
                    ax.axes.xaxis.set_ticks([])

        sns.despine(**despine_kwargs)

    finally:
        with warnings.catch_warnings():
            # filter out depreciation warnings when resetting defaults
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            # reset defaults
            plt.rcParams.update(defaults)




if __name__ == "__main__":
    main_cli()
