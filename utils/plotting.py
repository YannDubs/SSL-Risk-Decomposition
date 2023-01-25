import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import shap
import xgboost as xgb
from matplotlib.colors import SymLogNorm, LogNorm
from utils.helpers import save_fig

import contextlib
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
import seaborn as sns

from utils.collect_results import COMPONENTS, COMPONENTS_ONLY_IMP
from utils.helpers import _prettify_df, min_max_scale, clean_model_name
import math

from utils.pretty_renamer import PRETTY_RENAMER

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
        is_despine=True,
        rc={"lines.linewidth": 2},
        is_use_tex=False,
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

    is_use_tex : bool, optional
        Whether to use tex for the labels.

    set_kwargs : dict, optional
        kwargs for matplotlib axes. Such as xlim, ylim, ...

    despine_kwargs : dict, optional
        Arguments to `sns.despine`.
    """
    defaults = plt.rcParams.copy()

    try:
        rc["font.family"] = font
        if is_use_tex:
            rc["text.usetex"] = True
        else:
            rc["text.usetex"] = False
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


        if is_despine:
            sns.despine(**despine_kwargs)

    finally:
        with warnings.catch_warnings():
            # filter out depreciation warnings when resetting defaults
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            # reset defaults
            plt.rcParams.update(defaults)

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta





def get_radar_data(results,
                   min_max="pre_select",
                   models=slice(None),
                   components=COMPONENTS,
                   decimals=2,
                   pretty_renamer={}):
    """Prepares the data for the radar chart by min max scaling and prettifying."""

    assert min_max in ["pre_select", "post_select", None]

    radar_data = results[components]

    if min_max == "pre_select":
        radar_data = radar_data.apply(min_max_scale, axis=0)

    radar_data = radar_data.loc[models, :].round(decimals=decimals).sort_values("agg_risk", ascending=True)

    radar_data = radar_data.rename(columns=pretty_renamer).rename(index=pretty_renamer)

    if min_max == "post_select":
        radar_data = radar_data.apply(min_max_scale, axis=0)

    return radar_data


def plot_radar(ax, theta, metrics, title=None, rgrids=[0, 1 / 3, 2 / 3, 1], ylim=(0, 1),
               is_ticks_label=False, labels=None, color=None):
    """Plot a single radar plot"""

    if title is not None:
        ax.set_title(title, position=(0.5, 1.1), ha='center')

    ax.set_ylim(*ylim)
    if is_ticks_label:
        ax.set_rgrids(rgrids)
    else:
        ax.set_rgrids(rgrids, labels="")

    line = ax.plot(theta, 1 - metrics, color=color)
    ax.fill(theta, 1 - metrics, alpha=0.25, color=color)

    if labels is not None:
        ax.set_varlabels(labels)

        angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles)
        #labels = []

        for i, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):

            x, y = label.get_position()
            lab = ax.text(x, y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            if i not in [2, 3]:  # don't rotate bottom
                lab.set_rotation(angle)
            #labels.append(lab)
        ax.set_xticklabels([])

    else:
        ax.set_xticklabels([])
        pass




def plot_radar_grid(results,
                    ncols=1,
                    save_path=None,
                    config_kwargs=dict(font_scale=1),
                    pretty_renamer=PRETTY_RENAMER,
                    is_plot_avg=True,
                    is_label_first: bool = True,
                    is_tex: bool = True,
                    space_per_col=3,
                    space_per_row=2.3,
                    is_plot_rest=True,
                    pad_inches=0.2,
                    **kwargs):
    """Plots a grid of radar plots"""
    with plot_config(**config_kwargs):
        results = results.copy()

        radar_data = get_radar_data(results, **kwargs)

        columns = radar_data.columns
        theta = radar_factory(len(columns), frame='polygon')

        n_plots = len(radar_data)
        if is_plot_avg:
            n_plots += 1
            rest_data = radar_data
            mean_kwargs = copy.deepcopy(kwargs)
            mean_kwargs["models"] = slice(None)
            all_radar_data = get_radar_data(results, **mean_kwargs)
            first_data = all_radar_data.mean(axis=0)
        else:
            first_data = radar_data.iloc[0, :]
            rest_data = radar_data.iloc[1:, :]  # will be empty

        colors = sns.color_palette("colorblind", n_colors=n_plots)
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(ncols * space_per_col, nrows * space_per_row),
                                 subplot_kw=dict(projection='radar'),
                                 squeeze=False)
        flat_axes = axes.reshape(-1)

        # first plot
        if is_label_first:
            if is_tex:
                labels = [rf"\Large{{{pretty_renamer[c]}}}" for c in columns]
            else:
                labels = [pretty_renamer[c] for c in columns]
        else:
            labels = None

        plot_radar(flat_axes[0], theta, first_data,
                   title=None,
                   is_ticks_label=False,
                   labels=labels,
                   color=colors[0])

        for i, (model, row) in enumerate(rest_data.iterrows(), start=1):
            model_arch, rest = clean_model_name(model, pretty_renamer=pretty_renamer)
            #rest = pretty_renamer[rest]
            if is_plot_rest:
                title = (rf"\Large{{{model_arch}}}"
                         "\n"
                         rf"\large{{{rest}}}")
            else:
                title = rf"\Large{{{model_arch}}}"
            plot_radar(flat_axes[i], theta, row, title=title, is_ticks_label=False, color=colors[i])

        if len(rest_data) > 0:
            for j in range(i+1, len(flat_axes)):
                flat_axes[j].axis("off")

        plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=pad_inches)


def plot_trend_ax(data, is_min=False, ax=None, pretty_renamer=PRETTY_RENAMER, **kwargs):
    curr_df = data.copy()
    if is_min:
        curr_df = curr_df.loc[curr_df.groupby('year').agg_risk.idxmin()]

    curr_df = curr_df[COMPONENTS_ONLY_IMP + ["year"]]
    curr_df.columns = [pretty_renamer[c] for c in curr_df.columns]
    values = curr_df.groupby("Year").mean()
    if ax is None:
        ax = plt.gca()

    ax = values.plot.area(alpha=0.7, ax=ax, **kwargs)
    ax.set_ylabel("Error")

    return ax


def plot_trend(df, is_min=False, save_path=None, figsize=(6.5, 5), **kwargs):
    """Plots a stacked plot to understand how components changed over time."""
    with plot_config(rc={'lines.linewidth': 2, 'font.family': 'sans-serif',
                         "ytick.labelsize": 13, "xtick.labelsize": 13},
                     ):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        plot_trend_ax(df, is_min=is_min, ax=ax, **kwargs)

        ax.xaxis.get_major_locator().set_params(integer=True)

        plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)


def plot_trend_split(df, split, ordering=None, is_min=False, save_path=None,
                     pretty_renamer=PRETTY_RENAMER, is_legend=True):
    curr_df = df.copy()

    curr_df[split] = [pretty_renamer[s] for s in curr_df[split]]

    if ordering is not None:
        curr_df[split] = curr_df[split].astype("category").cat.set_categories(ordering)

    with plot_config(font_scale=1.3,
                     rc={'lines.linewidth': 2, 'font.family': 'sans-serif',
                         "ytick.labelsize": 13, "xtick.labelsize": 13}):
        g = sns.FacetGrid(curr_df, col=split)
        g = g.map_dataframe(plot_trend_ax, is_min=is_min)
        if is_legend:
            g.add_legend()
        g.set_titles("{col_name}")
        g.set_axis_labels(x_var="Year", y_var="Error")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)

def plot_shap_importance(model, X, y=None, n_feat=7, is_avg_imp=True, plot_size=None, pretty_renamer=None, **kwargs):

    if pretty_renamer is not None:
        feature_names = [PRETTY_RENAMER[c] for c in X.columns]
    else:
        feature_names = X.columns

    dtrain = xgb.DMatrix(X, label=y,
                         enable_categorical=True,
                         feature_names=feature_names)
    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    shap_values = explainer(dtrain.get_data())
    shap.summary_plot(shap_values, X,
                      show=False,
                      plot_size=plot_size,
                      max_display=n_feat,
                      color_bar=False,
                      plot_type="bar" if is_avg_imp else None,
                      color="tab:blue",
                     **kwargs)
    plt.gca().set_xlabel("mean(|SHAP|)", fontsize=12)
    return plt.gca()

def plot_shap_components(param, df_shap,height=2, aspect=4,is_normalize=False, config_kwargs={},
                         rc={'lines.linewidth': 2, "xtick.labelsize": 12, "legend.fontsize": 12},
                         is_colorbar=False,
                         **kwargs):
    variable = PRETTY_RENAMER[param]
    shap_prfx = PRETTY_RENAMER["normshap_"] if is_normalize else PRETTY_RENAMER["shap_"]
    other={}

    if is_colorbar:
        kwargs["legend"] = False

    with plot_config(font_scale=1.4,
                     rc=rc,
                     **config_kwargs):

        g=sns.catplot(data=_prettify_df(df_shap, pretty_renamer=PRETTY_RENAMER),
                      x=f"{shap_prfx}{variable}",
                      y="Component",
                      hue=variable,
                      aspect=aspect,
                      kind="strip",
                      height=height,
                      **kwargs)

        plt.axvline(color = "gray", alpha=0.5, linewidth=1.5)
        for y in g.ax.get_yticks():
            plt.axhline(y=y, color = "gray", alpha=0.5, linewidth=0.5, linestyle=(0, (1, 5)))
        g.set(xlabel=shap_prfx.strip(), ylabel=None)

        if g._legend is not None:
            g._legend.set_title(None)

        if is_colorbar:
            HueNorm = type(kwargs.get("hue_norm", plt.Normalize()))
            norm = HueNorm(df_shap[param].min(), df_shap[param].max())
            cmap = sns.cubehelix_palette(light=1, as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = g.figure.colorbar(sm, pad=0.01, aspect=10, drawedges=False)
            cb.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"], width=0.5, length=4, which="both")
            cb.outline.set_color('white')
            cb.outline.set_linewidth(2)
            cb.dividers.set_color('white')
            cb.dividers.set_linewidth(2)
            other["colorbar"] = cb

    return g, other


