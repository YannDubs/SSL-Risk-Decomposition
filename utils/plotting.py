import os
import pdb
from typing import Any, Union

import numpy as np

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import contextlib
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning
import seaborn as sns

from utils.collect_results import COMPONENTS
from utils.helpers import to_numpy, min_max_scale
import math

from utils.pretty_renamer import PRETTY_RENAMER
#plt.rcParams['text.usetex'] = True

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

def save_fig(
    fig: Any, filename: Union[str, bytes, os.PathLike], dpi: int=300, is_tight: bool = True
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
        labels = []

        for i, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):

            x, y = label.get_position()
            lab = ax.text(x, y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va())
            if i not in [2, 3]:  # don't rotate bottom
                lab.set_rotation(angle)
            labels.append(lab)
        ax.set_xticklabels([])

    else:
        ax.set_xticklabels([])
        pass


def plot_radar_grid(results, ncols=1,
                    save_path=None,
                    config_kwargs=dict(font_scale=1),
                    pretty_renamer=PRETTY_RENAMER,
                    **kwargs):
    """Plots a grid of radar plots"""
    with plot_config(**config_kwargs):
        results = results.copy()

        radar_data = get_radar_data(results, **kwargs)

        columns = radar_data.columns
        theta = radar_factory(len(columns), frame='polygon')

        if len(radar_data) == 1:
            first_data = radar_data.iloc[0, :]
            rest_data = radar_data.iloc[1:, :]  # will be empty
            n_plots = 1
        else:
            first_data = radar_data.mean(axis=0)
            rest_data = radar_data
            n_plots = 1 + len(radar_data)  # add the avg

        colors = sns.color_palette("colorblind", n_colors=n_plots)
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=(ncols * 3.3, nrows * 2.3),
                                 subplot_kw=dict(projection='radar'),
                                 squeeze=False)
        flat_axes = axes.reshape(-1)

        # first plot
        plot_radar(flat_axes[0], theta, first_data,
                   title=None,
                   is_ticks_label=False,
                   labels=[rf"\LARGE{{{pretty_renamer[c]}}}" for c in columns],
                   color=colors[0])

        for i, (model, row) in enumerate(rest_data.iterrows(), start=1):
            model = model.replace(" ", "_")
            if model.count("_") >= 2:
                *model_arch, rest = model.split("_", 2)
                model_arch = "_".join(model_arch)
            else:
                model_arch = model
                rest = ""
            model_arch = pretty_renamer[model_arch]
            #rest = pretty_renamer[rest]
            title = (rf"\Large{{{model_arch}}}"
                     "\n"
                     rf"\normalsize{{{rest}}}")
            plot_radar(flat_axes[i], theta, row, title=title, is_ticks_label=False, color=colors[i])
        for j in range(i+1, len(flat_axes)):
            flat_axes[j].axis("off")

        plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)