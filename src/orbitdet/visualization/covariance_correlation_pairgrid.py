"""Corner-plot view of an estimation output's covariance.

One figure shows the same covariance three ways, in the layout of a corner
plot built on a :class:`seaborn.PairGrid`:

* **diagonal** — the formal 1σ standard deviation of each parameter,
* **lower triangle** — the pairwise covariance ellipses,
* **upper triangle** — the pairwise correlation coefficient.

The redundancy is deliberate.  The ellipse shows the *shape* of a pair's
uncertainty, the correlation number quantifies it, and the diagonal gives the
scale each pair is normalized by, so a reader can move between the three
without consulting another figure.

Everything is derived from one matrix.  When ``rotate_to_rsw`` is set the
covariance is rotated into the RSW frame first and the correlations are
recomputed from the rotated matrix, so the lower and upper triangles always
describe the same matrix (the correlations of the inertial matrix would not
match the rotated ellipses).
"""

from __future__ import annotations

import functools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, Rectangle
from tudatpy.dynamics import environment as env
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.reproducibility import RuntimeContext
from orbitdet.visualization.covariance_ellipses import (
    _get_inertial_state_at_epoch,
    _rotate_covariance_to_rsw,
)
from orbitdet.visualization.estimation_base import EstimationPlot, _cfg_get


def _unshare_axes(ax) -> None:
    """Detach *ax* from the grid's shared x/y groups.

    ``PairGrid`` always builds its axes with ``sharex=True, sharey=True``, and
    matplotlib offers no public way to undo that (``Axes.sharex`` documents
    "it is not possible to unshare axes").  Every cell in this figure has its
    own units — a standard deviation, an ellipse extent, a correlation — so the
    sharing is undone by removing the axes from the shared groups and clearing
    the back-references that propagate limits between siblings.
    """
    for name in ("x", "y"):
        ax._shared_axes[name].remove(ax)
        setattr(ax, f"_share{name}", None)


def _correlations_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Normalize a covariance matrix into a correlation matrix.

    Parameters
    ----------
    covariance : np.ndarray
        NxN covariance matrix.

    Returns
    -------
    np.ndarray
        NxN correlation matrix with unit diagonal.  Entries whose standard
        deviation is zero (an unestimated or degenerate parameter) are set to
        zero rather than ``nan``, so the heatmap stays drawable.
    """
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(sigma, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.where(denominator > 0.0, covariance / denominator, 0.0)
    return np.clip(correlations, -1.0, 1.0)


def _plot_standard_deviation(ax, std: float, max_std: float) -> None:
    """Draw one diagonal cell: a bar for the parameter's formal 1σ deviation.

    The y axis is shared across the diagonal (``0`` to ``max_std``) so the bars
    are directly comparable: a parameter whose bar is twice as tall is twice as
    uncertain, which a per-cell autoscaled axis would hide.  The parameter's
    name is not repeated here — the outer row/column labels already name it.
    """
    ax.bar([0.0], [std], width=0.6, color="steelblue", edgecolor="navy")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, max_std * 1.25 if max_std > 0.0 else 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(
        f"{std:.3e}",
        xy=(0.0, std),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )


def _plot_covariance_ellipse(
    x,
    y,
    *,
    covariance: np.ndarray,
    index_of: dict[str, int],
    n_sigma: float,
    ax=None,
    color=None,
    label=None,
    **kwargs,
) -> None:
    """Draw one lower-triangle cell: the ``n_sigma`` ellipse of a parameter pair.

    Seaborn calls this with the two column series and the target axes made
    current, so the cell is recovered from the series names.
    """
    ax = plt.gca() if ax is None else ax
    i = index_of[y.name]
    j = index_of[x.name]

    sub_covariance = np.array(
        [
            [covariance[i, i], covariance[i, j]],
            [covariance[j, i], covariance[j, j]],
        ]
    )

    eigenvalues, eigenvectors = np.linalg.eigh(sub_covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    major = n_sigma * np.sqrt(eigenvalues[1])
    minor = n_sigma * np.sqrt(eigenvalues[0])
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    ax.add_patch(
        Ellipse(
            xy=(0.0, 0.0),
            width=2.0 * major,
            height=2.0 * minor,
            angle=angle,
            facecolor="steelblue",
            edgecolor="navy",
            alpha=0.4,
            linewidth=1.5,
        )
    )

    # Symmetric limits around the origin, so the ellipse is centred in the cell
    # and its size is comparable with the neighbouring cells.
    extent = max(major, minor) * 1.3
    if extent <= 0.0:
        extent = 1.0
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")

    ax.axhline(0.0, color="0.6", linewidth=0.6, zorder=0)
    ax.axvline(0.0, color="0.6", linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=7)


def _plot_correlation_cell(
    x,
    y,
    *,
    correlations: np.ndarray,
    index_of: dict[str, int],
    cmap,
    annotate: bool,
    decimal_places: int,
    ax=None,
    color=None,
    label=None,
    **kwargs,
) -> None:
    """Draw one upper-triangle cell: a colour-coded correlation coefficient.

    The cell is painted in axes coordinates, so it fills the panel regardless of
    the data limits seaborn may have set.
    """
    ax = plt.gca() if ax is None else ax
    i = index_of[y.name]
    j = index_of[x.name]
    value = float(correlations[i, j])

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor=cmap((value + 1.0) / 2.0),
            edgecolor="white",
            linewidth=0.5,
        )
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])

    if annotate:
        text_color = "white" if abs(value) > 0.6 else "black"
        ax.text(
            0.5,
            0.5,
            f"{value:.{decimal_places}f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=text_color,
            fontsize=9,
        )


class CovarianceCorrelationPairGrid(EstimationPlot):
    """Corner plot of the covariance: σ on the diagonal, ellipses below, correlations above.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        Supported keys: ``rotate_to_rsw``, ``body_name``, ``central_body_name``,
        ``parameter_names``, ``n_sigma``, ``annotation``, ``decimal_places``,
        ``cmap``, ``figure.width``, ``figure.height``.
    estimation_output : est_an.EstimationOutput
        The estimation output whose covariance matrix is plotted.
    bodies : env.SystemOfBodies
        The system of bodies, used to define the RSW frame when rotating.
    ctx : RuntimeContext
        The runtime context, providing the reference epoch for the RSW frame.
    """

    requires_environment = True

    def __init__(
        self,
        cfg,
        estimation_output: est_an.EstimationOutput,
        bodies: env.SystemOfBodies,
        ctx: RuntimeContext,
    ):
        super().__init__(cfg, estimation_output)
        self.bodies = bodies
        self.ctx = ctx

    def _make_figure(self):
        plot_cfg = self.cfg
        covariance = self.estimation_output.covariance
        n_params = covariance.shape[0]

        # Default parameter names: RSW frame (the default is to rotate).
        default_names = ["R", "S", "W", "v_R", "v_S", "v_W"]
        parameter_names = self._parameter_names(n_params, default_names)

        # Optionally rotate to the RSW frame, exactly as CovarianceEllipses does.
        rotate_to_rsw = _cfg_get(plot_cfg, "rotate_to_rsw", default=True)
        if rotate_to_rsw:
            body_name = _cfg_get(plot_cfg, "body_name", default="Triton")
            central_body_name = _cfg_get(plot_cfg, "central_body_name", default="Neptune")
            inertial_state = _get_inertial_state_at_epoch(
                self.bodies, body_name, central_body_name, self.ctx.start_epoch
            )
            plot_covariance = _rotate_covariance_to_rsw(covariance, inertial_state)
        else:
            plot_covariance = covariance

        # Correlations are recomputed from the plotted matrix, so the upper
        # triangle describes the same matrix as the rotated ellipses.
        correlations = _correlations_from_covariance(plot_covariance)
        standard_deviations = np.sqrt(np.clip(np.diag(plot_covariance), 0.0, None))
        max_std = float(np.max(standard_deviations)) if n_params else 0.0

        # PairGrid needs a DataFrame to lay out the grid.  The values are never
        # drawn; the column names are only used to recover the (row, column)
        # index of each cell.  Internal keys keep the grid working even when two
        # parameters share a display name.
        keys = [f"p{i}" for i in range(n_params)]
        index_of = {key: i for i, key in enumerate(keys)}
        data = pd.DataFrame(correlations, columns=keys, index=keys)

        # ``figure.width``/``figure.height`` describe the whole figure, so the
        # per-cell size PairGrid wants is that size divided over the grid.
        figure_width, figure_height = self._figure_size(cfg=plot_cfg)
        cell_height = figure_height / max(n_params, 1)
        cell_width = figure_width / max(n_params, 1)

        grid = sns.PairGrid(
            data,
            height=cell_height,
            aspect=cell_width / cell_height,
            corner=False,
            despine=False,
        )

        # Every cell has its own units (a standard deviation, an ellipse extent,
        # a correlation), so the axes must not share limits: a shared axis would
        # let one cell's limits hide another cell's content.
        for ax in grid.axes.flat:
            _unshare_axes(ax)

        n_sigma = _cfg_get(plot_cfg, "n_sigma", default=1.0)
        cmap = plt.get_cmap(_cfg_get(plot_cfg, "cmap", default="coolwarm"))
        annotate = _cfg_get(plot_cfg, "annotation", default=True)
        decimal_places = _cfg_get(plot_cfg, "decimal_places", default=2)

        grid.map_lower(
            functools.partial(
                _plot_covariance_ellipse,
                covariance=plot_covariance,
                index_of=index_of,
                n_sigma=n_sigma,
            )
        )
        grid.map_upper(
            functools.partial(
                _plot_correlation_cell,
                correlations=correlations,
                index_of=index_of,
                cmap=cmap,
                annotate=annotate,
                decimal_places=decimal_places,
            )
        )

        # The diagonal is drawn directly rather than through ``map_diag``:
        # seaborn maps the diagonal onto a hidden twin axis, which would leave
        # the visible cell empty and its ticks meaningless.
        for i in range(n_params):
            _plot_standard_deviation(grid.axes[i, i], standard_deviations[i], max_std)

        # Label only the outer edges, as a corner plot does, using the display
        # names rather than the internal grid keys.
        for i in range(n_params):
            for j in range(n_params):
                ax = grid.axes[i, j]
                ax.set_xlabel(parameter_names[j] if i == n_params - 1 else "")
                ax.set_ylabel(parameter_names[i] if j == 0 else "")

        # The layout engine must be chosen before the colourbar is created:
        # matplotlib refuses to swap engines once a colourbar exists, because
        # the colourbar's gridspec belongs to the engine that made it.
        grid.figure.set_layout_engine("constrained")

        # One colourbar for the whole upper triangle, so the correlation colours
        # are readable without a legend in every cell.
        if n_params > 1:
            mappable = ScalarMappable(norm=Normalize(vmin=-1.0, vmax=1.0), cmap=cmap)
            mappable.set_array(np.array([]))
            grid.figure.colorbar(mappable, ax=grid.axes, label="Correlation", shrink=0.6)

        # The frame is part of the default title because the same covariance
        # rotates into two different matrices; a config that draws several of
        # these figures (one per model) overrides ``suptitle`` to say which is
        # which, and keeps the default otherwise.
        default_suptitle = (
            f"Covariance and correlation ({n_sigma}σ)"
            + (" [RSW frame]" if rotate_to_rsw else " [Inertial frame]")
        )
        grid.figure.suptitle(
            _cfg_get(plot_cfg, "suptitle", default=default_suptitle),
            fontsize=14,
        )
        return grid.figure, grid.axes


__all__ = ["CovarianceCorrelationPairGrid"]
