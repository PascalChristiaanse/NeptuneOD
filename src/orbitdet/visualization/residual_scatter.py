import functools

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from omegaconf import DictConfig
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import links
from tudatpy.estimation.observations import observations_processing as obs_proc

from orbitdet.observations import get_observatory_info
from orbitdet.visualization.base import Plot, _plain_hover_formatter


def _cfg_get(cfg: DictConfig | dict | None, *keys, default=None):
    cur = cfg
    for k in keys:
        if cur is None:
            return default
        try:
            # DictConfig supports get
            cur = cur.get(k)
        except Exception:
            try:
                cur = cur[k]
            except Exception:
                return default
    return default if cur is None else cur


def _rad_to_arcsec(angle_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(angle_rad) * 3600.0


def _principal_angle_rad(angle_rad: np.ndarray) -> np.ndarray:
    return np.remainder(angle_rad + np.pi, 2.0 * np.pi) - np.pi


def _seconds_since_j2000_to_datetimes(seconds_since_j2000: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(
        seconds_since_j2000,
        unit="s",
        origin=pd.Timestamp("2000-01-01T12:00:00"),
    )


def _make_hover_formatter(hover_x_label: str, hover_y_label: str):
    return functools.partial(
        _plain_hover_formatter, hover_x_label=hover_x_label, hover_y_label=hover_y_label
    )


def _format_colorbar_tick(value, _position: int | None = None) -> str:
    """Format colorbar ticks as ``YYYY-MM`` dates.

    Module-level (not a nested closure) so the figure remains picklable.
    """
    try:
        dt = mdates.num2date(mdates.epoch2num(float(value)))
        return dt.strftime("%Y-%m")
    except Exception:
        return f"{value:.0f}"


def _transit_slices(weight_matrix: np.ndarray, block_size: int = 2) -> list[tuple[int, int]]:
    """Split a block-diagonal weight matrix into per-transit observation ranges.

    Gaia's full weight matrix is block-diagonal in transits: the ``2x2`` block
    coupling observation ``i`` to observation ``i + 1`` is exactly zero across a
    transit boundary and (generically) non-zero within a transit.  Boundaries are
    therefore detected from the zero off-diagonal structure rather than from a
    time gap, which keeps the grouping exactly consistent with the covariance
    blocks the archive provides.

    Parameters
    ----------
    weight_matrix : np.ndarray
        The ``2n x 2n`` full weight matrix of a single observation set.
    block_size : int, optional
        Number of residual components per observation (2 for RA/Dec).

    Returns
    -------
    list[tuple[int, int]]
        ``(start, stop)`` observation-index pairs (``stop`` exclusive) for each
        transit, in observation order.
    """
    n_obs = weight_matrix.shape[0] // block_size
    boundaries = [0]
    for i in range(n_obs - 1):
        r0 = i * block_size
        c0 = (i + 1) * block_size
        coupling = weight_matrix[r0 : r0 + block_size, c0 : c0 + block_size]
        if not np.any(coupling):
            boundaries.append(i + 1)
    boundaries.append(n_obs)
    return [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]


def _transit_mean_covariance(covariance: np.ndarray) -> np.ndarray:
    """Reduce a ``2N x 2N`` transit covariance to the ``2x2`` covariance of its mean.

    Computes ``C_mean = A C A^T`` with ``A = (1/N) 1_N^T (x) I_2``, i.e. the
    covariance of the arithmetic mean of the ``N`` ``(RA, Dec)`` samples in the
    transit.  This is the uncertainty of the transit's mean position offset, which
    is the quantity that survives averaging over the CCD crossings.

    Parameters
    ----------
    covariance : np.ndarray
        ``2N x 2N`` covariance matrix for a single transit, in ``(RA, Dec)``
        coordinates (radians).

    Returns
    -------
    np.ndarray
        ``2x2`` covariance of the transit mean.
    """
    n_obs = covariance.shape[0] // 2
    if n_obs == 0:
        return np.zeros((2, 2))
    averaging = np.kron(np.ones((1, n_obs)) / n_obs, np.eye(2))
    return averaging @ covariance @ averaging.T


def _invert_weight_block(weight_block: np.ndarray) -> np.ndarray:
    """Invert a transit weight block, falling back to a pseudo-inverse.

    The archived transit covariance is often near rank-1 (condition number
    ~1e6-1e7), so a plain inverse can be numerically fragile.  ``pinv`` is used
    when the inverse raises or is not finite.
    """
    try:
        covariance = np.linalg.inv(weight_block)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(weight_block)
    if not np.all(np.isfinite(covariance)):
        return np.linalg.pinv(weight_block)
    return covariance


def _ellipse_parameters(covariance: np.ndarray, n_sigma: float) -> tuple[float, float, float]:
    """Return ``(semi_major, semi_minor, angle_deg)`` of the ``n_sigma`` ellipse.

    The ellipse is the ``n_sigma`` contour of the ``2x2`` covariance: its
    semi-axes are ``n_sigma * sqrt(eigenvalue)`` and its orientation follows the
    major eigenvector.
    """
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    semi_major = n_sigma * np.sqrt(values[0])
    semi_minor = n_sigma * np.sqrt(values[1])
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    # An ellipse axis is a line, so fold the orientation into (-90, 90] to make
    # it independent of the (arbitrary) eigenvector sign.
    angle = (angle + 90.0) % 180.0 - 90.0
    return semi_major, semi_minor, angle


class ResidualScatter(Plot):
    """Plot RA vs DEC residuals as a scatter plot with time encoded as color."""

    #: Top-level config key this plot reads its settings from.  Subclasses may
    #: override it to read a different (e.g. extended) config block.
    _config_key = "residual_scatter"

    def __init__(
        self,
        cfg: DictConfig,
        observation_collection: obs.ObservationCollection,
        observation_parsers: list[obs_proc.ObservationParserType] | None = None,
    ):
        super().__init__(cfg)
        self.observation_collection = observation_collection
        self.observation_parsers = observation_parsers

    def _get_observation_sets(self) -> list[obs.SingleObservationSet]:
        """Return the observation sets to plot, honouring optional parsers."""
        if self.observation_parsers is None:
            return self.observation_collection.get_single_observation_sets()
        return self.observation_collection.get_single_observation_sets(self.observation_parsers)

    def _plot_observations(self, ax, plot_cfg) -> tuple[float, float, str]:
        """Draw the RA/Dec scatter layer onto ``ax``.

        Each observatory group uses a distinct marker shape. Points are colored
        on a rainbow gradient corresponding to observation time so that temporal
        trends are visible even though time is not an explicit axis.

        Returns
        -------
        tuple[float, float, str]
            ``(global_t_min, global_t_max, target_name)`` used by the caller for
            the colorbar and titles.
        """
        observation_sets = self._get_observation_sets()
        self._observation_sets = observation_sets

        marker_types = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "<",
            ">",
            "P",
            "X",
        ]  # cycle through marker types for different observatories

        # First pass: collect all timestamps to determine global time range for color mapping
        all_times_sec = []
        for obs_set in observation_sets:
            obs_times_sec_j2000 = np.array(
                [epoch.to_float() for epoch in obs_set.observation_times]
            )
            all_times_sec.append(obs_times_sec_j2000)

        all_times_sec = np.concatenate(all_times_sec)
        global_t_min = float(np.min(all_times_sec))
        global_t_max = float(np.max(all_times_sec))
        time_range = global_t_max - global_t_min if global_t_max > global_t_min else 1.0

        cmap_name = _cfg_get(plot_cfg, "styling", "cmap", default="rainbow")
        cmap = plt.get_cmap(cmap_name)
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)

        for set_index, obs_set in enumerate(observation_sets):
            if len(obs_set.observation_times) == 0:
                continue

            link_ends = obs_set.link_definition.link_ends

            # Resolve a display label for this observation set.
            if links.receiver in link_ends:
                ref_point = link_ends[links.receiver].reference_point
                if ref_point == "":
                    body = link_ends[links.receiver].body_name
                    info = {"code": "", "name": body, "region": "Spacecraft"}
                else:
                    info = get_observatory_info(self.cfg, ref_point)
            elif links.observer in link_ends:
                body = link_ends[links.observer].body_name
                info = {"code": "", "name": body, "region": body}
            else:
                first_key = next(iter(link_ends.keys()))
                body = link_ends[first_key].body_name
                info = {"code": "", "name": body, "region": body}

            # Determine target (observed body)
            if links.transmitter in link_ends:
                target_name = link_ends[links.transmitter].body_name
            elif links.observed_body in link_ends:
                target_name = link_ends[links.observed_body].body_name
            else:
                target_name = list(link_ends.values())[-1].body_name
            marker = marker_types[set_index % len(marker_types)]

            obs_times_sec_j2000 = np.array(
                [epoch.to_float() for epoch in obs_set.observation_times]
            )
            residuals = np.array(obs_set.residuals)
            # n x 2 array of RA and DEC residuals in radians

            # RA residuals are circular; fold them to the principal interval before converting.
            ra_residuals_arcsec = _rad_to_arcsec(_principal_angle_rad(residuals[:, 0]))
            dec_residuals_arcsec = _rad_to_arcsec(residuals[:, 1])

            # Normalize time to [0, 1] for color mapping
            time_normalized = (obs_times_sec_j2000 - global_t_min) / time_range
            colors = cmap(time_normalized)

            ax.scatter(
                ra_residuals_arcsec,
                dec_residuals_arcsec,
                marker=marker,
                s=marker_size,
                c=colors,
                label=f"{info['name']} - {info['region']}",
                alpha=0.5,
                edgecolors="none",
            )

        self._target_name = target_name
        return global_t_min, global_t_max, target_name

    def _resolve_plot_cfg(self, cfg):
        """Locate this plot's configuration block.

        Figure presets are composed under ``cfg.figures.<name>`` (e.g.
        ``/figures/residual_scatter_ellipses@figures.residual_scatter_ellipses``),
        but some callers pass the block at the top level.  Prefer the
        ``cfg.figures.<name>`` location and fall back to the top level so both
        conventions work.
        """
        plot_cfg = _cfg_get(cfg, "figures", self._config_key, default=None)
        if plot_cfg is None:
            plot_cfg = _cfg_get(cfg, self._config_key, default=None)
        return plot_cfg

    def _make_figure(self):
        cfg = self.cfg

        # Load plotting configuration
        plot_cfg = self._resolve_plot_cfg(cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27)

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

        global_t_min, global_t_max, target_name = self._plot_observations(ax, plot_cfg)

        cmap_name = _cfg_get(plot_cfg, "styling", "cmap", default="rainbow")
        cmap = plt.get_cmap(cmap_name)

        # Titles and labels (configurable)
        title = _cfg_get(
            plot_cfg, "titles", "title", default=f"RA vs DEC Residuals for {target_name}"
        )
        # Allow templates like "RA vs DEC Residuals for {target_name}" in config
        try:
            if isinstance(title, str):
                title = title.format(target_name=target_name)
        except Exception:
            pass

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="RA Residual [arcsec]")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="DEC Residual [arcsec]")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Add a colorbar for time
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=global_t_min, vmax=global_t_max),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(_cfg_get(plot_cfg, "axes", "colorbar_label", default="Epoch"))

        # Format colorbar ticks as dates
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(_format_colorbar_tick))

        # Hover formatter
        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="RA Residual [arcsec]")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="DEC Residual [arcsec]")
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        ax.format_coord = fmt

        # Legend placement (configurable)
        try:
            legend_ncols = int(_cfg_get(plot_cfg, "legend", "ncols", default=2))
        except Exception:
            legend_ncols = 2

        bbox = _cfg_get(plot_cfg, "legend", "bbox_to_anchor", default={"x": 0.5, "y": -0.15})
        try:
            if isinstance(bbox, dict):
                bbox_tuple = (float(bbox.get("x", 0.5)), float(bbox.get("y", -0.15)))
            else:
                bbox_tuple = tuple(float(x) for x in bbox)
        except Exception:
            bbox_tuple = (0.5, -0.15)

        ax.legend(ncols=legend_ncols, loc="upper center", bbox_to_anchor=bbox_tuple)
        fig.set_tight_layout(True)

        return fig, ax


class ResidualScatterWithEllipses(ResidualScatter):
    """RA vs DEC residual scatter with per-transit uncertainty ellipses overlaid.

    Extends :class:`ResidualScatter` by drawing, for every Gaia transit, the
    ``n_sigma`` ellipse of the transit's **mean** position offset.  The ellipse
    covariance is obtained from the full transit weight matrix carried by the
    observation set (``W = blockdiag(inv(C_transit))``):

    .. math::

        C_{\\bar r} = A\\, C\\, A^\\top, \\qquad A = \\tfrac{1}{N}\\,\\mathbf 1_N^\\top \\otimes I_2

    where :math:`C` is the ``2N x 2N`` transit covariance and :math:`N` the number
    of CCD crossings in the transit.  Transits are detected from the block
    structure of the weight matrix (see :func:`_transit_slices`).

    Observation sets without a full weight-matrix contribution (i.e. non-Gaia
    sets, or Gaia sets whose covariance was discarded by the weighting engine) are
    skipped, since they carry no intra-transit covariance to draw.
    """

    _config_key = "residual_scatter_ellipses"

    def _plot_observations(self, ax, plot_cfg) -> tuple[float, float, str]:
        result = super()._plot_observations(ax, plot_cfg)
        self._overlay_ellipses(ax, plot_cfg)
        return result

    def _overlay_ellipses(self, ax, plot_cfg) -> None:
        """Draw the per-transit mean-covariance ellipses onto ``ax``."""
        import logging

        from matplotlib.colors import to_rgba

        logger = logging.getLogger(__name__)

        ellipses_cfg = _cfg_get(plot_cfg, "ellipses", default=None)
        n_sigma = float(_cfg_get(ellipses_cfg, "n_sigma", default=3.0))
        color = _cfg_get(ellipses_cfg, "color", default="black")
        alpha = float(_cfg_get(ellipses_cfg, "alpha", default=0.6))
        facecolor = _cfg_get(ellipses_cfg, "facecolor", default=color)
        face_alpha = float(_cfg_get(ellipses_cfg, "face_alpha", default=0.15))
        linewidth = float(_cfg_get(ellipses_cfg, "linewidth", default=1.0))
        min_obs_per_transit = int(_cfg_get(ellipses_cfg, "min_obs_per_transit", default=2))
        max_ellipses = int(_cfg_get(ellipses_cfg, "max_ellipses", default=500))

        # Light, semi-transparent fill so overlapping ellipses remain readable.
        edge_rgba = to_rgba(color, alpha)
        face_rgba = to_rgba(facecolor, face_alpha)

        n_drawn = 0
        n_skipped_sets = 0
        n_duplicates = 0
        # Signatures of already-drawn ellipses, used to drop exact duplicates that
        # can arise when the same observation set is present more than once in the
        # collection (which would otherwise double the number of ellipses).
        drawn_signatures: list[tuple[float, float, float, float, float]] = []

        for obs_set in getattr(self, "_observation_sets", []):
            if len(obs_set.observation_times) == 0:
                continue

            # Only sets carrying a full (off-diagonal) weight matrix have an
            # intra-transit covariance to visualise.
            if not getattr(obs_set, "has_full_weight_matrix_contribution", False):
                n_skipped_sets += 1
                continue

            weight_matrix = np.asarray(obs_set.weight_matrix, dtype=float)
            residuals = np.array(obs_set.residuals)
            if residuals.ndim == 1:
                residuals = residuals.reshape(-1, 2)

            for start, stop in _transit_slices(weight_matrix):
                if stop - start < min_obs_per_transit:
                    continue
                if n_drawn >= max_ellipses:
                    break

                block = weight_matrix[2 * start : 2 * stop, 2 * start : 2 * stop]
                covariance = _invert_weight_block(block)
                mean_covariance = _transit_mean_covariance(covariance)
                semi_major, semi_minor, angle = _ellipse_parameters(mean_covariance, n_sigma)

                # Ellipse center: the transit's mean residual (RA folded to the
                # principal interval, matching the scatter layer).
                mean_residual = residuals[start:stop].mean(axis=0)
                center_ra = _rad_to_arcsec(_principal_angle_rad(mean_residual[0]))
                center_dec = _rad_to_arcsec(mean_residual[1])

                # Semi-axes are in radians; convert to the scatter's arcsec units.
                width = 2.0 * _rad_to_arcsec(semi_major)
                height = 2.0 * _rad_to_arcsec(semi_minor)

                # Skip exact duplicates (same center, size and orientation).
                signature = (center_ra, center_dec, width, height, angle)
                if any(
                    abs(signature[0] - s[0]) < 1e-9
                    and abs(signature[1] - s[1]) < 1e-9
                    and abs(signature[2] - s[2]) < 1e-9
                    and abs(signature[3] - s[3]) < 1e-9
                    and abs(signature[4] - s[4]) < 1e-9
                    for s in drawn_signatures
                ):
                    n_duplicates += 1
                    continue
                drawn_signatures.append(signature)

                label = f"Transit {n_sigma:g}$\\sigma$" if n_drawn == 0 else None
                ellipse = Ellipse(
                    xy=(center_ra, center_dec),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=face_rgba,
                    edgecolor=edge_rgba,
                    linewidth=linewidth,
                    label=label,
                )
                ax.add_patch(ellipse)
                n_drawn += 1

        logger.info(
            "ResidualScatterWithEllipses: drew %d ellipse(s) at %g sigma "
            "(%d duplicate(s) skipped, %d set(s) without a full weight matrix).",
            n_drawn,
            n_sigma,
            n_duplicates,
            n_skipped_sets,
        )
