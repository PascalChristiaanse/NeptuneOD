"""Whitened-residual diagnostics for correlated (scanning) observations.

Dividing each residual component by its own marginal sigma is only correct when
the weight matrix is diagonal.  For a scanning instrument such as Gaia the
applied weight matrix carries an along-scan / across-scan correlation inside
every transit, so the marginal normalization leaves that correlation in and
inflates the variance.  The correct reduction is **whitening**::

    W = L L^T,   z = L^T r,   Cov(z) = I.

Whitening is only unique up to an orthogonal rotation, and the three plots in
this module make the choice explicit:

``WhitenedResidualScatter``
    The 2-D scatter of the two whitened components of each observation.  A
    well-calibrated solution forms a unit-variance circular cloud centred on the
    origin; the axes are the whitened components, not along/across-scan, because
    whitening mixes them.

``SymmetricWhitenedResiduals``
    The two whitened components against epoch, using the **symmetric** square
    root ``W^{1/2}`` (ZCA).  Unlike the Cholesky factor it privileges no
    component, so the correlation is spread evenly over the pair instead of being
    dumped onto the first component.

``MahalanobisResiduals``
    One point per observation: the Mahalanobis distance ``||z_i||`` of that
    observation's residual block.  This is the only reduction that is invariant
    to the choice of whitening factor, so it is the honest per-observation
    calibration diagnostic (``E[||z_i||^2] = block_size``).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orbitdet.analysis import Whitening, rms
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get
from orbitdet.visualization.residuals import (
    _configure_datetime_axis,
    _seconds_since_j2000_to_datetimes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WhitenedResidualScatter",
    "SymmetricWhitenedResiduals",
    "MahalanobisResiduals",
]


class _WhitenedPlotBase(ResidualPlot):
    """Shared plumbing for the whitened-residual plots.

    Resolves the clustering columns, the scan-frame rows and the per-set
    whitening, so the concrete plots only differ in what they draw.
    """

    #: Whitening factor used by this plot (``"cholesky"`` or ``"symmetric"``).
    method: str = "cholesky"

    # -- Data -----------------------------------------------------------

    def _scan_frame(self) -> pd.DataFrame:
        """Return the scan-frame rows, filtered to the ones this plot should show.

        Raises
        ------
        ValueError
            If the frame carries no scan angle, so there is nothing to whiten.
        """
        frame = self._frame(unit=None, scan=True, normalized=False)
        al_col = "residual_along_scan_rad"
        ac_col = "residual_across_scan_rad"
        has_angle = "scan_angle" in frame.columns and frame["scan_angle"].notna().any()
        if not has_angle or al_col not in frame.columns or ac_col not in frame.columns:
            raise ValueError(
                f"{self.__class__.__name__} requires scan angles, but the frame carries no "
                "scan_angle (the source did not record a position_angle_scan)."
            )
        return frame.loc[frame["scan_angle"].notna()]

    def _whitened_set(
        self, rows: pd.DataFrame, al_col: str, ac_col: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(epochs, z)`` for one set's rows, whitened, or ``None``.

        ``z`` is the ``(n_obs, 2)`` array of whitened components: column 0 is the
        whitened first component, column 1 the whitened second.  When the set's
        weight matrix is rank-deficient the per-observation 2x2 blocks are
        whitened instead, so the pair structure survives.
        """
        if rows.empty:
            return None
        first = rows[al_col].to_numpy("float64")
        second = rows[ac_col].to_numpy("float64")
        keep = np.isfinite(first) & np.isfinite(second)
        if not keep.any():
            return None

        set_index = int(rows["observation_set_index"].iloc[0])
        weight_matrix = self.access.weight_matrix(set_index, basis="scan")
        n_obs = len(rows)
        if weight_matrix.shape[0] == 2 * n_obs:
            index = np.repeat(keep, 2)
            weight_matrix = weight_matrix[np.ix_(index, index)]

        residual = np.stack([first[keep], second[keep]], axis=1).ravel()
        if weight_matrix.shape[0] != residual.size:
            return None

        epochs = pd.to_numeric(rows["epoch_tdb"], errors="coerce").to_numpy("float64")[keep]
        whitening = Whitening.from_weight_matrix(weight_matrix, method=self.method)
        z = whitening.apply(residual)
        if z.size == residual.size:
            return epochs, z.reshape(-1, 2)

        # Rank-deficient: whiten each observation's own 2x2 block.
        out = np.full((residual.size // 2, 2), np.nan)
        for i in range(residual.size // 2):
            block = weight_matrix[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
            block_whitening = Whitening.from_weight_matrix(block, method=self.method)
            z_pair = block_whitening.apply(residual[2 * i : 2 * i + 2])
            out[i, : z_pair.size] = z_pair
        return epochs, out

    def _grouped_whitened(self, frame: pd.DataFrame, group_cols: list[str]):
        """Yield ``(label, epochs, z)`` per cluster, with ``z`` an ``(n, 2)`` array."""
        al_col = "residual_along_scan_rad"
        ac_col = "residual_across_scan_rad"
        for key, cluster in frame.groupby(group_cols, dropna=False, sort=True):
            label = self._group_label(key, group_cols)
            set_column = (
                cluster["observation_set_index"].astype(int)
                if "observation_set_index" in cluster.columns
                else None
            )
            set_indices = (
                cluster["observation_set_index"].dropna().astype(int).unique().tolist()
                if set_column is not None
                else []
            )
            epoch_chunks: list[np.ndarray] = []
            z_chunks: list[np.ndarray] = []
            for set_index in set_indices:
                rows = cluster if set_column is None else cluster.loc[set_column == set_index]
                whitened = self._whitened_set(rows, al_col, ac_col)
                if whitened is None:
                    continue
                epochs, z = whitened
                epoch_chunks.append(epochs)
                z_chunks.append(z)
            if not epoch_chunks:
                continue
            yield label, np.concatenate(epoch_chunks), np.vstack(z_chunks)


class WhitenedResidualScatter(_WhitenedPlotBase):
    """2-D scatter of the two whitened residual components per observation.

    Each observation contributes one point ``(z_1, z_2)``.  A well-calibrated
    solution forms a unit-variance circular cloud centred on the origin, so the
    reference circles at 1, 2 and 3 sigma make miscalibration immediately
    visible.  The axes are the *whitened* components, not along/across-scan:
    whitening mixes the two, so labelling them AL/AC would be wrong.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra experiment configuration.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``group_by`` and ``include_rejected`` are read from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        frame = self._scan_frame()
        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (label, _epochs, z) in enumerate(self._grouped_whitened(frame, group_cols)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            radius = rms(z.ravel())
            ax.scatter(
                z[:, 0],
                z[:, 1],
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=f"{label} - RMS(z): {radius:.3f}" if np.isfinite(radius) else label,
            )

        # Reference circles at 1, 2 and 3 sigma.
        theta = np.linspace(0.0, 2.0 * np.pi, 256)
        for radius in (1.0, 2.0, 3.0):
            ax.plot(
                radius * np.cos(theta),
                radius * np.sin(theta),
                color="grey",
                linestyle="--",
                linewidth=0.8,
                zorder=0,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="black", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="datalim")

        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Whitened component 1 [$\\sigma$]")
        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Whitened component 2 [$\\sigma$]")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(self._suptitle(frame, "suptitle", "Whitened Residuals for {target_name}"))

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="z1")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="z2")
        ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label, datetime_x=False)

        self._legend(ax)
        fig.set_tight_layout(True)
        return fig, ax


class SymmetricWhitenedResiduals(_WhitenedPlotBase):
    """Whitened residual components against epoch, using symmetric whitening.

    Identical in layout to :class:`~orbitdet.visualization.residuals.NormalizedResidualScan`
    but whitened through the symmetric square root ``W^{1/2}`` (ZCA) instead of
    the Cholesky factor.  The Cholesky factor is lower-triangular, so it dumps the
    whole intra-block correlation onto the first component; the symmetric root
    spreads it evenly, which makes the two panels comparable.

    The panels are still *mixtures* of along- and across-scan, so they are
    labelled generically.  Use :class:`MahalanobisResiduals` for a reduction that
    does not depend on the whitening factor at all.
    """

    def __init__(self, *args, **kwargs):
        # This plot is defined by its whitening method; ignore any other choice.
        super().__init__(*args, **kwargs)

    method = "symmetric"

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)

        frame = self._scan_frame()
        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (label, epochs, z) in enumerate(self._grouped_whitened(frame, group_cols)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            datetimes = _seconds_since_j2000_to_datetimes(epochs)
            for component, ax in enumerate(axs):
                values = z[:, component]
                radius = rms(values)
                ax.scatter(
                    datetimes,
                    values,
                    marker=marker,
                    s=marker_size,
                    color=color,
                    alpha=0.5,
                    label=f"{label} - RMS(z): {radius:.3f}" if np.isfinite(radius) else label,
                )

        for ax in axs:
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax.axhline(-1.0, color="grey", linestyle="--", linewidth=0.8)

        title_first = _cfg_get(plot_cfg, "titles", "first", default="Whitened component 1")
        title_second = _cfg_get(plot_cfg, "titles", "second", default="Whitened component 2")
        axs[0].set_title(title_first)
        axs[1].set_title(title_second)

        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Whitened residual [$\\sigma$]")
        axs[0].set_ylabel(y_label)
        axs[1].set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        axs[0].set_xlabel(x_label)
        axs[1].set_xlabel(x_label)
        _configure_datetime_axis(axs[0])
        _configure_datetime_axis(axs[1])

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(
            plot_cfg, "axes", "hover_y_label", default="Whitened residual [$\\sigma$]"
        )
        fmt = _make_hover_formatter(hover_x_label, hover_y_label)
        axs[0].format_coord = fmt
        axs[1].format_coord = fmt

        self._legend(axs[0])
        self._legend(axs[1])
        fig.suptitle(
            self._suptitle(frame, "suptitle", "Symmetric Whitened Residuals for {target_name}")
        )
        fig.set_tight_layout(True)
        return fig, axs


class MahalanobisResiduals(_WhitenedPlotBase):
    """Per-observation Mahalanobis distance against epoch.

    Each observation contributes a single point: the Mahalanobis distance of its
    residual block, ``||z_i|| = sqrt(r_i^T W_i r_i)``.  This is the only
    reduction that is invariant to the choice of whitening factor (Cholesky and
    symmetric whitening give the same value), so it is the honest
    per-observation calibration diagnostic.

    For a 2-component block the squared distance is chi-squared with 2 degrees of
    freedom, so ``E[||z_i||^2] = 2`` and the reference line sits at
    ``sqrt(2) ~ 1.414``.  A well-calibrated solution scatters around it.
    """

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size()
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        frame = self._scan_frame()
        group_cols = self._grouping()
        self._require_columns(frame, group_cols)

        colors, marker_size, marker_types = self._styling()

        for index, (label, epochs, z) in enumerate(self._grouped_whitened(frame, group_cols)):
            color = colors(index % colors.N)
            marker = marker_types[index % len(marker_types)]
            distance = np.sqrt(np.nansum(z**2, axis=1))
            mean_distance = float(np.nanmean(distance)) if distance.size else np.nan
            ax.scatter(
                _seconds_since_j2000_to_datetimes(epochs),
                distance,
                marker=marker,
                s=marker_size,
                color=color,
                alpha=0.5,
                label=(
                    f"{label} - mean d: {mean_distance:.3f}"
                    if np.isfinite(mean_distance)
                    else label
                ),
            )

        # E[d^2] = block_size = 2, so the reference line is sqrt(2).
        expected = np.sqrt(2.0)
        ax.axhline(expected, color="black", linewidth=0.8, label=f"E[d] = sqrt(2) = {expected:.3f}")
        ax.axhline(2.0 * expected, color="grey", linestyle="--", linewidth=0.8)

        y_label = _cfg_get(plot_cfg, "axes", "y_label", default="Mahalanobis distance [$\\sigma$]")
        ax.set_ylabel(y_label)
        x_label = _cfg_get(plot_cfg, "axes", "x_label", default="Epoch")
        ax.set_xlabel(x_label)
        _configure_datetime_axis(ax)

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="Mahalanobis distance")
        ax.format_coord = _make_hover_formatter(hover_x_label, hover_y_label)

        ax.set_title(self._suptitle(frame, "suptitle", "Mahalanobis Residuals for {target_name}"))
        self._legend(ax)
        fig.set_tight_layout(True)
        return fig, ax
