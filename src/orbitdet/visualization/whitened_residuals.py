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
from omegaconf import DictConfig

from orbitdet.analysis import ObservationAccess, Whitening, analysis_frame, rms
from orbitdet.visualization.base import Plot, _make_hover_formatter, _snake_case
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


def _cfg_get(cfg, *keys, default=None):
    """Read a nested config value, returning ``default`` when any key is missing."""
    node = cfg
    for key in keys:
        if node is None:
            return default
        try:
            node = node[key]
        except (KeyError, TypeError, IndexError):
            return default
    return default if node is None else node


class _WhitenedPlotBase(Plot):
    """Shared plumbing for the whitened-residual plots.

    Resolves the figure config, the grouping columns, the scan-frame rows and the
    per-set whitening, so the concrete plots only differ in what they draw.
    """

    def __init__(
        self,
        cfg: DictConfig,
        access: ObservationAccess,
        use_figure_name: str | None = None,
        group_by: str | list[str] | None = None,
        method: str = "cholesky",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        include_rejected: bool = False,
    ):
        super().__init__(cfg)
        self.access = access
        self.use_figure_name = use_figure_name
        self.group_by = group_by
        self.method = method
        self.include_rejected = include_rejected
        if (fig is not None and ax is None) or (fig is None and ax is not None):
            raise ValueError("Both fig and ax must be provided together.")
        self.fig = fig
        self.ax = ax
        self.name = use_figure_name or _snake_case(self.__class__.__name__)

    # -- Configuration --------------------------------------------------

    def _plot_cfg(self):
        """Return the figure config block for this plot."""
        return _cfg_get(self.cfg.figures, self.use_figure_name, default=None)

    def _grouping(self, plot_cfg) -> list[str]:
        """Resolve the grouping columns: argument, then config, then default."""
        group_by = self.group_by
        if group_by is None:
            group_by = _cfg_get(plot_cfg, "group_by", default=None)
        if group_by is None:
            group_by = "observatory_code"
        return [group_by] if isinstance(group_by, str) else list(group_by)

    @staticmethod
    def _group_label(key, columns: list[str]) -> str:
        """Render a groupby key as a legend label."""
        values = key if isinstance(key, tuple) else (key,)
        if len(columns) == 1:
            return str(values[0])
        return ", ".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))

    # -- Data -----------------------------------------------------------

    def _scan_frame(self) -> pd.DataFrame:
        """Return the scan-frame rows, filtered to the ones this plot should show.

        Raises
        ------
        ValueError
            If the frame carries no scan angle, so there is nothing to whiten.
        """
        frame = analysis_frame(self.access, unit=None, scan=True, normalized=False)
        al_col = "residual_along_scan_rad"
        ac_col = "residual_across_scan_rad"
        has_angle = "scan_angle" in frame.columns and frame["scan_angle"].notna().any()
        if not has_angle or al_col not in frame.columns or ac_col not in frame.columns:
            raise ValueError(
                f"{self.__class__.__name__} requires scan angles, but the frame carries no "
                "scan_angle (the source did not record a position_angle_scan)."
            )
        frame = frame.loc[frame["scan_angle"].notna()]
        if not self.include_rejected and "rejected" in frame.columns:
            frame = frame.loc[~frame["rejected"].astype(bool)]
        return frame

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
        """Yield ``(label, epochs, z)`` per group, with ``z`` an ``(n, 2)`` array."""
        al_col = "residual_along_scan_rad"
        ac_col = "residual_across_scan_rad"
        for key, group in frame.groupby(group_cols, dropna=False, sort=True):
            label = self._group_label(key, group_cols)
            set_column = (
                group["observation_set_index"].astype(int)
                if "observation_set_index" in group.columns
                else None
            )
            set_indices = (
                group["observation_set_index"].dropna().astype(int).unique().tolist()
                if set_column is not None
                else []
            )
            epoch_chunks: list[np.ndarray] = []
            z_chunks: list[np.ndarray] = []
            for set_index in set_indices:
                rows = group if set_column is None else group.loc[set_column == set_index]
                whitened = self._whitened_set(rows, al_col, ac_col)
                if whitened is None:
                    continue
                epochs, z = whitened
                epoch_chunks.append(epochs)
                z_chunks.append(z)
            if not epoch_chunks:
                continue
            yield label, np.concatenate(epoch_chunks), np.vstack(z_chunks)

    def _target_name(self, frame: pd.DataFrame) -> str:
        """Return the observed object's name for the suptitle, or ``"target"``."""
        if "observed_object" in frame.columns and not frame["observed_object"].dropna().empty:
            return str(frame["observed_object"].dropna().iloc[0])
        return "target"

    def _suptitle(self, plot_cfg, frame: pd.DataFrame, default: str) -> str:
        """Resolve and format the suptitle."""
        suptitle = _cfg_get(plot_cfg, "titles", "suptitle", default=default)
        try:
            if isinstance(suptitle, str):
                suptitle = suptitle.format(target_name=self._target_name(frame))
        except Exception:
            pass
        return suptitle

    def _legend(self, ax: plt.Axes, plot_cfg) -> None:
        """Apply the configured legend layout to an axes."""
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

    def _styling(self, plot_cfg):
        """Return ``(colors, marker_size, marker_types)`` from the figure config."""
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)
        marker_types = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
        return colors, marker_size, marker_types


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
    use_figure_name : str | None
        Name of the figure configuration block under ``cfg.figures``.
    group_by : str | list[str] | None
        Standard-frame column(s) to group the scatter series by.
    method : str
        ``"cholesky"`` (default) or ``"symmetric"`` whitening.
    fig, ax : optional
        A pre-existing figure and axes to draw into.
    include_rejected : bool
        Whether to plot rows that outlier rejection removed.
    """

    def _make_figure(self):
        plot_cfg = self._plot_cfg()
        logger.debug("Plotting configuration for '%s': %s", self.use_figure_name, plot_cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27)

        if self.fig is None and self.ax is None:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        else:
            fig, ax = self.fig, self.ax

        frame = self._scan_frame()
        group_cols = self._grouping(plot_cfg)
        missing = [column for column in group_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

        colors, marker_size, marker_types = self._styling(plot_cfg)

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

        x_label = _cfg_get(
            plot_cfg, "axes", "x_label", default="Whitened component 1 [$\\sigma$]"
        )
        y_label = _cfg_get(
            plot_cfg, "axes", "y_label", default="Whitened component 2 [$\\sigma$]"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(self._suptitle(plot_cfg, frame, "Whitened Residuals for {target_name}"))

        hover_x_label = _cfg_get(plot_cfg, "axes", "hover_x_label", default="z1")
        hover_y_label = _cfg_get(plot_cfg, "axes", "hover_y_label", default="z2")
        ax.format_coord = _make_hover_formatter(
            hover_x_label, hover_y_label, datetime_x=False
        )

        self._legend(ax, plot_cfg)
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

    def __init__(self, *args, method: str = "symmetric", **kwargs):
        # This plot is defined by its whitening method; ignore any other choice.
        super().__init__(*args, method=method, **kwargs)

    def _make_figure(self):
        plot_cfg = self._plot_cfg()
        logger.debug("Plotting configuration for '%s': %s", self.use_figure_name, plot_cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

        if self.fig is None and self.ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True)
        else:
            if len(self.ax) != 2:
                raise ValueError("Expected two subplots for the two whitened components.")
            fig, axs = self.fig, self.ax

        frame = self._scan_frame()
        group_cols = self._grouping(plot_cfg)
        missing = [column for column in group_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

        colors, marker_size, marker_types = self._styling(plot_cfg)

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

        self._legend(axs[0], plot_cfg)
        self._legend(axs[1], plot_cfg)
        fig.suptitle(
            self._suptitle(plot_cfg, frame, "Symmetric Whitened Residuals for {target_name}")
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
        plot_cfg = self._plot_cfg()
        logger.debug("Plotting configuration for '%s': %s", self.use_figure_name, plot_cfg)
        fig_w = _cfg_get(plot_cfg, "figure", "width", default=8.27 * 2)
        fig_h = _cfg_get(plot_cfg, "figure", "height", default=8.27 * 2 / 2)

        if self.fig is None and self.ax is None:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        else:
            fig, ax = self.fig, self.ax

        frame = self._scan_frame()
        group_cols = self._grouping(plot_cfg)
        missing = [column for column in group_cols if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Grouping column(s) {missing} are not in the standard frame. "
                f"Available columns: {sorted(frame.columns)}."
            )

        colors, marker_size, marker_types = self._styling(plot_cfg)

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

        ax.set_title(
            self._suptitle(plot_cfg, frame, "Mahalanobis Residuals for {target_name}")
        )
        self._legend(ax, plot_cfg)
        fig.set_tight_layout(True)
        return fig, ax
