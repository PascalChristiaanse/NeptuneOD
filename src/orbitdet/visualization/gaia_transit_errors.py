"""Gaia random/systematic error components from the transit structure.

Gaia samples a source several times within one *transit* (the crossing of the
focal plane), and revisits it in later transits.  That structure splits the
residual scatter into two parts:

``random``
    The scatter *within* a transit, where the observing geometry is essentially
    fixed.  Estimating it as the pooled RMS of the residuals about each transit's
    own mean captures the per-crossing noise without contamination from slow
    (transit-to-transit) errors.

``systematic``
    The scatter *between* transits.  Each transit's mean is a low-noise estimate
    of that transit's systematic offset, so the spread of the transit means is the
    systematic error.

The two are computed in the Gaia scan frame (along-scan ``AL`` / across-scan
``AC``), which is where Gaia's errors are physically defined: ``AL`` is the
precise direction, ``AC`` is coarse.  Mixing the two into RA/Dec would average a
sub-mas axis with a ~mas axis and hide the structure.

This module provides the pure computation (:func:`compute_transit_error_components`,
unit-testable without a system of bodies) and the figure that shows it
(:class:`GaiaTransitErrorComponents`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from orbitdet.analysis import ObservationAccess, analysis_frame
from orbitdet.visualization.base import _make_hover_formatter
from orbitdet.visualization.residual_base import ResidualPlot, _cfg_get
from orbitdet.visualization.residuals import (
    _configure_datetime_axis,
    _rad_to_unit,
    _seconds_since_j2000_to_datetimes,
)

logger = logging.getLogger(__name__)

#: Default transit gap (seconds).  Within a transit the samples span tens of
#: seconds; successive transits of the same source are separated by at least a
#: Gaia rotation (~106 min).  30 minutes sits safely between the two.
DEFAULT_TRANSIT_GAP_SECONDS = 1800.0


@dataclass
class TransitErrorComponents:
    """Result of :func:`compute_transit_error_components`.

    All angular quantities are in radians and ``backend`` order (sorted by
    epoch).  ``transit_*`` arrays are per transit; the rest are per observation.
    """

    #: Observation epochs (seconds since J2000 TDB), sorted.
    epochs: np.ndarray
    #: Along-scan residual per observation (radians).
    al: np.ndarray
    #: Across-scan residual per observation (radians).
    ac: np.ndarray
    #: Integer transit label per observation (0, 0, ..., 1, 1, ...).
    transit_id: np.ndarray
    #: Mean epoch of each transit (seconds since J2000 TDB).
    transit_epoch: np.ndarray
    #: Mean along-scan residual of each transit (radians).
    transit_mean_al: np.ndarray
    #: Mean across-scan residual of each transit (radians).
    transit_mean_ac: np.ndarray
    #: Number of observations in each transit.
    transit_count: np.ndarray
    #: Pooled within-transit 1σ (random) along-scan error (radians); NaN if
    #: undefined (no transit has more than one observation).
    random_sigma_al: float
    #: Pooled within-transit 1σ (random) across-scan error (radians).
    random_sigma_ac: float
    #: Spread (sample std) of the transit means (systematic) along-scan (radians);
    #: NaN if fewer than two transits.
    systematic_sigma_al: float
    #: Spread (sample std) of the transit means (systematic) across-scan (radians).
    systematic_sigma_ac: float


def _transit_labels(epochs: np.ndarray, gap_seconds: float) -> np.ndarray:
    """Label observations with a transit index, splitting on epoch gaps.

    A new transit starts wherever consecutive epochs differ by more than
    ``gap_seconds``.  ``epochs`` must be sorted ascending.
    """
    if epochs.size == 0:
        return np.zeros(0, dtype=int)
    if epoch_sorted_violation := np.any(np.diff(epochs) < 0):
        raise ValueError(
            f"epochs must be sorted ascending; found a decrease ({epoch_sorted_violation})."
        )
    starts = np.concatenate([[False], np.diff(epochs) > gap_seconds])
    return np.cumsum(starts).astype(int)


def compute_transit_error_components(
    epochs: np.ndarray,
    al: np.ndarray,
    ac: np.ndarray,
    gap_seconds: float = DEFAULT_TRANSIT_GAP_SECONDS,
) -> TransitErrorComponents:
    """Split scan-frame residual scatter into random and systematic components.

    The observations are grouped into transits by epoch gaps (see
    :func:`_transit_labels`).  Then:

    * **random** (within transit) — pooled RMS of the residuals about each
      transit's own mean, ``sqrt( Σ_j Σ_i (x_ij − x̄_j)² / (N − M) )`` for ``M``
      transits and ``N`` observations.  Undefined (NaN) when no transit has more
      than one observation.
    * **systematic** (between transits) — sample standard deviation of the ``M``
      transit means.  Undefined (NaN) with fewer than two transits.

    Non-finite epochs or residuals are dropped, and the survivors are sorted by
    epoch so the transit labels are well defined.

    Args:
        epochs: Observation epochs (seconds since J2000 TDB).
        al: Along-scan residual per observation (radians).
        ac: Across-scan residual per observation (radians).
        gap_seconds: Epoch gap above which a new transit starts.

    Returns:
        A :class:`TransitErrorComponents`.

    Raises:
        ValueError: If the three inputs do not have matching lengths.
    """
    epochs = np.asarray(epochs, dtype=float)
    al = np.asarray(al, dtype=float)
    ac = np.asarray(ac, dtype=float)
    if not (epochs.size == al.size == ac.size):
        raise ValueError(
            f"epochs, al and ac must have matching lengths, got "
            f"{epochs.size}, {al.size}, {ac.size}."
        )

    finite = np.isfinite(epochs) & np.isfinite(al) & np.isfinite(ac)
    epochs, al, ac = epochs[finite], al[finite], ac[finite]

    order = np.argsort(epochs, kind="stable")
    epochs, al, ac = epochs[order], al[order], ac[order]

    n = epochs.size
    if n == 0:
        return TransitErrorComponents(
            epochs=epochs,
            al=al,
            ac=ac,
            transit_id=np.zeros(0, dtype=int),
            transit_epoch=np.zeros(0),
            transit_mean_al=np.zeros(0),
            transit_mean_ac=np.zeros(0),
            transit_count=np.zeros(0, dtype=int),
            random_sigma_al=float("nan"),
            random_sigma_ac=float("nan"),
            systematic_sigma_al=float("nan"),
            systematic_sigma_ac=float("nan"),
        )

    transit_id = _transit_labels(epochs, gap_seconds)
    m = int(transit_id[-1]) + 1

    transit_epoch = np.empty(m, dtype=float)
    transit_mean_al = np.empty(m, dtype=float)
    transit_mean_ac = np.empty(m, dtype=float)
    transit_count = np.empty(m, dtype=int)
    for j in range(m):
        segment = transit_id == j
        transit_count[j] = int(segment.sum())
        transit_epoch[j] = float(epochs[segment].mean())
        transit_mean_al[j] = float(al[segment].mean())
        transit_mean_ac[j] = float(ac[segment].mean())

    # Random: pooled within-transit variance (dof = N - M).
    within_al = al - transit_mean_al[transit_id]
    within_ac = ac - transit_mean_ac[transit_id]
    dof = n - m
    if dof > 0:
        random_sigma_al = float(np.sqrt(np.sum(within_al**2) / dof))
        random_sigma_ac = float(np.sqrt(np.sum(within_ac**2) / dof))
    else:
        random_sigma_al = float("nan")
        random_sigma_ac = float("nan")

    # Systematic: spread of the transit means.
    if m > 1:
        systematic_sigma_al = float(np.std(transit_mean_al, ddof=1))
        systematic_sigma_ac = float(np.std(transit_mean_ac, ddof=1))
    else:
        systematic_sigma_al = float("nan")
        systematic_sigma_ac = float("nan")

    return TransitErrorComponents(
        epochs=epochs,
        al=al,
        ac=ac,
        transit_id=transit_id,
        transit_epoch=transit_epoch,
        transit_mean_al=transit_mean_al,
        transit_mean_ac=transit_mean_ac,
        transit_count=transit_count,
        random_sigma_al=random_sigma_al,
        random_sigma_ac=random_sigma_ac,
        systematic_sigma_al=systematic_sigma_al,
        systematic_sigma_ac=systematic_sigma_ac,
    )


def _scan_frame_residuals(access: ObservationAccess, include_rejected: bool = False):
    """Return ``(epochs, al, ac)`` in the scan frame, read from the analysis frame.

    The scan-frame rotation is already applied by
    :func:`orbitdet.analysis.frames.analysis_frame`, so this only selects the
    rows that carry a finite scan angle and returns the rotated columns.  Returns
    ``None`` when no row has a scan angle (a non-scanning source).
    """
    frame = analysis_frame(access, unit=None, scan=True, normalized=False)
    al_col = "residual_along_scan_rad"
    ac_col = "residual_across_scan_rad"
    if "scan_angle" not in frame.columns or al_col not in frame.columns:
        logger.debug("No scan angles in the frame; skipping transit error components.")
        return None
    frame = frame.loc[frame["scan_angle"].notna()]
    if not include_rejected and "rejected" in frame.columns:
        frame = frame.loc[~frame["rejected"].astype(bool)]
    if frame.empty:
        return None
    epochs = frame["epoch_tdb"].to_numpy("float64")
    al = frame[al_col].to_numpy("float64")
    ac = frame[ac_col].to_numpy("float64")
    return epochs, al, ac


class GaiaTransitErrorComponents(ResidualPlot):
    """Plot the Gaia random/systematic error components of the scan-frame residuals.

    One panel per scan-frame component (along-scan ``AL``, across-scan ``AC``).
    Each panel shows the per-observation residuals (light) and the per-transit
    means (bold), so the within-transit scatter (random) and the spread of the
    transit means (systematic) are both visible; the two derived 1σ values are
    annotated in the panel.

    Only Gaia observations are meaningful here, and only rows with a scan angle
    contribute; other rows are skipped.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (``conf/figures/<name>.yaml``).
        ``axes.unit``, ``include_rejected`` and ``transit_gap_seconds`` are read
        from here.
    access : ObservationAccess
        The access layer whose derived frame is plotted.
    """

    @property
    def transit_gap_seconds(self) -> float:
        """Epoch gap (seconds) above which a new transit starts, from the config."""
        return float(
            _cfg_get(
                self.cfg,
                "transit_gap_seconds",
                default=DEFAULT_TRANSIT_GAP_SECONDS,
            )
        )

    def components(self) -> TransitErrorComponents:
        """Compute the pooled transit error components across all scan rows."""
        gap = self.transit_gap_seconds

        at_scan = _scan_frame_residuals(self.access, self.include_rejected)
        if at_scan is None:
            raise ValueError(
                "GaiaTransitErrorComponents: no observation with a scan angle and residuals "
                "was found."
            )
        epochs, al, ac = at_scan

        return compute_transit_error_components(epochs, al, ac, gap_seconds=gap)

    def _make_figure(self):
        plot_cfg = self.cfg
        fig_w, fig_h = self._figure_size(8.27 * 2, 8.27 * 2 / 2)
        unit = _cfg_get(plot_cfg, "axes", "unit", default="mas")
        cmap = _cfg_get(plot_cfg, "styling", "cmap", default="tab10")
        colors = plt.get_cmap(cmap)
        marker_size = _cfg_get(plot_cfg, "styling", "marker_size", default=30)

        components = self.components()
        times = _seconds_since_j2000_to_datetimes(components.epochs)
        transit_times = _seconds_since_j2000_to_datetimes(components.transit_epoch)

        al_unit = _rad_to_unit(components.al, unit)
        ac_unit = _rad_to_unit(components.ac, unit)
        mean_al_unit = _rad_to_unit(components.transit_mean_al, unit)
        mean_ac_unit = _rad_to_unit(components.transit_mean_ac, unit)
        random_al_unit = float(_rad_to_unit(components.random_sigma_al, unit))
        random_ac_unit = float(_rad_to_unit(components.random_sigma_ac, unit))
        syst_al_unit = float(_rad_to_unit(components.systematic_sigma_al, unit))
        syst_ac_unit = float(_rad_to_unit(components.systematic_sigma_ac, unit))

        fig, axs = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharex=False)
        panels = [
            (axs[0], "Along-scan", al_unit, mean_al_unit, random_al_unit, syst_al_unit),
            (axs[1], "Across-scan", ac_unit, mean_ac_unit, random_ac_unit, syst_ac_unit),
        ]
        for ax, title, values, means, sigma_random, sigma_syst in panels:
            ax.scatter(
                times,
                values,
                marker="o",
                s=marker_size * 0.5,
                color=colors(0),
                alpha=0.25,
                label="Observations",
            )
            ax.scatter(
                transit_times,
                means,
                marker="D",
                s=marker_size,
                color=colors(1),
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label="Transit means",
            )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel(f"Residual [{unit}]")
            ax.set_xlabel(_cfg_get(plot_cfg, "axes", "x_label", default="Epoch"))
            _configure_datetime_axis(ax)

            ax.annotate(
                f"random σ = {sigma_random:.3g} {unit}\n"
                f"systematic σ = {sigma_syst:.3g} {unit}\n"
                f"({components.transit_epoch.size} transits, "
                f"{components.epochs.size} obs)",
                xy=(0.02, 0.98),
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            ax.legend(loc="upper right", fontsize=8)

            hover_x = _cfg_get(plot_cfg, "axes", "hover_x_label", default="Epoch")
            hover_y = _cfg_get(plot_cfg, "axes", "hover_y_label", default=f"Residual [{unit}]")
            ax.format_coord = _make_hover_formatter(hover_x, hover_y)

        suptitle = _cfg_get(
            plot_cfg,
            "titles",
            "suptitle",
            default="Gaia Transit Error Components (scan frame)",
        )
        fig.suptitle(suptitle)
        fig.set_tight_layout(True)
        return fig, axs
