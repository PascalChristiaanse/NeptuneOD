"""Build one tidy analysis frame from an access layer.

The residual figures each rebuild the same thing by hand: loop the observation
sets, pull scan angles from the epoch-keyed registry for the Gaia sets, convert
radians to arcseconds, and (for the scanning instrument) rotate into the
along/across-scan frame.  This module does it once, on the **whole** derived
frame, so a plot becomes "build the analysis frame, select rows, draw".

It is a composition of the pure views (:mod:`orbitdet.analysis.views`) over the
access layer's derived frame (:class:`orbitdet.analysis.access.ObservationAccess`);
it computes no new statistics and does not touch config or the file system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .access import ObservationAccess
from .statistics import normalized_residuals
from .views import canonical_angle_columns, to_arcsec, to_mas, with_scan_frame

__all__ = ["analysis_frame"]


def analysis_frame(
    access: ObservationAccess,
    *,
    unit: str | None = "arcsec",
    scan: bool = False,
    normalized: bool = True,
) -> pd.DataFrame:
    """Return a tidy frame ready for plotting, derived from an access layer.

    The returned frame is a copy of :attr:`ObservationAccess.frame` (standard
    provenance columns plus the joined residuals, weights, sigmas and ``rejected``
    flag) with, by default, the reporting-unit columns, the normalized residuals,
    and - when ``scan`` is set - the along/across-scan columns added.

    Scan angles are looked up once for the whole frame in the epoch-keyed registry
    populated by the Gaia query, so the per-set loops the figures currently run
    are not needed.  A frame with no registered scan angles simply gets no scan
    columns (the ``scan`` request is advisory, matching the figures' behaviour of
    skipping the Gaia-only view for non-Gaia collections).

    Parameters
    ----------
    access : ObservationAccess
        The access layer whose derived frame is the base of the analysis frame.
    unit : str | None
        Reporting unit for the angle columns: ``"arcsec"`` (default), ``"mas"``,
        or ``None`` to add no unit columns.
    scan : bool
        Whether to add the along/across-scan columns.  Requires a registered scan
        angle (Gaia); ignored when none is available.
    normalized : bool
        Whether to add the ``z_first`` / ``z_second`` normalized-residual columns.

    Returns
    -------
    pd.DataFrame
        A copy of the derived frame with the requested views applied.  The
        canonical columns are kept, so a caller can mix representations.

    Raises
    ------
    ValueError
        If ``unit`` is not ``None``, ``"arcsec"`` or ``"mas"``.
    """
    if unit not in (None, "arcsec", "mas"):
        raise ValueError(f"Unknown unit '{unit}'; expected None, 'arcsec' or 'mas'.")

    frame = access.frame.copy()

    if normalized:
        frame = normalized_residuals(frame)

    if scan:
        angles = _scan_angles_for_frame(frame)
        # Only rotate when at least one row carries a finite scan angle; a frame
        # with an all-NaN scan_angle (non-scanning instrument) gets no scan view.
        if angles is not None and np.any(np.isfinite(angles)):
            frame["scan_angle"] = angles
            if {"residual_first_rad", "residual_second_rad"} <= set(frame.columns):
                frame = with_scan_frame(frame)

    if unit == "arcsec":
        frame = to_arcsec(frame, columns=_angle_columns(frame))
    elif unit == "mas":
        frame = to_mas(frame, columns=_angle_columns(frame))

    return frame


def _angle_columns(frame: pd.DataFrame) -> list[str]:
    """Canonical *and* scan-frame angle columns present in ``frame``."""
    columns = list(canonical_angle_columns(frame))
    for name in (
        "residual_along_scan_rad",
        "residual_across_scan_rad",
        "sigma_along_scan_rad",
        "sigma_across_scan_rad",
    ):
        if name in frame.columns:
            columns.append(name)
    return columns


def _scan_angles_for_frame(frame: pd.DataFrame) -> np.ndarray | None:
    """Return the ``scan_angle`` column as a float array, or ``None`` if absent.

    The scan angle is a standard-frame column, populated on ingest (from the
    Gaia archive's ``position_angle_scan``); it is read straight from the frame
    rather than looked up in an epoch-keyed registry.  Rows without a finite
    angle (a non-scanning instrument) are left as ``NaN`` by the caller's
    ``with_scan_frame`` mask; ``None`` here means the column is missing entirely.
    """
    if "scan_angle" not in frame.columns:
        return None
    return pd.to_numeric(frame["scan_angle"], errors="coerce").to_numpy("float64")
