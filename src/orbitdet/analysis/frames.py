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
from .views import to_arcsec, to_mas, with_scan_frame

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
        if angles is not None:
            frame["scan_angle"] = angles
            if {"residual_first_rad", "residual_second_rad"} <= set(frame.columns):
                frame = with_scan_frame(frame)

    if unit == "arcsec":
        frame = to_arcsec(frame)
    elif unit == "mas":
        frame = to_mas(frame)

    return frame


def _scan_angles_for_frame(frame: pd.DataFrame) -> np.ndarray | None:
    """Return the scan angle (radians) for every frame row, or ``None``.

    The angles live in an epoch-keyed registry, so one lookup covers the whole
    frame.  ``None`` means the frame's epochs are not registered (a non-Gaia
    collection), so no scan view is possible.
    """
    from orbitdet.data.gaia import get_scan_angles_for_epochs

    if "epoch_tdb" not in frame.columns:
        return None
    epochs = pd.to_numeric(frame["epoch_tdb"], errors="coerce").to_numpy("float64")
    if epochs.size == 0:
        return None
    angles = get_scan_angles_for_epochs(epochs)
    if angles is None or len(angles) != epochs.size:
        return None
    return np.asarray(angles, dtype="float64")
