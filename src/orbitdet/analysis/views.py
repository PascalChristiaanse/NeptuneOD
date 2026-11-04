"""Pure view functions over a derived observation frame.

The derived frame built by :class:`orbitdet.analysis.access.ObservationAccess`
keeps the **canonical** representation: radians, with the two scalar components
kept as ``first`` / ``second``.  A *view* is a pure function that returns a
**copy** in another representation - arcseconds, named ``(RA, Dec)``, or the
along/across-scan frame.  Nothing is ever written back to the frame, so there is
exactly one stored representation and the conversions are explicit, cheap and
impossible to drift out of sync.

All functions here

* take a dataframe and return a new one (the input is never mutated),
* leave the canonical columns in place and *add* the converted columns, so a
  view can be layered on another view,
* take quantities in radians and return radians unless the function name says
  otherwise (``to_arcsec`` is the deliberate exception).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ARCSEC_PER_RADIAN",
    "canonical_angle_columns",
    "to_arcsec",
    "to_mas",
    "rotate",
    "with_scan_frame",
]

#: Arcseconds in one radian - the factor a unit view multiplies radians by.
ARCSEC_PER_RADIAN: float = np.rad2deg(1.0) * 3600.0

#: Canonical angle columns that a unit view converts, in canonical order.
_ANGLE_COLUMNS: tuple[str, ...] = (
    "residual_first_rad",
    "residual_second_rad",
    "sigma_first_rad",
    "sigma_second_rad",
    "first_value",
    "second_value",
)

#: Unit-name -> (factor from radians, suffix).  ``arcsec`` is the project's
#: default reporting unit; ``mas`` is included because Gaia figures use it.
_UNIT_FACTORS: dict[str, float] = {
    "arcsec": ARCSEC_PER_RADIAN,
    "mas": ARCSEC_PER_RADIAN * 1e3,
}


def canonical_angle_columns(df: pd.DataFrame) -> list[str]:
    """Return the canonical angle columns present in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame (or any frame with some subset of the
        canonical angle columns).

    Returns
    -------
    list[str]
        The members of the canonical angle-column set that ``df`` contains.
    """
    return [name for name in _ANGLE_COLUMNS if name in df.columns]


def _to_unit(df: pd.DataFrame, unit: str, columns: list[str] | None) -> pd.DataFrame:
    """Return a copy of ``df`` with the angle columns converted to ``unit``.

    The converted columns are named ``<column>_<unit>``; the canonical columns
    are kept unchanged.  Every function in this module funnels through here so
    the unit handling stays in one place.
    """
    if unit not in _UNIT_FACTORS:
        raise ValueError(f"Unknown unit '{unit}'; expected one of {sorted(_UNIT_FACTORS)}.")
    factor = _UNIT_FACTORS[unit]
    out = df.copy()
    for name in columns if columns is not None else canonical_angle_columns(df):
        out[f"{name}_{unit}"] = out[name].astype("float64") * factor
    return out


def to_arcsec(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a copy with the canonical angle columns converted to arcseconds.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame.
    columns : list[str] | None
        The canonical columns to convert.  Defaults to every canonical angle
        column present.

    Returns
    -------
    pd.DataFrame
        A copy with ``<column>_arcsec`` columns added.  The canonical columns
        are unchanged.
    """
    return _to_unit(df, "arcsec", columns)


def to_mas(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a copy with the canonical angle columns converted to milliarcsec.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame.
    columns : list[str] | None
        The canonical columns to convert.  Defaults to every canonical angle
        column present.

    Returns
    -------
    pd.DataFrame
        A copy with ``<column>_mas`` columns added.  The canonical columns are
        unchanged.
    """
    return _to_unit(df, "mas", columns)


def rotate(
    first: np.ndarray | pd.Series,
    second: np.ndarray | pd.Series,
    dec: np.ndarray | pd.Series,
    psi: np.ndarray | pd.Series,
    *,
    de_projected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a two-component quantity into the scan frame.

    The convention matches :func:`orbitdet.analysis.basis.scan_basis` and the
    weighting strategies::

        east  = first * cos(dec)
        north = second
        along  = east * sin(psi) + north * cos(psi)
        across = east * cos(psi) - north * sin(psi)

    Parameters
    ----------
    first, second : array-like
        The two components of the quantity to rotate.
    dec : array-like
        Declination-like angle (radians) used for the ``cos(dec)`` factor.
        Ignored when ``de_projected`` is true.
    psi : array-like
        Position angle of scan, north through east (radians).
    de_projected : bool
        Whether ``first`` is already de-projected by ``cos(dec)`` (i.e. it holds
        ``Δα·cosδ`` rather than ``Δα``).  When ``True`` no ``cos(dec)`` factor is
        applied, which is the tangent-plane convention the standard frame's
        equatorial first component uses.  Defaults to ``False`` to preserve the
        historical raw-``Δα`` behaviour.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(along, across)``, same units as the inputs.
    """
    first = np.asarray(first, dtype="float64")
    second = np.asarray(second, dtype="float64")
    psi = np.asarray(psi, dtype="float64")

    east = first if de_projected else first * np.cos(np.asarray(dec, dtype="float64"))
    north = second
    along = east * np.sin(psi) + north * np.cos(psi)
    across = east * np.cos(psi) - north * np.sin(psi)
    return along, across


def with_scan_frame(
    df: pd.DataFrame,
    psi: np.ndarray | pd.Series | None = None,
    dec: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Return a copy with residuals and sigmas in the along/across-scan frame.

    Only meaningful for a scanning instrument (e.g. Gaia).  The scan position
    angle is taken from the ``scan_angle`` column when ``psi`` is not given, and
    the declination from ``second_value`` when ``dec`` is not given.  The scan
    frame is the frame the instrument's uncertainties are diagonal in; mixing it
    with the canonical frame would average a sub-mas axis with a mas axis and
    hide the structure.

    Parameters
    ----------
    df : pd.DataFrame
        A derived observation frame.
    psi : array-like | None
        Scan position angle (radians).  Defaults to the ``scan_angle`` column.
    dec : array-like | None
        Declination-like angle (radians) for the tangent-plane factor.  Defaults
        to the ``second_value`` column.

    Returns
    -------
    pd.DataFrame
        A copy with ``residual_along_scan_rad``, ``residual_across_scan_rad``,
        ``sigma_along_scan_rad`` and ``sigma_across_scan_rad`` added.  The
        canonical columns are unchanged.

    Raises
    ------
    ValueError
        If the scan angle is not supplied and the frame has no ``scan_angle``
        column, or if the frame is missing the residual columns the rotation
        needs.
    """
    if psi is None:
        if "scan_angle" not in df.columns:
            raise ValueError(
                "Scan frame requested but no psi was supplied and the frame has no "
                "'scan_angle' column."
            )
        psi = df["scan_angle"]
    if dec is None:
        if "second_value" not in df.columns:
            raise ValueError(
                "Scan frame requested but no dec was supplied and the frame has no "
                "'second_value' column."
            )
        dec = df["second_value"]

    missing = [
        name for name in ("residual_first_rad", "residual_second_rad") if name not in df.columns
    ]
    if missing:
        raise ValueError(f"Frame is missing residual column(s) required to rotate: {missing}.")

    out = df.copy()

    along, across = rotate(out["residual_first_rad"], out["residual_second_rad"], dec, psi)
    out["residual_along_scan_rad"] = along
    out["residual_across_scan_rad"] = across

    if "sigma_first_rad" in out.columns and "sigma_second_rad" in out.columns:
        # Sigma is a magnitude, not a signed component: rotate the variances as
        # a diagonal pair and take the root, which is exact for a scan-frame
        # diagonal covariance.
        var_first = out["sigma_first_rad"].to_numpy("float64") ** 2
        var_second = out["sigma_second_rad"].to_numpy("float64") ** 2
        e_var = var_first * np.cos(np.asarray(dec, dtype="float64")) ** 2
        n_var = var_second
        psi_arr = np.asarray(psi, dtype="float64")
        sin_psi = np.sin(psi_arr)
        cos_psi = np.cos(psi_arr)
        var_along = e_var * sin_psi**2 + n_var * cos_psi**2
        var_across = e_var * cos_psi**2 + n_var * sin_psi**2
        out["sigma_along_scan_rad"] = np.sqrt(np.clip(var_along, 0.0, None))
        out["sigma_across_scan_rad"] = np.sqrt(np.clip(var_across, 0.0, None))

    return out
