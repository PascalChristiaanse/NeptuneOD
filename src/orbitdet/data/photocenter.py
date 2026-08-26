"""
Photocenter-offset corrections to astrometric observations.

For a resolved body the detected photocenter is offset from the center of mass
along the Sun-target-observer direction.  This module computes the RA/DEC
correction (to be *added* to the observed coordinates) assuming a spherical
shape with isotropic scattering, using a given body diameter.

For asteroids the diameter is normally queried from JPL SBDB, but for other
bodies (e.g. Triton) it can be passed explicitly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / norm(vector)


def _solar_phase_angle(
    target_wrt_gaia_unit: np.ndarray, target_wrt_sun_unit: np.ndarray
) -> float:
    """Solar phase angle in radians (observer-target-Sun angle)."""
    return np.arccos(np.dot(target_wrt_gaia_unit, target_wrt_sun_unit))


def _offset_magnitude(
    solar_phase_angle: float, diameter: float, target_gaia_distance: float
) -> float:
    """Magnitude (rad) of the photocenter offset (Fuentes-Munoz 2024)."""
    cot = lambda x: np.cos(x) / np.sin(x)
    num = 2 * (
        np.sin(solar_phase_angle)
        + (np.pi - solar_phase_angle) * np.cos(solar_phase_angle)
    )
    denom = 3 * np.pi * (
        cot(solar_phase_angle / 2)
        - np.sin(solar_phase_angle / 2) * np.log(cot(solar_phase_angle / 4))
    )
    offset_ratio = num / denom  # Fraction of body radius
    offset_ratio = max(0.0, min(1.0, float(offset_ratio)))
    return offset_ratio * (diameter / 2) / target_gaia_distance


def _offset_direction(
    target_wrt_gaia_unit: np.ndarray, target_wrt_sun_unit: np.ndarray
) -> np.ndarray:
    """Direction (unit) of the photocenter offset vector."""
    offset_dir = -(
        target_wrt_sun_unit
        - np.dot(target_wrt_sun_unit, target_wrt_gaia_unit) * target_wrt_gaia_unit
    )
    return _unit(offset_dir)


def _offset_vector_to_corrections(offset_vec: np.ndarray, ra: float, dec: float):
    """Convert a plane-of-sky offset vector to RA/DEC corrections (small-angle)."""
    observed_dir = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    # true dir. + offset = observed dir.
    true_dir = observed_dir - offset_vec  # Small angle approximation
    true_dir = _unit(true_dir)

    ra_true = np.arctan2(true_dir[1], true_dir[0])
    dec_true = np.arctan2(true_dir[2], np.sqrt(true_dir[0] ** 2 + true_dir[1] ** 2))
    return ra_true - ra, dec_true - dec


def photocenter_offset_spherical(
    target_name: str,
    table: pd.DataFrame,
    bodies,
    diameter: float,
) -> np.ndarray:
    """Estimate the photocenter offset assuming spherical, isotropically scattering body.

    Args:
        target_name: Name of the observed body (e.g. ``"Triton"``) in ``bodies``.
        table: Observation table containing the target's observations and the
            archived Gaia state vectors.
        bodies: Tudat ``SystemOfBodies``; must contain the target body and the
            Sun, both with ephemerides loaded.
        diameter: Body diameter in meters.  For Gaia asteroids this is normally
            retrieved from SBDB; pass it explicitly for other bodies (e.g.
            Triton ~ 2706800 m).

    Returns:
        np.ndarray: (n, 2) array of [dRA, dDec] corrections (rad) to add.
    """
    # Geometry is approx constant over a transit (~39 s): use one row per transit.
    table_reduced = table.drop_duplicates(subset="transit_id", ignore_index=True)
    assert table_reduced["epoch"].is_monotonic_increasing
    assert not table_reduced.empty

    target_ephemeris = bodies.get(target_name).ephemeris  # SSB
    sun_ephemeris = bodies.get("Sun").ephemeris  # SSB

    ra_corrections = []
    dec_corrections = []

    for row in table_reduced.itertuples():
        target_wrt_ssb = target_ephemeris.cartesian_position(row.epoch)
        sun_wrt_ssb = sun_ephemeris.cartesian_position(row.epoch)
        gaia_wrt_ssb = np.array((row.x_gaia, row.y_gaia, row.z_gaia))

        target_wrt_sun_unit = _unit(target_wrt_ssb - sun_wrt_ssb)
        target_wrt_gaia_unit = _unit(target_wrt_ssb - gaia_wrt_ssb)
        target_gaia_distance = norm(target_wrt_ssb - gaia_wrt_ssb)

        solar_phase_angle = _solar_phase_angle(target_wrt_gaia_unit, target_wrt_sun_unit)
        offset_vec = (
            _offset_magnitude(solar_phase_angle, diameter, target_gaia_distance)
            * _offset_direction(target_wrt_gaia_unit, target_wrt_sun_unit)
        )

        ra_corr, dec_corr = _offset_vector_to_corrections(offset_vec, row.ra, row.dec)

        # Apply the same correction to every observation in the transit
        transit_length = int((table["transit_id"] == row.transit_id).sum())
        ra_corrections.extend([ra_corr] * transit_length)
        dec_corrections.extend([dec_corr] * transit_length)

    corrections = np.column_stack((ra_corrections, dec_corrections))
    assert corrections.shape == (len(table), 2)
    return corrections