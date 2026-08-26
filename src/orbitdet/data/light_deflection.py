"""
Relativistic light-deflection corrections to astrometric observations.

Post-Newtonian light-bending by the Sun (and optionally other bodies) shifts the
observed position of a target relative to its true direction.  This module
computes the per-observation RA/DEC correction that must be ADDED to the
observed coordinates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tudatpy.constants import SPEED_OF_LIGHT

logger = logging.getLogger(__name__)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-300:
        raise ZeroDivisionError("Cannot normalise a (near-)zero vector.")
    return vector / norm


def _calculate_light_deflection(
    gaia_state: np.ndarray,
    asteroid_state: np.ndarray,
    body_state: np.ndarray,
    mu_body: float,
) -> np.ndarray:
    """Helper to calculate the post-Newtonian light-bending contribution from one body.

    Args:
        gaia_state: SSB position of Gaia at time of observation.
        asteroid_state: SSB position of the target at time of emission.
        body_state: SSB position of the deflecting body at reference time.
        mu_body: Gravitational parameter of the deflecting body.

    Returns:
        np.ndarray: Individual contribution to total light deflection.
    """
    norm = np.linalg.norm

    # Define vectors
    r_upper = gaia_state - asteroid_state  # R
    r_ea = asteroid_state - body_state
    r_oa = gaia_state - body_state

    # Calculate contribution of deflection from body
    factor = (2 * mu_body) / (SPEED_OF_LIGHT**2)
    num = np.cross(r_upper, np.cross(r_ea, r_oa))
    denom = norm(r_upper) * norm(r_oa) * (norm(r_ea) * norm(r_oa) + np.dot(r_oa, r_ea))
    delta_k_pn = factor * num / denom
    assert delta_k_pn.shape == (3,)

    return delta_k_pn


def _deflection_vector_to_corrections(
    ra: float, dec: float, delta_k_pn: np.ndarray
) -> tuple[float, float]:
    """Convert a deflection vector into RA/DEC corrections.

    Args:
        ra: RA in radians.
        dec: Dec in radians.
        delta_k_pn: Light deflection vector.

    Returns:
        (ra_correction, dec_correction) in radians.
    """
    unit = lambda vec: vec / np.linalg.norm(vec)

    # Find corrected unit observation vector
    observed_unit_vector = -np.array(  # Minus because of Klioner sign convention
        [np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)]
    )
    true_unit_vector = observed_unit_vector - delta_k_pn  # Vector k
    true_unit_vector = -unit(true_unit_vector)  # Flip sign again

    # Convert asteroid vector to RA/DEC observables
    ra_true = np.arctan2(true_unit_vector[1], true_unit_vector[0])
    dec_true = np.arctan2(
        true_unit_vector[2],
        np.sqrt(true_unit_vector[0] ** 2 + true_unit_vector[1] ** 2),
    )
    assert -np.pi <= ra_true <= np.pi
    assert -np.pi / 2 <= dec_true <= np.pi / 2

    # Get corrections
    ra_corr = ra_true - ra
    dec_corr = dec_true - dec

    return ra_corr, dec_corr


def relativistic_light_deflection(
    target_name: str,
    table: pd.DataFrame,
    bodies,
    bodies_to_include: list = ("Sun",),
) -> np.ndarray:
    """Calculate offset due to relativistic bending of light around massive bodies.

    Uses the pre-loaded ephemerides of the target and major bodies to compute
    the RA/DEC correction to be *added* to the observations.

    Args:
        target_name: Name of the observed body (e.g. ``"Triton"``) in ``bodies``.
        table: Observation table containing the target's observations and the
            archived Gaia state vectors.
        bodies: Tudat ``SystemOfBodies``; must have the target body and all the
            light-deflecting bodies with ephemerides loaded.
        bodies_to_include: A list of bodies that exert light bending.

    Returns:
        np.ndarray: (n, 2) array of [dRA, dDec] corrections (rad) to add.
    """
    table = table.reset_index(drop=True)
    assert table["epoch"].is_monotonic_increasing
    assert not table.empty, "Observation table contains no observations"

    ra_corrections = []
    dec_corrections = []

    # Retrieve target ephemeris
    target_ephemeris = bodies.get(target_name).ephemeris

    # Loop over observations
    for row in table.itertuples():
        current_epoch = row.epoch

        # Light-time from observer to target (one iteration)
        gaia_state = np.array((row.x_gaia, row.y_gaia, row.z_gaia))
        light_time_observer_target = np.linalg.norm(
            gaia_state - target_ephemeris.cartesian_position(current_epoch)
        )
        light_time_observer_target /= SPEED_OF_LIGHT

        # Target state at time of emission
        target_state = target_ephemeris.cartesian_position(
            current_epoch - light_time_observer_target
        )

        delta_k_pn_total = []  # Total deflection from all bodies
        for body in bodies_to_include:
            body_ephemeris = bodies.get(body).ephemeris
            mu_body = bodies.get(body).gravitational_parameter

            # Reference time for the deflecting body (one iteration)
            light_time_observer_body = np.linalg.norm(
                gaia_state - body_ephemeris.cartesian_position(current_epoch)
            )
            light_time_observer_body /= SPEED_OF_LIGHT

            body_state = body_ephemeris.cartesian_position(
                current_epoch - light_time_observer_body
            )

            delta_k_pn = _calculate_light_deflection(
                gaia_state, target_state, body_state, mu_body
            )
            delta_k_pn_total.append(delta_k_pn)

        # Add contributions from all bodies
        delta_k_pn_total = -np.sum(delta_k_pn_total, axis=0)
        assert delta_k_pn_total.shape == (3,)

        ra_corr, dec_corr = _deflection_vector_to_corrections(
            row.ra, row.dec, delta_k_pn_total
        )

        ra_corrections.append(ra_corr)
        dec_corrections.append(dec_corr)

    corrections = np.column_stack((ra_corrections, dec_corrections))
    assert corrections.shape == (len(table), 2)
    return corrections