"""Dedicated frame conversions: apparent to astrometric, and back.

Light arriving from a distant body does not travel along the direction from
which it appears to have come.  Two observer-motion effects displace the
apparent direction:

Aberration
    The observer's velocity tilts the incoming wavefront.  For a ground station
    carried by the Earth this is up to **20.6 arcsec**, and it varies annually
    as the Earth's velocity direction changes.  A diurnal component of up to
    ~0.3 arcsec is added by the station's rotation.

Light-time
    The body moved while the light was in transit.  Handled by the observation
    model, not here.

Gravitational deflection
    Handled by :mod:`orbitdet.preprocessing.corrections` (relativistic light
    bending), because it needs the masses and positions of the deflecting
    bodies.

Why this is separate from precession
------------------------------------
Precession and aberration are independent, and conflating them is a common
source of error.  An observation may be astrometric in the equinox of date, or
apparent in J2000; the four combinations all occur in the historical record.
:mod:`orbitdet.preprocessing.precession` owns the *orientation* of the axes, and
this module owns the *kind* of direction.  They compose in either order because
both are rotations of the same unit vector.

What "astrometric" and "apparent" mean here
-------------------------------------------
``APPARENT``
    The direction as seen by the moving observer, including aberration.  This
    is what a telescope actually records, and what a catalogue of observed
    positions contains.
``ASTROMETRIC``
    The direction corrected for aberration but still including light-time and
    deflection: the direction from the observer to where the body *was*.  This
    is what an orbit determination needs to compare against.

So correcting an observation means **subtracting** aberration, which is done by
applying the transformation with the *negated* observer velocity.

Sign convention
---------------
The relativistic aberration formula is applied as it appears in the IAU SOFA
``iauAb`` routine, using the exact relativistic composition of velocities
rather than the classical small-angle approximation.  The relativistic terms
matter at the 0.001 arcsec level for Earth's orbital motion, which is inside the
precision of the best modern observations, so the exact form is used
unconditionally.

References
----------
* IAU SOFA, ``iauAb`` - apply aberration to transform natural direction into
  proper direction.
* Klioner, S. A. (2003), "A practical relativistic model for microarcsecond
  astrometry in the solar system", AJ 125, 1580.
* Explanatory Supplement to the Astronomical Almanac, 3rd ed., section 7.2.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

from orbitdet.preprocessing.precession import (
    radec_to_unit_vectors,
    unit_vectors_to_radec,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DirectionKind",
    "SPEED_OF_LIGHT",
    "aberration_beta_from_velocity",
    "apply_aberration",
    "correct_to_astrometric",
    "correct_to_apparent",
    "aberration_angle_radians",
]


#: Speed of light in vacuum, in m/s.  Matches the value used by TudatPy and
#: SPICE, so the aberration and light-time corrections stay consistent.
SPEED_OF_LIGHT = 299792458.0


class DirectionKind(StrEnum):
    """Whether a direction includes the observer's aberration.

    Mirrors :class:`orbitdet.preprocessing.schema.ReferenceFrame`, but lives
    here so this module has no dependency on the dataframe schema.
    """

    APPARENT = "apparent"
    ASTROMETRIC = "astrometric"


# ---------------------------------------------------------------------------
# Observer velocity
# ---------------------------------------------------------------------------


def aberration_beta_from_velocity(velocity: np.ndarray) -> np.ndarray:
    """Convert an observer velocity to the aberration vector ``beta = v / c``.

    Args:
        velocity: Observer velocity in m/s, in the same frame as the directions
            being corrected.  Shape ``(3,)`` for a single velocity or ``(n, 3)``
            for one per observation.

    Returns:
        ``beta``, dimensionless, with the same shape as ``velocity``.

    Raises:
        ValueError: If ``velocity`` does not have 3 components on the last axis.
    """
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape == ():
        raise ValueError("velocity must be an array of 3-vectors, got a scalar")
    if velocity.shape[-1] != 3:
        raise ValueError(
            f"velocity must have 3 components on the last axis, got shape {velocity.shape}"
        )
    return velocity / SPEED_OF_LIGHT


# ---------------------------------------------------------------------------
# Aberration
# ---------------------------------------------------------------------------


def apply_aberration(
    directions: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Apply relativistic aberration to unit direction vectors.

    Uses the exact relativistic composition of the incoming direction with the
    observer velocity, as in IAU SOFA ``iauAb``:

    .. math::

        \\mathbf{u}' = \\frac{\\sqrt{1-\\beta^2}\\,\\mathbf{u}
            + \\left(1 + \\frac{\\mathbf{u}\\cdot\\boldsymbol\\beta}{1+\\sqrt{1-\\beta^2}}\\right)
            \\boldsymbol\\beta}
            {\\left\\|\\dots\\right\\|}

    The classical small-angle form :math:`\\mathbf{u}' = \\mathbf{u} +
    \\boldsymbol\\beta` differs from this at the 0.001 arcsec level for Earth's
    orbital motion, so the exact form is used unconditionally.

    Applying the transformation with ``beta`` converts an astrometric direction
    to an apparent one; applying it with ``-beta`` converts back.  The
    transformation is exactly invertible.

    Args:
        directions: Unit vectors, shape ``(3,)`` or ``(n, 3)``.  Need not be
            normalised.
        beta: Observer velocity divided by the speed of light.  Shape ``(3,)``,
            or ``(n, 3)`` to pair one velocity with each direction.

    Returns:
        Aberrated unit vectors, same shape as ``directions``.

    Raises:
        ValueError: If ``beta`` has a shape that cannot pair with ``directions``,
            or if any ``|beta| >= 1``.

    Examples:
        Zero observer velocity leaves the direction unchanged:

        >>> import numpy as np
        >>> u = np.array([[1.0, 0.0, 0.0]])
        >>> bool(np.allclose(apply_aberration(u, np.zeros(3)), u))
        True
    """
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    beta = np.asarray(beta, dtype=float)

    if beta.ndim == 1:
        if beta.shape[0] != 3:
            raise ValueError(f"beta must have 3 components, got shape {beta.shape}")
        beta = np.broadcast_to(beta, directions.shape)
    elif beta.shape != directions.shape:
        raise ValueError(
            f"beta has shape {beta.shape} but directions has shape {directions.shape}; "
            f"they must match, or beta must be a single 3-vector"
        )

    beta_squared = np.einsum("ij,ij->i", beta, beta)
    if np.any(beta_squared >= 1.0):
        raise ValueError(
            f"|beta| must be below 1, got a maximum of {np.sqrt(beta_squared.max()):.6f}"
        )

    # Normalise the input so the formula's assumptions hold regardless of how
    # the caller scaled their vectors.
    norms = np.linalg.norm(directions, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = directions / safe[:, None]

    beta_magnitude_complement = np.sqrt(1.0 - beta_squared)
    dot = np.einsum("ij,ij->i", unit, beta)
    weight = 1.0 + dot / (1.0 + beta_magnitude_complement)

    aberrated = beta_magnitude_complement[:, None] * unit + weight[:, None] * beta
    result = aberrated / np.linalg.norm(aberrated, axis=1)[:, None]

    # Preserve null input rows rather than turning them into a spurious unit
    # vector pointing along beta.
    result = np.where(norms[:, None] == 0.0, np.nan, result)
    return result


def correct_to_astrometric(
    ra: np.ndarray,
    dec: np.ndarray,
    velocity: np.ndarray,
    *,
    wrap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove aberration, converting apparent directions to astrometric ones.

    This is the operation the pipeline performs on an observation that was
    reduced to an apparent reference frame: the recorded direction is what the
    telescope saw, and orbit determination needs the direction before the
    observer's motion tilted it.

    Args:
        ra: Apparent right ascension in radians.
        dec: Apparent declination in radians.
        velocity: Observer velocity in m/s, in the frame of the coordinates.
            Shape ``(3,)`` or ``(n, 3)``.
        wrap: If ``True``, wrap the output right ascension into ``[0, 2*pi)``.
            If ``False`` (default), return it in ``[-pi, pi]``.

    Returns:
        Tuple ``(ra, dec)`` in radians, astrometric, same shape as the input.

    Examples:
        A zero velocity is the identity:

        >>> import numpy as np
        >>> ra, dec = correct_to_astrometric(np.array([1.0]), np.array([0.5]), np.zeros(3))
        >>> bool(np.isclose(ra[0], 1.0) and np.isclose(dec[0], 0.5))
        True
    """
    return _aberrate_radec(ra, dec, velocity, sign=-1.0, wrap=wrap)


def correct_to_apparent(
    ra: np.ndarray,
    dec: np.ndarray,
    velocity: np.ndarray,
    *,
    wrap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply aberration, converting astrometric directions to apparent ones.

    The forward direction of :func:`correct_to_astrometric`.  Useful for
    comparing a computed (geometric) direction against a real observation, and
    for generating simulated apparent observations.

    Args:
        ra: Astrometric right ascension in radians.
        dec: Astrometric declination in radians.
        velocity: Observer velocity in m/s, in the frame of the coordinates.
            Shape ``(3,)`` or ``(n, 3)``.
        wrap: If ``True``, wrap the output right ascension into ``[0, 2*pi)``.

    Returns:
        Tuple ``(ra, dec)`` in radians, apparent, same shape as the input.
    """
    return _aberrate_radec(ra, dec, velocity, sign=1.0, wrap=wrap)


def _aberrate_radec(
    ra: np.ndarray,
    dec: np.ndarray,
    velocity: np.ndarray,
    *,
    sign: float,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared implementation for the two directional wrappers."""
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec must have matching shapes, got {ra.shape} and {dec.shape}")

    beta = aberration_beta_from_velocity(velocity)
    if beta.ndim == 2 and beta.shape[0] == 1 and ra.shape[0] > 1:
        beta = np.broadcast_to(beta, (ra.shape[0], 3))

    # Rows with a bad angle or a missing velocity stay NaN instead of aborting
    # a long series.
    finite_beta = np.isfinite(beta).all(axis=-1) if beta.ndim == 2 else np.isfinite(beta).all()
    valid = np.isfinite(ra) & np.isfinite(dec) & np.asarray(finite_beta)

    out_ra = np.full_like(ra, np.nan)
    out_dec = np.full_like(dec, np.nan)
    if not valid.any():
        return out_ra, out_dec

    directions = radec_to_unit_vectors(ra[valid], dec[valid])
    applied = apply_aberration(directions, sign * beta[valid] if beta.ndim == 2 else sign * beta)
    out_ra[valid], out_dec[valid] = unit_vectors_to_radec(applied, wrap=wrap)

    return out_ra, out_dec


def aberration_angle_radians(
    ra: np.ndarray,
    dec: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    """Return the angular size of the aberration displacement, in radians.

    The displacement depends on the angle between the line of sight and the
    observer velocity, so it is zero along the direction of motion and largest
    perpendicular to it.  This helper makes that field visible, which is useful
    for deciding whether a dataset's ``reference_frame`` actually matters.

    Args:
        ra: Right ascension in radians.
        dec: Declination in radians.
        velocity: Observer velocity in m/s.  Shape ``(3,)`` or ``(n, 3)``.

    Returns:
        The angular separation between the astrometric and apparent directions,
        in radians, same shape as ``ra``.  NaN where the input is not finite.
    """
    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    beta = aberration_beta_from_velocity(velocity)

    finite_beta = np.isfinite(beta).all(axis=-1) if beta.ndim == 2 else np.isfinite(beta).all()
    valid = np.isfinite(ra) & np.isfinite(dec) & np.asarray(finite_beta)
    out = np.full(ra.shape, np.nan)
    if not valid.any():
        return out

    directions = radec_to_unit_vectors(ra[valid], dec[valid])
    # ``directions`` are taken as the astrometric input, so comparing them with
    # the aberrated output gives the displacement directly.  Comparing the +beta
    # and -beta images instead would double it.
    apparent = apply_aberration(directions, beta[valid] if beta.ndim == 2 else beta)

    cosine = np.clip(np.einsum("ij,ij->i", directions, apparent), -1.0, 1.0)
    out[valid] = np.arccos(cosine)
    return out
