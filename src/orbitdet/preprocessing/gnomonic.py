"""Gnomonic (tangent-plane) projection and its inverse.

A tangent-plane image records a satellite's offset from the planet as two
``X`` / ``Y`` standard coordinates: arcseconds along increasing right ascension
and along increasing declination, measured on a *plane* tangent to the celestial
sphere at the planet's position.  The observation model measures an *angle* from
the observatory, so the offsets must be projected back onto the sphere before
they mean anything geometrically.

This is the pure half of that work: spherical trigonometry with no ephemerides,
no Tudat and no config.  Supplying the tangent point - the planet's topocentric
direction, which needs the system of bodies - is the caller's job.

Two conventions are easy to confuse, so they are named explicitly:

``standard -> relative``
    Plane offsets to a direction on the sphere, expressed as offsets from the
    tangent point.  This is the direction the ingestion path needs: a raw file
    holds X/Y, and the observable wants a relative direction.
``relative -> standard``
    The inverse, for reproducing or validating a source's own reduction.

References:
    Montenbruck & Gill, *Satellite Orbits: Models, Methods and Applications*,
    section on the gnomonic projection.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "standard_to_relative_radec",
    "relative_to_standard_radec",
    "apply_position_angle_offset",
]


def standard_to_relative_radec(
    xi_rad: np.ndarray | float,
    eta_rad: np.ndarray | float,
    ra0_rad: np.ndarray | float,
    dec0_rad: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Project gnomonic standard coordinates into relative RA/Dec offsets.

    Converts plane offsets :math:`(\xi, \eta)` about a tangent point
    :math:`(\alpha_0, \delta_0)` into the exact spherical offsets
    :math:`(\Delta\alpha, \Delta\delta)` they represent:

    .. math::

        \Delta\alpha &= \arctan\!\left(
            \frac{\xi}{\cos\delta_0 - \eta \sin\delta_0}\right) \\
        \sin\delta &= \frac{\sin\delta_0 + \eta \cos\delta_0}
            {\sqrt{1 + \xi^2 + \eta^2}}

    The denominators are why this cannot be approximated by returning
    :math:`(\xi, \eta)` directly: the projection stretches with distance from the
    tangent point, and the :math:`\cos\delta_0` term makes a given plane offset
    correspond to a larger angular offset at high declination.

    Args:
        xi_rad: Standard X coordinate (along increasing right ascension), radians.
        eta_rad: Standard Y coordinate (along increasing declination), radians.
        ra0_rad: Right ascension of the tangent point, radians.
        dec0_rad: Declination of the tangent point, radians.

    Returns:
        Tuple ``(delta_ra, delta_dec)`` in radians, shaped like the input.  The
        RA offset is returned through ``arctan`` rather than wrapped, so it stays
        in ``(-pi/2, pi/2)`` - the whole point of standard coordinates is that a
        tangent plane cannot represent a half-sky, and a value beyond that range
        means the input was not a tangent-plane offset.

    Examples:
        Zero offset is the tangent point itself:

        >>> import numpy as np
        >>> dra, ddec = standard_to_relative_radec(0.0, 0.0, 1.0, 0.5)
        >>> bool(np.isclose(dra, 0.0)) and bool(np.isclose(ddec, 0.0))
        True

        A small offset is recovered nearly unchanged near the equator:

        >>> import numpy as np
        >>> dra, _ = standard_to_relative_radec(1e-6, 0.0, 0.0, 0.0)
        >>> bool(np.isclose(dra, 1e-6))
        True
    """
    xi = np.asarray(xi_rad, dtype="float64")
    eta = np.asarray(eta_rad, dtype="float64")
    dec0 = np.asarray(dec0_rad, dtype="float64")

    cos_d0 = np.cos(dec0)
    sin_d0 = np.sin(dec0)

    delta_ra = np.arctan2(xi, cos_d0 - eta * sin_d0)
    sin_dec = (sin_d0 + eta * cos_d0) / np.sqrt(1.0 + xi**2 + eta**2)
    delta_dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0)) - dec0
    return delta_ra, delta_dec


def relative_to_standard_radec(
    delta_ra_rad: np.ndarray | float,
    delta_dec_rad: np.ndarray | float,
    ra0_rad: np.ndarray | float,
    dec0_rad: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Project relative RA/Dec offsets into gnomonic standard coordinates.

    The inverse of :func:`standard_to_relative_radec`, obtained by inverting the
    two relations directly:

    .. math::

        \xi &= \Delta\alpha \left(\cos\delta_0 - \eta \sin\delta_0\right) \\
        \eta &= \frac{\sqrt{1 + \Delta\alpha^2}\,\sin\Delta\delta - \sin\delta_0}
            {\cos\delta_0}

    applied in that order, with :math:`\Delta\delta` the declination offset.  It
    is written as an explicit inversion rather than a closed-form pair because a
    closed form invites an algebra slip that a round-trip test catches only by
    accident.

    Provided because reproducing a source's own plate reduction requires going
    the other way, and because the round trip is the cheapest check that the two
    directions agree.

    Args:
        delta_ra_rad: Right-ascension offset from the tangent point, radians.
        delta_dec_rad: Declination offset from the tangent point, radians.
        ra0_rad: Right ascension of the tangent point, radians.
        dec0_rad: Declination of the tangent point, radians.

    Returns:
        Tuple ``(xi_rad, eta_rad)`` in radians.
    """
    d_ra = np.asarray(delta_ra_rad, dtype="float64")
    d_dec = np.asarray(delta_dec_rad, dtype="float64")
    dec0 = np.asarray(dec0_rad, dtype="float64")

    cos_d0 = np.cos(dec0)
    sin_d0 = np.sin(dec0)
    sin_dec = np.sin(dec0 + d_dec)

    eta = (np.sqrt(1.0 + d_ra**2) * sin_dec - sin_d0) / cos_d0
    xi = d_ra * (cos_d0 - eta * sin_d0)
    return xi, eta


def apply_position_angle_offset(
    position_angle_rad: np.ndarray | float,
    delta_ra_rad: np.ndarray | float,
    delta_dec_rad: np.ndarray | float,
) -> np.ndarray:
    r"""Shift a position angle by a reference-frame rotation.

    A position angle is measured from the frame's north pole through east, so it
    is *not* invariant under a change of frame: the pole moves, and with it the
    zero point of the angle.  The shift is

    .. math::

        \theta' = \theta - \Delta\theta, \qquad
        \Delta\theta = \operatorname{atan2}(\Delta\alpha, \Delta\delta)

    where :math:`(\Delta\alpha, \Delta\delta)` is the offset of the *new* pole's
    direction from the old one, measured at the observed position.  When the two
    frames share a pole - which is the case for any pure precession between
    equinoxes of the same type - the offset is zero and the angle is unchanged.

    Args:
        position_angle_rad: Position angle from north through east, radians.
        delta_ra_rad: Right-ascension component of the pole offset, radians.
        delta_dec_rad: Declination component of the pole offset, radians.

    Returns:
        The shifted position angle, wrapped into ``[-pi, pi)``.

    Examples:
        A zero pole offset leaves the angle alone:

        >>> import numpy as np
        >>> float(apply_position_angle_offset(0.3, 0.0, 0.0))
        0.3
    """
    angle = np.asarray(position_angle_rad, dtype="float64")
    shift = np.arctan2(
        np.asarray(delta_ra_rad, dtype="float64"),
        np.asarray(delta_dec_rad, dtype="float64"),
    )
    return (angle - shift + np.pi) % (2.0 * np.pi) - np.pi
