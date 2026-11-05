"""Astrometric corrections that shift a measured direction to a true one.

Photocenter offset
    A resolved body's detected photocenter is offset from its centre of mass
    along the Sun-target-observer direction, by a fraction of the body radius
    that depends on the solar phase angle.  For a body a few thousand kilometres
    across observed at a range of a few AU, the offset reaches mas to tens of
    mas - comparable to the Gaia measurement noise, so it is not negligible.

Relativistic light deflection
    Post-Newtonian light bending by the Sun (and, at smaller amplitude, the
    giant planets) shifts the apparent direction of a target.  Near the solar
    limb this reaches 1.75 arcsec; it falls off with the angular distance from
    the deflecting body, but stays measurable across the sky.

"""

from __future__ import annotations

from .light_deflection import relativistic_light_deflection
from .photocenter import photocenter_offset_spherical

__all__ = [
    "relativistic_light_deflection",
    "photocenter_offset_spherical",
]
