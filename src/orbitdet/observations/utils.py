"""Shared utility functions for the observations subsystem."""

from __future__ import annotations

import tudatpy.estimation.observations as obs


def get_set_identifier(observation_set: obs.SingleObservationSet) -> str:
    """Return a human-readable identifier for an observation set.

    The identifier is derived from the receiver link end:
    - Ground stations: the observatory code (e.g. ``"689"``).
    - Spacecraft: the body name (e.g. ``"Voyager 2"``).
    - Geocentric (no reference point): ``"Geocentric"``.

    Parameters
    ----------
    observation_set : SingleObservationSet
        The observation set to identify.

    Returns
    -------
    str
        A string identifier for the set.
    """
    from tudatpy.estimation.observable_models_setup import links

    link_ends = observation_set.link_definition.link_ends
    receiver = link_ends.get(links.receiver)
    if receiver is None:
        return str(observation_set)

    reference_point = receiver.reference_point
    if reference_point == "":
        return receiver.body_name
    try:
        code = int(reference_point)
        if code < 0:
            return receiver.body_name
    except (ValueError, TypeError):
        pass
    return reference_point