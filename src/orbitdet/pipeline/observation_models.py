"""Rebuilding observation models from a serialized collection.

``ObservationModelSettings`` has no public constructor, so settings cannot be
rebuilt by instantiating a class. Each observable type must go through its
factory function instead. Only the types this project actually produces are
mapped; an unmapped type raises rather than silently dropping observations.

This is needed because the ``ObservationCollection`` preserves each observable
type and its link definitions, but not the settings objects themselves. A stage
that only has the serialized collection can still rebuild the simulators by
walking that preserved structure.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observable_models_setup import model_settings as ms

logger = logging.getLogger(__name__)

#: Observable type -> factory function producing its ``ObservationModelSettings``.
_OBSERVABLE_FACTORIES = {
    ms.ObservableType.one_way_range_type: ms.one_way_range,
    ms.ObservableType.n_way_range_type: ms.n_way_range,
    ms.ObservableType.angular_position_type: ms.angular_position,
    ms.ObservableType.azimuth_elevation_type: ms.azimuth_elevation,
    ms.ObservableType.relative_angular_position_type: ms.relative_angular_position,
    ms.ObservableType.position_observable_type: ms.cartesian_position,
    ms.ObservableType.velocity_observable_type: ms.cartesian_velocity,
    ms.ObservableType.relative_position_observable_type: ms.relative_cartesian_position,
    ms.ObservableType.one_way_instantaneous_doppler_type: ms.one_way_doppler_instantaneous,
    ms.ObservableType.one_way_averaged_doppler_type: ms.one_way_doppler_averaged,
    ms.ObservableType.two_way_instantaneous_doppler_type: ms.two_way_doppler_instantaneous,
    ms.ObservableType.n_way_averaged_doppler_type: ms.n_way_doppler_averaged,
    ms.ObservableType.dsn_n_way_range_type: ms.dsn_n_way_range,
    ms.ObservableType.dsn_n_way_averaged_doppler_type: ms.dsn_n_way_doppler_averaged,
    ms.ObservableType.doppler_measured_frequency_type: ms.doppler_measured_frequency,
    ms.ObservableType.differenced_time_of_arrival_type: ms.differenced_time_of_arrival,
    ms.ObservableType.differenced_frequency_of_arrival_type: ms.differenced_frequency_of_arrival,
    ms.ObservableType.pixel_coordinates_type: ms.pixel_coordinates,
    ms.ObservableType.position_angle_type: ms.position_angle,
    ms.ObservableType.separation_distance_type: ms.separation_distance,
    ms.ObservableType.position_angle_and_separation_type: ms.position_angle_and_separation,
}


def rebuild_observation_models(observations: obs.ObservationCollection) -> list:
    """Reconstruct observation model settings from a loaded collection.

    Parameters
    ----------
    observations:
        The collection to rebuild from. Its
        ``link_definitions_per_observable`` survives serialization, which is all
        ``create_observation_simulators`` needs.

    Returns
    -------
    list
        One ``ObservationModelSettings`` per (observable type, link definition)
        pair, in the order ``create_observation_simulators`` expects.

    Raises
    ------
    NotImplementedError
        If the collection contains an observable type with no registered
        factory. Failing loudly is deliberate: silently skipping a type would
        drop observations from the fit without any visible error.
    """
    models = []
    for observable_type, link_definitions in observations.link_definitions_per_observable.items():
        factory = _OBSERVABLE_FACTORIES.get(observable_type)
        if factory is None:
            raise NotImplementedError(
                f"No factory registered for observable type {observable_type!r}. "
                "Add it to _OBSERVABLE_FACTORIES in orbitdet/pipeline/observations.py."
            )
        for link_definition in link_definitions:
            models.append(factory(link_definition))
    logger.info("Reconstructed %d observation model setting(s).", len(models))
    return models


def collect_observatory_codes(observations: obs.ObservationCollection) -> set[str]:
    """Find every ground-station code referenced by the collection's link ends.

    Building the observation collection adds ground stations to the system of
    bodies as a *side effect*. A stage that rebuilds the environment from config
    alone therefore lacks them, and simulator construction fails with
    "point is not a ground station". This recovers the codes so they can be
    re-added.
    """
    codes: set[str] = set()
    for link_definitions in observations.link_definitions_per_observable.values():
        for link_definition in link_definitions:
            for link_end_id in link_definition.link_ends.values():
                station = link_end_id.station_name
                if station:
                    codes.add(str(station))
    return codes


def add_observatories(cfg: DictConfig, bodies, codes: set[str]) -> None:
    """Re-add ground stations to *bodies* (in place)."""
    from orbitdet.observations.helpers import add_observatory_to_SOB

    for code in sorted(codes):
        add_observatory_to_SOB(cfg, bodies, code)
    if codes:
        logger.info("Re-added %d ground station(s): %s", len(codes), sorted(codes))
