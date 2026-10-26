"""Stage: pre-fit residuals with respect to the SPICE ephemeris.

Computes residuals for every observation against the ephemeris-based
observation simulators, then persists them as a tidy CSV.

The CSV is the important part. Without it, every residual figure needs a live
``ObservationCollection`` plus a furnished environment, so plotting cannot be
separated from computation. With it, the figure rules become pure functions of
a file and can run in parallel, be re-run independently, and be inspected
without tudat.

CSV schema (one row per scalar residual component)::

    epoch_sec_j2000, epoch_iso, set_index, dataset_id, observable_type,
    component, residual_rad, residual_arcsec,
    receiver_body, receiver_ref_point, transmitter_body

The link-end columns exist so the figure stage can label each series without
loading the collection or importing tudat.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig
from tudatpy.astro.time_representation import DateTime
from tudatpy.estimation.observable_models_setup import model_settings as ms
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.pipeline.context import build_context
from orbitdet.pipeline.setup import furnish_kernels
from orbitdet.simulation import get_environment

logger = logging.getLogger(__name__)

RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0

# ``ObservationModelSettings`` has no public constructor, so settings cannot be
# rebuilt by instantiating the class. Each observable type must go through its
# factory function instead. Only the types this project actually produces are
# mapped; an unmapped type raises rather than silently dropping observations.
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

    The collection preserves each observable type and its link definitions, but
    not the settings objects themselves. This walks the preserved structure and
    re-creates the settings through the appropriate factory function.

    Returns
    -------
    list
        One ``ObservationModelSettings`` per (observable type, link definition)
        pair, in the order ``create_observation_simulators`` expects.
    """
    models = []
    for observable_type, link_definitions in observations.link_definitions_per_observable.items():
        factory = _OBSERVABLE_FACTORIES.get(observable_type)
        if factory is None:
            raise NotImplementedError(
                f"No factory registered for observable type {observable_type!r}. "
                "Add it to _OBSERVABLE_FACTORIES in orbitdet/pipeline/prefit.py."
            )
        for link_definition in link_definitions:
            models.append(factory(link_definition))
    return models


def collect_observatory_codes(observations: obs.ObservationCollection) -> set[str]:
    """Find every ground-station code referenced by the collection's link ends.

    Building the observation collection adds ground stations to the system of
    bodies as a side effect. A stage that rebuilds the environment from config
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


def _observable_type_name(observable_type) -> str:
    """Human-readable name for an ``ObservableType`` enum value."""
    try:
        return observable_type.name
    except AttributeError:
        return str(observable_type)


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _dataset_id_for_index(metadata: dict, index: int) -> str:
    """Reverse-lookup the ``dataset_id`` for a set index from the metadata."""
    for metas in metadata.values():
        for meta in metas:
            if meta.get("index") == index:
                return str(meta.get("dataset_id", ""))
    return ""


def run_prefit_spice(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Compute pre-fit residuals wrt SPICE and write them to CSV.

    Parameters
    ----------
    cfg:
        Resolved configuration.
    observations_path:
        Serialized ``ObservationCollection`` from the observations stage.
    metadata_path:
        ``dataset_metadata.json`` from the observations stage.
    output_dir:
        Directory for the CSV output.

    Returns
    -------
    Path
        The written ``residuals.csv``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(cfg, output_dir)

    # Each rule is a separate process, so the SPICE pool starts empty.
    furnish_kernels(cfg)

    bodies = get_environment(cfg, ctx)

    observations = obs.ObservationCollection.load_from_binary(
        str(Path(observations_path).with_suffix(""))
    )
    logger.info(
        "Loaded %d observations from %s",
        len(observations.concatenated_times),
        Path(observations_path).name,
    )

    # Rebuild the observation model settings from the collection itself. The
    # link definitions and observable types survive serialization, which is all
    # create_observation_simulators needs.
    observation_models = rebuild_observation_models(observations)
    logger.info("Reconstructed %d observation model setting(s).", len(observation_models))

    # Ground stations are added to the system of bodies as a side effect of
    # building the collection, so a freshly-built environment lacks them.
    add_observatories(cfg, bodies, collect_observatory_codes(observations))

    simulators = obs_sim_setup.create_observation_simulators(observation_models, bodies)
    obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)
    logger.info("Pre-fit residuals computed wrt SPICE.")

    metadata = _load_metadata(Path(metadata_path))
    csv_path = output_dir / "residuals.csv"
    n_rows = _write_residuals_csv(observations, metadata, csv_path)
    logger.info("Wrote %d residual rows to %s", n_rows, csv_path.name)

    return csv_path


def _link_end_labels(obs_set) -> tuple[str, str, str]:
    """Extract ``(receiver_body, receiver_ref_point, transmitter_body)``.

    The figure stage needs these to label each series. Persisting them in the
    CSV keeps that stage free of any tudat dependency.
    """
    from tudatpy.estimation.observable_models_setup import links

    link_ends = obs_set.link_definition.link_ends
    receiver = link_ends.get(links.receiver)
    transmitter = link_ends.get(links.transmitter)
    return (
        getattr(receiver, "body_name", "") or "",
        getattr(receiver, "reference_point", "") or "",
        getattr(transmitter, "body_name", "") or "",
    )


def _write_residuals_csv(
    observations: obs.ObservationCollection,
    metadata: dict,
    csv_path: Path,
) -> int:
    """Flatten per-set residuals into a tidy CSV. Returns the row count."""
    n_rows = 0
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch_sec_j2000",
                "epoch_iso",
                "set_index",
                "dataset_id",
                "observable_type",
                "component",
                "residual_rad",
                "residual_arcsec",
                "receiver_body",
                "receiver_ref_point",
                "transmitter_body",
            ]
        )

        for set_index, obs_set in enumerate(observations.get_single_observation_sets()):
            if len(obs_set.observation_times) == 0:
                continue

            residuals = np.atleast_2d(np.array(obs_set.residuals))
            # ``residuals`` is (n_epochs, n_components); transpose so we can
            # iterate per epoch regardless of component count.
            if residuals.shape[0] != len(obs_set.observation_times):
                residuals = residuals.T

            dataset_id = _dataset_id_for_index(metadata, set_index)
            type_name = _observable_type_name(obs_set.observable_type)
            receiver_body, receiver_ref_point, transmitter_body = _link_end_labels(obs_set)

            for epoch_index, epoch in enumerate(obs_set.observation_times):
                epoch_sec = epoch.to_float()
                epoch_iso = DateTime.from_epoch_time_object(epoch).to_iso_string()
                for component, value in enumerate(residuals[epoch_index]):
                    writer.writerow(
                        [
                            f"{epoch_sec:.6f}",
                            epoch_iso,
                            set_index,
                            dataset_id,
                            type_name,
                            component,
                            f"{value:.12e}",
                            f"{value * RAD_TO_ARCSEC:.9e}",
                            receiver_body,
                            receiver_ref_point,
                            transmitter_body,
                        ]
                    )
                    n_rows += 1

    return n_rows
