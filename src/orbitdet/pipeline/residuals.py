"""Stage: compute residuals against an ephemeris or the propagated trajectory.

Two callers use this stage:

* the **pre-fit, ephemeris** rule, which compares the observations against the
  SPICE ephemeris and is what outlier rejection and weighting need; and
* the **pre/post-fit, propagated** rules, which compare against a numerically
  propagated state so the fit's quality can be inspected.

Both write the residuals into the collection *and* flatten them into a tidy CSV.
The CSV is the important part: without it every residual figure needs a live
``ObservationCollection`` plus a furnished environment, so plotting cannot be
separated from computation. With it, the figure rules become pure functions of
a file and can run in parallel, be re-run independently, and be inspected
without tudat.

CSV schema (one row per scalar residual component)::

    epoch_sec_j2000, epoch_iso, set_index, dataset_id, observable_type,
    component, residual_rad, residual_arcsec,
    receiver_body, receiver_ref_point, transmitter_body

The link-end columns exist so a figure can label each series from the CSV alone.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig
from tudatpy.astro.time_representation import DateTime
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.pipeline.artifacts import load_collection, load_metadata, save_collection
from orbitdet.pipeline.observation_models import rebuild_observation_models
from orbitdet.pipeline.setup import prepare_environment

logger = logging.getLogger(__name__)

RAD_TO_ARCSEC = 180.0 / np.pi * 3600.0


def _observable_type_name(observable_type) -> str:
    """Human-readable name for an ``ObservableType`` enum value."""
    try:
        return observable_type.name
    except AttributeError:
        return str(observable_type)


def _dataset_id_for_index(metadata: dict, index: int) -> str:
    """Reverse-lookup the ``dataset_id`` for a set index from the metadata."""
    for metas in metadata.values():
        for meta in metas:
            if meta.get("index") == index:
                return str(meta.get("dataset_id", ""))
    return ""


def _link_end_labels(obs_set) -> tuple[str, str, str]:
    """Extract ``(receiver_body, receiver_ref_point, transmitter_body)``.

    The figure stage needs these to label each series. Persisting them in the
    CSV keeps that stage free of the collection and of tudat.
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


def write_residuals_csv(
    observations: obs.ObservationCollection,
    metadata: dict,
    csv_path: Path,
) -> int:
    """Flatten per-set residuals into a tidy CSV. Returns the row count."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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

        for index, obs_set in enumerate(observations.get_single_observation_sets()):
            residuals = np.asarray(obs_set.residuals)
            if residuals.size == 0:
                logger.debug("Skipping empty observation set %d", index)
                continue

            # Residuals are [n_times] for a scalar observable and
            # [n_times, n_components] otherwise. Normalize to 2-D so the loop
            # below is uniform.
            if residuals.ndim == 1:
                residuals = residuals.reshape(-1, 1)

            times = obs_set.observation_times
            dataset_id = _dataset_id_for_index(metadata, index)
            observable = _observable_type_name(obs_set.observable_type)
            receiver_body, receiver_ref_point, transmitter_body = _link_end_labels(obs_set)

            for i, epoch in enumerate(times):
                epoch_sec = float(epoch.to_float())
                epoch_iso = DateTime.from_epoch_time_object(epoch).to_iso_string()
                for component in range(residuals.shape[1]):
                    value = float(residuals[i, component])
                    writer.writerow(
                        [
                            f"{epoch_sec:.9f}",
                            epoch_iso,
                            index,
                            dataset_id,
                            observable,
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


def run_residuals_wrt_ephemeris(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    output_name: str = "observations_with_residuals.tudat",
) -> tuple[Path, Path]:
    """Compute residuals against the SPICE ephemeris.

    This is the reference for outlier rejection and weighting, which both
    compare observations against the ephemeris rather than a fitted trajectory.

    Parameters
    ----------
    cfg:
        Resolved configuration.
    observations_path:
        Serialized ``ObservationCollection``.
    metadata_path:
        ``dataset_metadata.json``.
    output_dir:
        Directory this stage writes into.
    output_name:
        Filename for the collection carrying the residuals.

    Returns
    -------
    tuple[Path, Path]
        ``(collection_path, residuals_csv_path)``.
    """
    output_dir = Path(output_dir)
    observations = load_collection(observations_path)
    _, bodies = prepare_environment(cfg, output_dir, observations)

    observation_models = rebuild_observation_models(observations)
    simulators = obs_sim_setup.create_observation_simulators(observation_models, bodies)
    obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)
    logger.info("Residuals computed wrt SPICE ephemeris.")

    collection_path = save_collection(observations, output_dir / output_name)
    metadata = load_metadata(metadata_path)
    csv_path = output_dir / "residuals.csv"
    n_rows = write_residuals_csv(observations, metadata, csv_path)
    logger.info("Wrote %d residual rows to %s", n_rows, csv_path.name)

    return collection_path, csv_path
