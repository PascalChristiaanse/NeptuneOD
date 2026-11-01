"""Stage: propagate the trajectory and compute residuals against it.

Used twice in the DAG:

**pre-fit** (``prefit_propagation``)
    Propagates from the configured initial state -- the SPICE ephemeris state at
    ``initial_epoch`` -- and compares the observations against it. This is the
    "what the fit started from" reference.

**post-fit** (``postfit_propagation``)
    Propagates from the *estimated* best-iteration parameters and compares again.
    This is the "what the fit achieved" result.

Both write the propagation results and a residuals CSV, so every trajectory
figure is a pure function of files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig

from orbitdet.pipeline.artifacts import load_collection, load_metadata, save_collection
from orbitdet.pipeline.dynamics import build_dynamics, build_observation_simulators, propagate
from orbitdet.pipeline.residuals import write_residuals_csv
from orbitdet.pipeline.setup import prepare_environment

logger = logging.getLogger(__name__)


def _residuals_and_csv(
    observations,
    bodies,
    metadata: dict,
    output_dir: Path,
    collection_name: str,
    csv_name: str,
) -> tuple[Path, Path]:
    """Compute residuals against the propagated state and write both artifacts."""
    simulators = build_observation_simulators(bodies, observations)
    obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)

    collection_path = save_collection(observations, output_dir / collection_name)
    csv_path = output_dir / csv_name
    n_rows = write_residuals_csv(observations, metadata, csv_path)
    logger.info("Wrote %d residual rows to %s", n_rows, csv_name)
    return collection_path, csv_path


def load_best_parameters(output_dir: str | Path) -> np.ndarray:
    """Read the best-iteration parameters written by the estimation stage.

    Kept in JSON rather than recovered from the serialized ``EstimationOutput``
    so the post-fit stage does not need to deserialize the (large) output.
    """
    path = Path(output_dir) / "estimation_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No estimation summary at {path}; the estimation stage must run first."
        )
    summary = json.loads(path.read_text())
    return np.asarray(summary["final_parameters"], dtype=float)


def run_propagation(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    initial_states: np.ndarray | None = None,
    tag: str = "prefit",
) -> tuple[Path, Path, Path]:
    """Propagate the trajectory and compute residuals against it.

    Parameters
    ----------
    cfg:
        Resolved configuration.
    observations_path:
        ``ObservationCollection`` to compute residuals for.
    metadata_path:
        ``dataset_metadata.json``.
    output_dir:
        Directory this stage writes into.
    initial_states:
        Initial state to propagate from. When ``None`` the propagator uses the
        state it was built with (the configured initial estimate).
    tag:
        Prefix for the output filenames, e.g. ``"prefit"`` or ``"postfit"``.

    Returns
    -------
    tuple[Path, Path, Path]
        ``(propagation_results_path, collection_path, residuals_csv_path)``.
    """
    output_dir = Path(output_dir)
    observations = load_collection(observations_path)
    ctx, bodies = prepare_environment(cfg, output_dir, observations)

    propagator_settings, _ = build_dynamics(cfg, ctx, bodies)
    simulator = propagate(cfg, ctx, bodies, propagator_settings, initial_states)

    results_path = output_dir / f"{tag}_propagation_result.tudat"
    simulator.propagation_results.save_to_binary(str(results_path.with_suffix("")))
    if not results_path.exists():
        raise RuntimeError(f"Expected {results_path} after save_to_binary, but it was not created.")
    logger.info("Propagation results saved to %s", results_path.name)

    metadata = load_metadata(metadata_path)
    collection_path, csv_path = _residuals_and_csv(
        observations,
        bodies,
        metadata,
        output_dir,
        collection_name=f"{tag}_observations_with_residuals.tudat",
        csv_name=f"{tag}_residuals.csv",
    )

    return results_path, collection_path, csv_path


def run_postfit_propagation(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    estimation_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Propagate from the estimated parameters and compute post-fit residuals."""
    initial_states = load_best_parameters(estimation_dir)
    logger.info("Propagating from estimated state (%d values).", len(initial_states))
    return run_propagation(
        cfg,
        observations_path,
        metadata_path,
        output_dir,
        initial_states=initial_states,
        tag="postfit",
    )
