"""Stage: propagate the estimated covariance over the full time arc.

This stage is the reason the pipeline could be split at all. Propagating a
covariance needs a ``CombinedStateTransitionAndSensitivityMatrixInterface``,
which has no serialization of its own -- normally it can only be obtained from
a live ``Estimator``, which would have forced the covariance propagation to run
inside the estimation process.

tudatpy stores the variational results and interface metadata inside
``EstimationOutput`` when the estimation runs with
``save_state_history_per_iteration=True``. This stage rebuilds the interface
from that stored payload, in a fresh process, from the file alone::

    estimation_output.tudat
        +--> create_state_transition_interface_from_estimation_output()
        +--> propagate_covariance_from_estimation_output()

Two outputs:

``propagated_formal_errors.npz``
    Cartesian formal errors, and the same rotated into the RSW frame using the
    estimated state. Everything the formal-error and RSW-uncertainty figures
    need.
``covariance_summary.json``
    Peak formal errors, so a report can quote numbers without loading the archive.

Note on degenerate covariances
------------------------------
A singular normal matrix makes tudatpy's inversion return NaN rather than
raising, and those NaNs propagate into every downstream figure. This stage does
not treat that as an error -- the propagation itself succeeded, and the rest of
the run is still worth producing -- but it records
``formal_errors_are_finite`` in the summary and warns, so a NaN never gets
presented as a measurement.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tudatpy.astro import frame_conversion
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.pipeline.artifacts import load_collection, tudat_stem
from orbitdet.pipeline.dynamics import covariance_output_times, state_history_arrays
from orbitdet.pipeline.setup import prepare_environment

logger = logging.getLogger(__name__)


def run_covariance(
    cfg: DictConfig,
    observations_path: str | Path,
    estimation_path: str | Path,
    propagated_states_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Propagate the covariance and write formal errors.

    Parameters
    ----------
    cfg:
        Resolved configuration.
    observations_path:
        The weighted collection (used to rebuild the environment).
    estimation_path:
        Serialized ``EstimationOutput`` carrying the covariance-propagation payload.
    propagated_states_path:
        Propagation results from the post-fit stage. The estimated state is
        interpolated from here to define the RSW frame at each output epoch, so
        the rotation is consistent with the plotted trajectory.
    output_dir:
        Directory this stage writes into.

    Returns
    -------
    tuple[Path, Path]
        ``(npz_path, summary_path)``.
    """
    output_dir = Path(output_dir)
    observations = load_collection(observations_path)
    ctx, bodies = prepare_environment(cfg, output_dir, observations)

    estimation_output = est_an.EstimationOutput.load_from_binary(
        tudat_stem(estimation_path)
    )
    if not estimation_output.has_covariance_propagation_data:
        raise RuntimeError(
            f"{Path(estimation_path).name} carries no covariance-propagation data. "
            "Re-run the estimation stage: it must set "
            "save_state_history_per_iteration=True, and only single-arc estimation "
            "stores the payload."
        )
    logger.info("Loaded estimation output with covariance-propagation data.")

    output_times = covariance_output_times(cfg, ctx)
    logger.info(
        "Propagating covariance at %d epoch(s) from the initial epoch.", len(output_times)
    )

    # Propagate from the stored payload. No Estimator object is involved, which
    # is what lets this run as a separate process: the helper rebuilds the
    # state-transition interface internally from what the estimation saved.
    propagated_covariances = est_an.propagate_covariance_from_estimation_output(
        estimation_output, output_times
    )

    epochs = np.array(list(propagated_covariances.keys()))

    # Formal errors come from sqrt(diag(P)) rather than a separate call to
    # ``propagate_formal_errors``. The two are mathematically identical, but
    # deriving them from the same propagated covariance guarantees consistency:
    # ``propagate_formal_errors`` takes the raw ``estimation_output.covariance``
    # and would ignore any consider-parameter contribution that
    # ``propagate_covariance_from_estimation_output`` folds in.
    formal_errors = np.array(
        [np.sqrt(np.diag(propagated_covariances[epoch])) for epoch in epochs]
    )

    if not np.isfinite(formal_errors).all():
        logger.warning(
            "%d of %d propagated formal errors are non-finite. The estimated "
            "covariance is degenerate (see the estimation stage's warning); the "
            "formal-error figures will contain NaN.",
            int((~np.isfinite(formal_errors)).sum()),
            formal_errors.size,
        )

    # Rotate into RSW using the *estimated* state, so the frame matches the
    # trajectory the covariance belongs to.
    est_states_interp = _interpolate_estimated_state(propagated_states_path, epochs)
    fe_rsw = _rotate_to_rsw(epochs, est_states_interp, propagated_covariances)
    rsw_distance = _rsw_distance(epochs, est_states_interp, bodies)

    npz_path = output_dir / "propagated_formal_errors.npz"
    np.savez(
        npz_path,
        epochs=epochs,
        formal_errors=formal_errors,
        formal_errors_rsw=fe_rsw,
        rsw_distance=rsw_distance,
    )
    logger.info("Formal errors written to %s", npz_path.name)

    summary = {
        "num_epochs": int(len(epochs)),
        "epoch_start_s": float(epochs[0]),
        "epoch_end_s": float(epochs[-1]),
        "max_position_sigma_m": float(np.max(fe_rsw[:, :3])),
        "max_velocity_sigma_m_s": float(np.max(fe_rsw[:, 3:])),
        "median_position_sigma_m": float(np.median(fe_rsw[:, :3])),
        "median_velocity_sigma_m_s": float(np.median(fe_rsw[:, 3:])),
        # Stated explicitly so a report can say "not available" instead of
        # printing NaN as a measurement.
        "formal_errors_are_finite": bool(np.isfinite(formal_errors).all()),
    }
    summary_path = output_dir / "covariance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Peak position sigma %.3e m, peak velocity sigma %.3e m/s",
        summary["max_position_sigma_m"],
        summary["max_velocity_sigma_m_s"],
    )
    return npz_path, summary_path


def _interpolate_estimated_state(propagated_states_path: str | Path, epochs: np.ndarray):
    """Linearly interpolate the propagated state onto the covariance epochs."""
    from orbitdet.pipeline.artifacts import load_propagation_results

    results = load_propagation_results(propagated_states_path)
    est_epochs, est_states = state_history_arrays(results)
    return np.column_stack(
        [np.interp(epochs, est_epochs, est_states[:, i]) for i in range(est_states.shape[1])]
    )


def _rotate_to_rsw(epochs: np.ndarray, states: np.ndarray, covariances: dict) -> np.ndarray:
    """Rotate each Cartesian covariance into the RSW frame and take sqrt(diag).

    ``propagate_formal_errors`` returns standard deviations (not variances), so
    the RSW values must come from ``sqrt(diag(cov_rsw))`` rather than from
    rotating the formal-error vector.
    """
    out = np.zeros((len(epochs), 6))
    for i, epoch in enumerate(epochs):
        rotation = frame_conversion.inertial_to_rsw_rotation_matrix(states[i])
        full = np.block(
            [
                [rotation, np.zeros((3, 3))],
                [np.zeros((3, 3)), rotation],
            ]
        )
        covariance = covariances[epoch] if epoch in covariances else covariances[float(epoch)]
        out[i] = np.sqrt(np.diag(full @ covariance @ full.T))
    return out


def _rsw_distance(epochs: np.ndarray, states: np.ndarray, bodies) -> np.ndarray:
    """RSW-frame offset between the ephemeris and the estimated trajectory."""
    out = np.zeros((len(epochs), 3))
    ephemeris = bodies.get("Triton Spice").ephemeris
    for i, epoch in enumerate(epochs):
        spice_state = ephemeris.cartesian_state(epoch)
        relative = spice_state[:3] - states[i][:3]
        rotation = frame_conversion.inertial_to_rsw_rotation_matrix(states[i])
        out[i] = rotation @ relative
    return out
