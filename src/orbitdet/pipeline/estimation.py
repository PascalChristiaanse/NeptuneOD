"""Stage: perform the least-squares estimation.

This is the pipeline's inflection point. It consumes the weighted collection and
produces an ``EstimationOutput`` that everything downstream reads:

``estimation_output.tudat``
    The full ``EstimationOutput``, serialized with its native
    ``save_to_binary``. This single file carries the estimated parameters, the
    covariance, the residual history per iteration, the design matrices, and --
    crucially -- the covariance-propagation payload (see the note below).
``estimation_log.log``
    tudatpy's per-iteration terminal output, redirected to a file.
``final_parameters.json``
    The best-iteration parameters, so the final-propagation stage does not need
    tudatpy in order to find them.

The covariance-propagation payload
----------------------------------
Covariance propagation needs a
``CombinedStateTransitionAndSensitivityMatrixInterface``, which is normally
obtained from a live ``Estimator``. That object has no serialization of its
own, so the covariance stage could not run in a fresh process.

tudatpy solves this by storing the variational results and the interface
metadata *inside* ``EstimationOutput`` when the estimation is run with
``save_state_history_per_iteration=True``. The covariance stage then rebuilds
the interface from the file via
``create_state_transition_interface_from_estimation_output``. That flag is
therefore **required** for this pipeline, and the serialized output is what
makes the covariance stage a true file boundary rather than part of the
estimation process. Only single-arc estimation stores the payload.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tudatpy.estimation import estimation_analysis as est_an
from tudatpy.util import redirect_std

from orbitdet.estimation import get_apriori_covariance_matrix, get_estimatable_parameters
from orbitdet.pipeline.artifacts import load_collection, tudat_stem
from orbitdet.pipeline.dynamics import build_propagator
from orbitdet.pipeline.setup import prepare_environment

logger = logging.getLogger(__name__)


def run_estimation(
    cfg: DictConfig,
    observations_path: str | Path,
    output_dir: str | Path,
    output_name: str = "estimation_output.tudat",
) -> tuple[Path, Path]:
    """Run the least-squares estimation and persist its output.

    Parameters
    ----------
    cfg:
        Resolved configuration with ``estimation`` and ``dynamics`` sections.
    observations_path:
        Weighted ``ObservationCollection``.
    output_dir:
        Directory this stage writes into.
    output_name:
        Filename for the serialized ``EstimationOutput``.

    Returns
    -------
    tuple[Path, Path]
        ``(estimation_output_path, summary_path)``.
    """
    output_dir = Path(output_dir)
    observations = load_collection(observations_path)
    ctx, bodies = prepare_environment(cfg, output_dir, observations)

    propagator_settings, observation_models = build_propagator(cfg, ctx, bodies, observations)

    parameter_set = get_estimatable_parameters(cfg, ctx, propagator_settings, bodies)
    logger.info("Initial parameter set: %s", parameter_set.parameter_vector)

    estimator = est_an.Estimator(
        bodies,
        parameter_set,
        observation_models,
        propagator_settings,
        False,
    )

    max_iterations = cfg.estimation.max_iterations
    max_iterations_without_improvement = cfg.estimation.get(
        "max_iterations_without_improvement", max_iterations
    )
    convergence_settings = est_an.estimation_convergence_checker(
        maximum_iterations=max_iterations,
        number_of_iterations_without_improvement=max_iterations_without_improvement,
    )
    logger.info(
        "Convergence settings: max_iterations=%d, max_iterations_without_improvement=%d",
        max_iterations,
        max_iterations_without_improvement,
    )

    inverse_apriori_covariance = get_apriori_covariance_matrix(cfg, parameter_set)
    apply_apriori_parameter_deviation = bool(
        cfg.estimation.get("apply_apriori_parameter_deviation", False)
    )

    if inverse_apriori_covariance is not None:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            inverse_apriori_covariance=inverse_apriori_covariance,
            convergence_checker=convergence_settings,
            apply_apriori_parameter_deviation=apply_apriori_parameter_deviation,
        )
        logger.info(
            "Estimation input created with inverse a priori covariance of shape %s "
            "(apply_apriori_parameter_deviation=%s)",
            inverse_apriori_covariance.shape,
            apply_apriori_parameter_deviation,
        )
    else:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            convergence_checker=convergence_settings,
        )
        logger.info("Estimation input created without inverse a priori covariance matrix.")

    # ``save_state_history_per_iteration`` MUST be True: it is what makes
    # tudatpy store the covariance-propagation payload in the output. Without
    # it the covariance stage cannot rebuild the state-transition interface and
    # fails with "no covariance-propagation data is stored".
    estimation_input.define_estimation_settings(
        save_state_history_per_iteration=True,
        save_residuals_and_parameters_per_iteration=True,
    )

    log_path = output_dir / "estimation_log.log"
    logger.info("Starting estimation (log: %s)...", log_path.name)
    start = time.perf_counter()
    with redirect_std(str(log_path)):
        estimation_output = estimator.perform_estimation(estimation_input)
    duration = time.perf_counter() - start
    logger.info("Estimation completed in %.2f seconds.", duration)

    output_path = Path(tudat_stem(output_dir / output_name) + ".tudat")
    estimation_output.save_to_binary(tudat_stem(output_dir / output_name))
    if not output_path.exists():
        raise RuntimeError(
            f"Expected {output_path} after save_to_binary, but it was not created."
        )
    logger.info("Estimation output saved to %s", output_path.name)

    if not estimation_output.has_covariance_propagation_data:
        raise RuntimeError(
            "EstimationOutput has no covariance-propagation data. The covariance "
            "stage cannot run. This indicates save_state_history_per_iteration did "
            "not take effect, or the estimation is not single-arc."
        )

    _warn_if_covariance_is_degenerate(estimation_output)

    summary_path = _write_summary(estimation_output, output_dir, duration)
    return output_path, summary_path


def _warn_if_covariance_is_degenerate(estimation_output) -> None:
    """Warn loudly when the estimated covariance is not finite.

    A singular normal matrix does not make tudatpy raise: the inversion simply
    produces NaN, which then propagates into every formal-error figure and into
    the covariance summary. Without this warning the pipeline would report
    success and quietly emit NaN plots.

    Singularity is usually a *configuration* problem rather than a code one --
    more parameters than the observations can constrain, or two parameters that
    are perfectly correlated. This is a warning, not an error, because the
    estimation output is still useful for inspecting the fit itself.
    """
    covariance = np.asarray(estimation_output.covariance)
    if np.isfinite(covariance).all():
        return

    inverse_covariance = np.asarray(estimation_output.inverse_covariance)
    rank = int(np.linalg.matrix_rank(inverse_covariance))
    size = inverse_covariance.shape[0]

    logger.warning(
        "Estimated covariance contains %d non-finite entries; the covariance is "
        "degenerate. The normal matrix has rank %d of %d, so at least one "
        "estimated parameter is not constrained by these observations. "
        "Downstream formal-error figures will contain NaN. Consider removing "
        "parameters from estimation.parameters_to_estimate, or constraining "
        "them with an a priori covariance.",
        int((~np.isfinite(covariance)).sum()),
        rank,
        size,
    )


def _write_summary(estimation_output, output_dir: Path, duration: float) -> Path:
    """Write the post-fit summary metrics and best-iteration parameters as JSON.

    Keeping these as a small JSON file means downstream figure and reporting
    steps can read a summary without loading the (large) ``EstimationOutput``.
    """
    final_residuals = np.asarray(estimation_output.final_residuals)
    n_iterations = int(estimation_output.residual_history.shape[1])

    per_iteration_rms = [
        float(np.sqrt(np.mean(np.square(np.asarray(estimation_output.residual_history)[:, i]))))
        for i in range(n_iterations)
    ]

    best_iteration = int(estimation_output.best_iteration)
    parameters = np.asarray(estimation_output.parameter_history)[:, best_iteration]

    summary = {
        "estimation_duration_s": float(duration),
        "num_iterations": n_iterations,
        "best_iteration": best_iteration,
        "num_observations": int(final_residuals.size),
        # The size of the estimated parameter vector. A sweep varies this
        # directly, so it travels in the summary rather than being recomputed
        # by whatever reads the table.
        "num_parameters": int(len(parameters)),
        "final_parameters": [float(v) for v in parameters],
        "per_iteration_rms": per_iteration_rms,
        "residuals_rms": float(np.sqrt(np.mean(np.square(final_residuals)))),
        "residuals_mean": float(np.mean(final_residuals)),
        "residuals_std": float(np.std(final_residuals)),
        "residuals_max_abs": float(np.abs(final_residuals).max()),
        "postfit_rms_best_iteration": per_iteration_rms[best_iteration],
        "has_covariance_propagation_data": bool(
            estimation_output.has_covariance_propagation_data
        ),
        # Recorded so a report can state whether the formal errors are usable
        # rather than presenting NaN as a result.
        "covariance_is_finite": bool(np.isfinite(np.asarray(estimation_output.covariance)).all()),
    }

    path = output_dir / "estimation_summary.json"
    path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Estimation summary: RMS=%.6e, best iteration=%d, iterations=%d",
        summary["residuals_rms"],
        best_iteration,
        n_iterations,
    )
    return path
