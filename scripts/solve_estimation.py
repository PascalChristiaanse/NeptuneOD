"""Pipeline stage: solve the parameter estimation problem.

This stage does one thing: it solves the least-squares estimator problem for the
estimatable parameters named by the config and saves the resulting
:class:`~tudatpy.estimation.estimation_analysis.EstimationOutput`.  It computes
no residuals, applies no outlier rejection or weighting, propagates no
post-fit trajectory and draws no figures, so a change to a figure title or a
post-fit diagnostic never re-runs it.

The observations it consumes are the artifacts of the collection stage, not the
dataset config: ``observations`` and ``standard_frame`` name the
``observations.tudat`` / ``standard_frame.parquet`` pair written by
``scripts/generate_observation_collection.py`` (and carried through by
``scripts/prefit_residuals.py``).  The observation *models* are not serialized
with the collection, so they are rebuilt from it - which is what keeps this
stage independent of the collection config.

Runnable in three ways:

* standalone, with the default config::

      python scripts/solve_estimation.py

* standalone, against a different config::

      python scripts/solve_estimation.py --config-name=chapter_3/estimation_pole_models

* as a Snakemake rule, which additionally pins the input and output paths::

      python scripts/solve_estimation.py \\
          observations=<collection>/observations.tudat \\
          standard_frame=<collection>/standard_frame.parquet \\
          hydra.run.dir=results/<stage>

Artifacts written into the Hydra run directory:

``estimation_output.tudat``
    The :class:`~tudatpy.estimation.estimation_analysis.EstimationOutput`,
    carrying the estimated parameters, the covariance, the residual history and
    the per-iteration parameter history.  Every post-fit stage and every
    estimation figure consumes this.
``estimation_log.log``
    The estimator's own iteration-by-iteration output, captured verbatim.  It is
    the solver's log, not this stage's - this stage's logging goes to the
    console.

The stage is self-contained: it furnishes SPICE itself, because kernel state is
process-local and cannot be handed to another process.
"""

import logging
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.estimation import estimation_analysis as est_an
from tudatpy.estimation import observations as obs
from tudatpy.util import redirect_std

from orbitdet.data import KernelManager
from orbitdet.estimation import get_apriori_covariance_matrix, get_estimatable_parameters
from orbitdet.observations import rebuild_observation_models
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)
from orbitdet.utility import load_tudat_object, save_tudat_object

logger = logging.getLogger(__name__)


def _build_estimation_input(
    cfg: DictConfig,
    observations,
    parameter_set,
) -> est_an.EstimationInput:
    """Assemble the estimation input, with or without an a priori constraint.

    The a priori covariance is optional: ``get_apriori_covariance_matrix``
    returns ``None`` when the config constrains nothing, and in that case the
    estimator runs unconstrained and no ``inverse_apriori_covariance`` is set.
    Whether the a priori parameter deviations are applied is likewise a config
    choice, so it is read here rather than assumed.
    """
    convergence_settings = est_an.estimation_convergence_checker(
        maximum_iterations=cfg.estimation.max_iterations,
        number_of_iterations_without_improvement=cfg.estimation.get(
            "max_iterations_without_improvement", cfg.estimation.max_iterations
        ),
    )

    inverse_apriori_covariance = get_apriori_covariance_matrix(cfg, parameter_set)
    apply_deviation = bool(cfg.estimation.get("apply_apriori_parameter_deviation", False))

    if inverse_apriori_covariance is None:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            convergence_checker=convergence_settings,
        )
        logger.info("Estimation input created without an a priori covariance.")
    else:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            inverse_apriori_covariance=inverse_apriori_covariance,
            convergence_checker=convergence_settings,
            apply_apriori_parameter_deviation=apply_deviation,
        )
        logger.info(
            "Estimation input created with an a priori covariance of shape %s "
            "(apply_apriori_parameter_deviation=%s).",
            inverse_apriori_covariance.shape,
            apply_deviation,
        )

    # The per-iteration parameter and residual histories are what the estimation
    # figures and the best-iteration selection read, so they are always saved.
    # The state history is not: it is large, and nothing downstream consumes it.
    estimation_input.define_estimation_settings(
        save_state_history_per_iteration=True,
        save_residuals_and_parameters_per_iteration=True,
    )
    return estimation_input

def _build_dependent_variables(cfg: DictConfig) -> list:
    """Build every dependent-variable save setting, preserving config order."""
    entries = cfg.get("dependent_variables") or []
    return [_build_dependent_variable(entry) for entry in entries]

    
def _build_dependent_variable(entry: DictConfig):
    """Build one dependent-variable save setting from a config entry.

    The config names the variable with a ``type`` key and the bodies it refers
    to with ``body`` / ``central_body`` / ``secondary_body``.  Keeping the
    mapping here means the config stays declarative and the script stays free of
    hard-coded body names.
    """
    dependent_variable = prop_setup.dependent_variable
    variable_type = entry.type

    match variable_type:
        case "keplerian_state":
            return dependent_variable.keplerian_state(entry.body, entry.central_body)
        case "modified_equinoctial_state":
            return dependent_variable.modified_equinoctial_state(
                entry.body, entry.central_body
            )
        case "relative_position":
            return dependent_variable.relative_position(entry.body, entry.secondary_body)
        case "relative_velocity":
            return dependent_variable.relative_velocity(entry.body, entry.secondary_body)
        case _:
            raise ValueError(
                f"Unknown dependent variable type '{variable_type}'. Known: "
                "keplerian_state, modified_equinoctial_state, relative_position, "
                "relative_velocity."
            )

@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="solve_estimation",
)
@enforce_initialization
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    ctx: RuntimeContext = initialize(cfg)

    # Inject the epochs the propagator and the estimator need.
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

    # Kernel state is process-local, so every stage furnishes SPICE itself.
    logger.info("Loading kernels...")
    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    km.download_all_data_files()
    logger.info("Kernels loaded and furnished successfully.")

    bodies = get_environment(cfg, ctx)
    accelerations = get_dynamical_model(cfg, ctx, bodies)
    integrator = get_integrator_settings(cfg, ctx)
    dependent_variables_to_save = _build_dependent_variables(cfg)
    propagator_settings = get_propagator_settings(
        cfg, ctx, accelerations, integrator, dependent_variables_to_save=dependent_variables_to_save
    )
    logger.info("Environment and propagator settings created successfully.")

    logger.info("Loading observation collection from %s", cfg.observations)
    observations = load_tudat_object(obs.ObservationCollection, cfg.observations)
    logger.info(
        "Observation collection loaded with %d observation(s) in %d set(s).",
        len(observations.concatenated_times),
        len(observations.get_single_observation_sets()),
    )

    # The collection carries the observable types, link ends and epochs, but not
    # the model settings - those are not serializable.  Rebuilding them from the
    # collection is what keeps this stage independent of the collection config.
    observation_models = rebuild_observation_models(observations)
    logger.info("Rebuilt %d observation model(s) from the collection.", len(observation_models))

    parameter_set = get_estimatable_parameters(cfg, ctx, propagator_settings, bodies)
    logger.info("Parameter set for estimation created successfully.")
    logger.info("Initial parameter vector: %s", parameter_set.parameter_vector)

    estimator = est_an.Estimator(
        bodies,
        parameter_set,
        observation_models,
        propagator_settings,
        False,
    )
    estimation_input = _build_estimation_input(cfg, observations, parameter_set)

    # -- Solve ---------------------------------------------------------
    estimation_log_path = output_dir / "estimation_log.log"
    logger.info("Starting estimation...")
    with redirect_std(str(estimation_log_path)):
        estimation_output = estimator.perform_estimation(estimation_input)
    logger.info("Estimation completed. Solver output logged to %s", estimation_log_path)

    # -- Artifact ------------------------------------------------------
    output_path = save_tudat_object(
        estimation_output, os.path.join(output_dir, "estimation_output")
    )
    logger.info("Estimation output saved to %s", output_path)
    logger.info("Final estimated parameters: %s", estimation_output.final_parameters)
    logger.info("Best iteration: %d", estimation_output.best_iteration)

    output_path = save_tudat_object(
        estimation_output.simulation_results_per_iteration[estimation_output.best_iteration],
        os.path.join(output_dir, "final_iteration_simulation_results"),
    )
    logger.info("Final iteration simulation results saved to %s", output_path)
       
    logger.info("solve_estimation stage completed.")


if __name__ == "__main__":
    main()
