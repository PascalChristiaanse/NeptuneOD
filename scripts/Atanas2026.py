import logging

import hydra
import tudatpy.dynamics.propagation_setup as prop_setup
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.data import KernelManager
from orbitdet.estimation import get_estimatable_parameters
from orbitdet.observations import create_observation_collection
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/atanas2026_simulated",
)
@enforce_initialization
def main(cfg: DictConfig):
    ctx: RuntimeContext = initialize(cfg)

    # Inject start and end epochs into the runtime context
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)

    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    logger.info("Configuration loaded and runtime initialized successfully.")

    bodies = get_environment(cfg, ctx)
    logger.info("Environment created successfully.")
    acc = get_dynamical_model(cfg, ctx, bodies)
    logger.info("Dynamical model created successfully.")

    integ = get_integrator_settings(cfg, ctx)
    logger.info("Integrator settings created successfully.")
    dep_vars = [
        prop_setup.dependent_variable.relative_position("Triton", "Neptune"),
        prop_setup.dependent_variable.single_acceleration(
            prop_setup.acceleration.spherical_harmonic_gravity_type, "Triton", "Neptune"
        ),  # noqa: E501
    ]
    prop = get_propagator_settings(cfg, ctx, acc, integ, dependent_variables_to_save=dep_vars)
    logger.info("Propagator settings created successfully.")

    logger.info("Generating observations from collection...")


    observations, observation_models = create_observation_collection(cfg, bodies)

    # observations, observation_models = generate_observations(cfg, ctx, bodies)
    logger.info("Observations generated successfully.")

    logger.info("Simulation setup complete. Ready for propagation and estimation.")

    parameter_set = get_estimatable_parameters(cfg, ctx, prop, bodies)
    logger.info("Parameter set for estimation created successfully.")
    logger.info(f"Initial parameter set: {parameter_set.parameter_vector}")

    estimator = est_an.Estimator(
        bodies,
        parameter_set,
        observation_models,
        prop,
        False,
    )
    convergence_settings = est_an.estimation_convergence_checker(
        maximum_iterations=cfg.estimation.max_iterations
    )
    estimation_input = est_an.EstimationInput(
        observations_and_times=observations,
        # inverse_apriori_covariance=inverse_apriori_cov,
        convergence_checker=convergence_settings,
    )

    # Set methodological options
    estimation_input.define_estimation_settings(
        save_state_history_per_iteration=True, save_residuals_and_parameters_per_iteration=True
    )

    logger.info("Starting estimation...")
    estimation_output: est_an.EstimationOutput = estimator.perform_estimation(estimation_input)

    logger.info("Estimation completed successfully.")

    import matplotlib.pyplot as plt
    import numpy as np

    state_history: dict[float, np.ndarray] = estimation_output.simulation_results_per_iteration[
        0
    ].dynamics_results.state_history

    times = sorted(state_history.keys())
    states = np.array([state_history[t] for t in times])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(states[:, 0], states[:, 1], states[:, 2], label="Triton Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Triton Trajectory from Estimation")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()


# Observations match
# Observation times match
# Initial parameter vectors match
# Simulation state history DOES NOT MATCH
