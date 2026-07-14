import logging

import hydra
import numpy as np
import tudatpy.dynamics.propagation_setup as prop_setup
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.data import KernelManager
from orbitdet.estimation import get_apriori_covariance_matrix, get_estimatable_parameters
from orbitdet.observations import create_observation_collection
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)

logger = logging.getLogger(__name__)


def compute_apriori_vs_design_matrix_ratio(
    estimation_output: est_an.EstimationOutput, inverse_a_priori: np.ndarray
) -> float:
    """
    Computes the ratio of the a priori covariance matrix to the design matrix.

    Parameters:
        estimation_output (est_an.EstimationOutput): The output of the estimation process.
        inverse_a_priori (np.ndarray): The inverse of the a priori covariance matrix.

    Returns:
        float: The ratio of the a priori covariance matrix to the design matrix.
    """
    H = estimation_output.design_matrix
    W = np.identity(H.shape[0])  # Assuming equal weights for all observations
    HtWH = H.T @ W @ H

    logger.info("Design matrix vs a priori covariance ratio:")
    logger.info(f"Design matrix (HtWH):\n{HtWH}")
    logger.info(f"Inverse a priori covariance:\n{inverse_a_priori}")
    logger.info(f"Ratio (HtWH / inverse_a_priori):\n{np.diag(HtWH) / np.diag(inverse_a_priori)}")
    return HtWH, np.diag(inverse_a_priori) / np.diag(HtWH)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/atanas_triton_state",
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
        prop_setup.dependent_variable.relative_position("Triton Spice", "Triton"),
        prop_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
        prop_setup.dependent_variable.keplerian_state("Triton Spice", "Neptune"),
        # prop_setup.dependent_variable.relative_velocity("Triton Spice", "Triton"),
    ]
    prop = get_propagator_settings(cfg, ctx, acc, integ, dependent_variables_to_save=dep_vars)
    logger.info("Propagator settings created successfully.")

    logger.info("Generating observations from collection...")

    observations, observation_models = create_observation_collection(cfg, bodies)
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
    # Build inverse a priori covariance matrix from configuration
    inverse_apriori_covariance = get_apriori_covariance_matrix(cfg)

    estimation_input = est_an.EstimationInput(
        observations_and_times=observations,
        inverse_apriori_covariance=inverse_apriori_covariance,
        convergence_checker=convergence_settings,
    )

    # Set methodological options
    estimation_input.define_estimation_settings(
        save_state_history_per_iteration=True, save_residuals_and_parameters_per_iteration=True
    )

    logger.info("Starting estimation...")
    estimation_output = estimator.perform_estimation(estimation_input)

    logger.info("Estimation completed successfully.")

    # Plot post-fit residuals
    from orbitdet.visualization import plot_residuals, plot_residuals_psd

    fig_residuals, ax_residuals = plot_residuals(cfg, observations)

    # Plot residual PSD
    residuals_psd_cfg = cfg.get("residuals_psd", {})
    window_length_days = residuals_psd_cfg.get("window_length_days", 30.0)
    fig_psd, ax_psd = plot_residuals_psd(
        cfg, observations, window_length_days, cfg.figures.get("residuals_psd", {})
    )

    # Plot residual RMS per iteration
    from orbitdet.visualization.residual_rms_per_iteration import plot_residual_rms_per_iteration

    fig_rms, ax_rms = plot_residual_rms_per_iteration(cfg, estimation_output)

    # Plot parameter correlation heatmap
    from orbitdet.visualization.parameter_correlation_heatmap import (
        plot_parameter_correlation_heatmap,
    )

    fig_corr, ax_corr = plot_parameter_correlation_heatmap(cfg, estimation_output)

    # Plot parameter history per iteration
    from orbitdet.visualization.parameter_history_per_iteration import (
        plot_parameter_history_per_iteration,
    )

    fig_param, ax_param = plot_parameter_history_per_iteration(cfg, estimation_output)

    # Plot covariance ellipses
    from orbitdet.visualization.covariance_ellipses import plot_covariance_ellipses

    fig_ellipses, axes_ellipses = plot_covariance_ellipses(cfg, estimation_output, bodies, ctx)

    from orbitdet.visualization.dependent_variable_differenced import (
        plot_differenced_dependent_variables,
    )

    fig_diff, axes_diff = plot_differenced_dependent_variables(
        cfg,
        reference_result=estimation_output.simulation_results_per_iteration[0].dynamics_results,
        comparison_results=[estimation_output.simulation_results_per_iteration[0].dynamics_results],
        reference_dependent_variable=dep_vars[2],
        comparison_dependent_variables=[dep_vars[1]],
    )

    # Plot RSW decomposition of relative position (Triton Spice vs Triton)
    from orbitdet.visualization.RSW_distance import plot_RSW_distance

    fig_rsw, axes_rsw = plot_RSW_distance(
        cfg,
        estimation_output.simulation_results_per_iteration[-1].dynamics_results,
        dep_vars[0],
        central_body="Neptune",
    )

    # Save all figures to the output directory
    from pathlib import Path

    from hydra.core.hydra_config import HydraConfig

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_residuals_path = output_dir / "postfit_residuals.pdf"
    fig_residuals.savefig(fig_residuals_path)
    logger.info(f"Post-fit residuals plot saved to {fig_residuals_path}")

    fig_psd_path = output_dir / "postfit_residuals_psd.pdf"
    fig_psd.savefig(fig_psd_path)
    logger.info(f"Post-fit residual PSD plot saved to {fig_psd_path}")

    fig_rms_path = output_dir / "residual_rms_per_iteration.pdf"
    fig_rms.savefig(fig_rms_path)
    logger.info(f"Residual RMS per iteration plot saved to {fig_rms_path}")

    fig_corr_path = output_dir / "parameter_correlation_heatmap.pdf"
    fig_corr.savefig(fig_corr_path)
    logger.info(f"Parameter correlation heatmap saved to {fig_corr_path}")

    fig_param_path = output_dir / "parameter_history_per_iteration.pdf"
    fig_param.savefig(fig_param_path)
    logger.info(f"Parameter history per iteration plot saved to {fig_param_path}")

    fig_ellipses_path = output_dir / "covariance_ellipses.pdf"
    fig_ellipses.savefig(fig_ellipses_path)
    logger.info(f"Covariance ellipses plot saved to {fig_ellipses_path}")

    fig_rsw_path = output_dir / "rsw_distance.pdf"
    fig_rsw.savefig(fig_rsw_path)
    logger.info(f"RSW distance plot saved to {fig_rsw_path}")

    # fig_traj_path = output_dir / "triton_trajectory.pdf"
    # fig_traj.savefig(fig_traj_path)
    # logger.info(f"Triton trajectory plot saved to {fig_traj_path}")

    from matplotlib import pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()
