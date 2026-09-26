import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tudatpy.dynamics.propagation_setup as prop_setup
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro import frame_conversion
from tudatpy.astro.time_representation import DateTime, iso_string_to_epoch_time_object
from tudatpy.dynamics import simulator as sim
from tudatpy.estimation import estimation_analysis as est_an
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup
from tudatpy.util import redirect_std

from orbitdet.data import KernelManager
from orbitdet.estimation import get_apriori_covariance_matrix, get_estimatable_parameters
from orbitdet.observations import OutlierEngine, WeightEngine, create_observation_collection
from orbitdet.reproducibility import (
    RuntimeContext,
    aim_log_artifact_reference,
    aim_log_metrics,
    enforce_initialization,
    initialize,
)
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)
from orbitdet.utility import save_tudat_object
from orbitdet.visualization import (
    CovarianceEllipses,
    ParameterCorrelationHeatmap,
    ParameterHistoryPerIteration,
    PropagatedFormalErrorsCartesian,
    PropagatedFormalErrorsRSW,
    RSWDistanceWithUncertainty,
    ResidualHistogram,
    ResidualQQ,
    ResidualRMSPerIteration,
    Residuals,
    ResidualScatter,
    RSWDistance,
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/classic_triton_state",
)
@enforce_initialization
def main(cfg: DictConfig):
    logger.info(f"Starting main() with on process PID {os.getpid()}")

    # Inject start and end epochs into the runtime context
    ctx: RuntimeContext = initialize(cfg)
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)
    logging.info(
        f"""Running simulation from epochs"""
        f"""{DateTime.from_epoch_time_object(ctx.start_epoch).to_iso_string()}"""
        f"""to {DateTime.from_epoch_time_object(ctx.end_epoch).to_iso_string()}"""
    )

    # Load kernels
    logger.info("Loading kernels...")
    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    logger.info("Kernels loaded and furnished successfully.")

    # Prep environment
    bodies = get_environment(cfg, ctx)
    acc = get_dynamical_model(cfg, ctx, bodies)
    integ = get_integrator_settings(cfg, ctx)
    dep_vars = [
        prop_setup.dependent_variable.relative_position("Triton Spice", "Triton"),
        prop_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
        prop_setup.dependent_variable.keplerian_state("Triton Spice", "Neptune"),
        # prop_setup.dependent_variable.relative_velocity("Triton Spice", "Triton"),
    ]
    prop = get_propagator_settings(cfg, ctx, acc, integ, dependent_variables_to_save=dep_vars)
    logger.info("Environment and propagator settings created successfully.")

    # Create observation collection
    logger.info("Creating observation collection...")
    observations, observation_models, dataset_metadata = create_observation_collection(cfg, bodies)
    # Create observation simulators for pre-fit residuals
    ephemeris_observation_simulators = obs_sim_setup.create_observation_simulators(
        observation_models, bodies
    )
    logger.info(
        f"""Observation collection and observation simulators created """
        f"""successfully with {len(observations.concatenated_times)} """
        f"""observations."""
    )

    ################################################################
    ############################ PREFIT ############################
    ################################################################
    logger.info("Computing pre-fit residuals wrt spice...")
    obs.compute_residuals_and_dependent_variables(
        observations, ephemeris_observation_simulators, bodies
    )
    # Wrt spice
    Residuals(cfg, "prefit_residuals_spice", observations).plot()
    logger.info("Pre-fit residuals computed and plotted successfully.")

    logger.info("Computing pre-fit residuals wrt propagation...")
    sim.create_dynamics_simulator(bodies, prop)
    obs.compute_residuals_and_dependent_variables(
        observations, ephemeris_observation_simulators, bodies
    )
    Residuals(cfg, "prefit_residuals_prop", observations).plot()
    logger.info("Pre-fit residuals computed and plotted successfully.")

    ################################################################
    ###################### Outlier rejection #######################
    ################################################################
    outlier_cfg = OmegaConf.select(cfg, "outlier_rejection")
    if outlier_cfg is not None and outlier_cfg.get("enabled", False):
        logger.info("Applying outlier rejection...")
        outlier_engine = OutlierEngine.from_config(outlier_cfg)
        observations, rejection_metadata = outlier_engine.apply(observations, bodies)
        logger.info(
            "Outlier rejection complete: %d accepted, %d rejected out of %d",
            rejection_metadata["n_accepted"],
            rejection_metadata["n_rejected"],
            rejection_metadata["n_total_observations"],
        )
        # Save rejection metadata to JSON
        import json

        from hydra.core.hydra_config import HydraConfig

        rejection_path = (
            Path(HydraConfig.get().runtime.output_dir) / "outlier_rejection_metadata.json"
        )
        with open(rejection_path, "w") as f:
            json.dump(rejection_metadata, f, indent=2, default=str)
        logger.info("Outlier rejection metadata saved to %s", rejection_path)
    else:
        logger.info("Outlier rejection disabled.")

    ################################################################
    ########################## Weighting ###########################
    ################################################################
    weighting_cfg = OmegaConf.select(cfg, "weighting")
    if weighting_cfg is not None and weighting_cfg.get("enabled", False):
        logger.info("Applying weighting...")
        weight_engine = WeightEngine.from_config(weighting_cfg)
        from hydra.core.hydra_config import HydraConfig

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        observations, weights_df = weight_engine.apply(
            observations, bodies, output_dir=output_dir, dataset_metadata=dataset_metadata
        )
        logger.info(
            "Weighting complete: %s strategy applied, %d observations weighted",
            weighting_cfg.get("strategy", "unknown"),
            len(weights_df) if not weights_df.empty else 0,
        )
    else:
        logger.info("Weighting disabled.")

    # Plot weight groups if weighting was applied
    if weighting_cfg is not None and weighting_cfg.get("enabled", False):
        from orbitdet.visualization import WeightGroups, WeightSummaryTable

        fig_weight_groups, _ = WeightGroups(cfg, weights_df).plot()
        logger.info("Weight groups figure saved.")

        # Build and save the per-source summary table
        table_builder = WeightSummaryTable(cfg, weights_df)
        src_table = table_builder.build_source()
        grp_table = table_builder.build_group()
        # print("\n=== Per-source summary ===\n" + table_builder.to_string_source(src_table))
        # print("\n=== Per-group summary ===\n" + table_builder.to_string_group(grp_table))

        src_table.to_csv(output_dir / "weight_summary_source.csv", index=False)
        grp_table.to_csv(output_dir / "weight_summary_group.csv", index=False)
        logger.info("Weight summary tables saved to %s", output_dir)

        with open(output_dir / "weight_summary_source.tex", "w") as f:
            f.write(table_builder.to_latex_source(src_table))
        with open(output_dir / "weight_summary_group.tex", "w") as f:
            f.write(table_builder.to_latex_group(grp_table))
        logger.info("LaTeX weight summary tables saved to %s", output_dir)

        # Save as Excel with two sheets
        with pd.ExcelWriter(output_dir / "weight_summary.xlsx") as writer:
            src_table.to_excel(writer, sheet_name="Per source", index=False)
            grp_table.to_excel(writer, sheet_name="Per group", index=False)
        logger.info("Excel weight summary tables saved to %s", output_dir)

    ################################################################
    ######################### ESTIMATION ###########################
    ################################################################

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
    max_iterations_without_improvement = cfg.estimation.get(
        "max_iterations_without_improvement", cfg.estimation.max_iterations
    )
    convergence_settings = est_an.estimation_convergence_checker(
        maximum_iterations=cfg.estimation.max_iterations,
        number_of_iterations_without_improvement=max_iterations_without_improvement,
    )
    logger.info(
        f"Estimation convergence settings: max_iterations={cfg.estimation.max_iterations}, "
        f"max_iterations_without_improvement={max_iterations_without_improvement}"
    )
    # Build inverse a priori covariance matrix from configuration
    inverse_apriori_covariance = get_apriori_covariance_matrix(cfg)

    if inverse_apriori_covariance is not None:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            inverse_apriori_covariance=inverse_apriori_covariance,
            convergence_checker=convergence_settings,
        )
        logger.info(
            "Estimation input created with inverse a priori covariance matrix of shape %s",
            inverse_apriori_covariance.shape,
        )
    else:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            convergence_checker=convergence_settings,
        )
        logger.info("Estimation input created without inverse a priori covariance matrix.")
    # Set methodological options
    estimation_input.define_estimation_settings(
        save_state_history_per_iteration=False, save_residuals_and_parameters_per_iteration=True
    )
    from hydra.core.hydra_config import HydraConfig

    logger.info("Starting estimation...")
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    estimation_log_path = output_dir / "estimation_log.log"
    try:
        with redirect_std(str(estimation_log_path)):
            estimation_output = estimator.perform_estimation(estimation_input)
    except Exception as e:
        logger.error("Estimation failed: %s", e)
        logger.info("Estimation progression logged to %s", estimation_log_path)
        if not output_dir.exists():
            logger.warning("Unable to find estimation log file at %s", estimation_log_path)
        raise

    logger.info("Estimation progression logged to %s", estimation_log_path)
    save_tudat_object(estimation_output, output_dir.with_name("estimation_output"))
    save_tudat_object(observations, output_dir.with_name("observations"))
    logger.info("Estimation output saved to %s", output_dir.with_name("estimation_output"))
    logger.info("Observations saved to %s", output_dir.with_name("observations"))
    logger.info("Estimation completed successfully.")

    logger.info("Final estimated parameters: %s", estimation_output.final_parameters)

    ############################################################################
    ############################# LOG STATISTICS ###############################
    ############################################################################

    num_iterations = estimation_output.residual_history.shape[1]
    logger.info("Logging per-iteration metrics to Aim...")
    for i in range(num_iterations):
        rms_i = np.sqrt(np.mean(np.square(estimation_output.residual_history[:, i])))
        aim_log_metrics(
            {"residual_rms": float(rms_i)},
            step=i,
            context={"metric_type": "iteration"},
        )
    # Log final residual summary metrics to Aim
    final_residuals = estimation_output.final_residuals
    logger.info("Logging summary metrics to Aim...")
    aim_log_metrics(
        {
            "residuals_rms": float(np.sqrt(np.mean(np.square(final_residuals)))),
            "residuals_mean": float(np.mean(final_residuals)),
            "residuals_max": float(np.abs(final_residuals).max()),
            "residuals_std": float(np.std(final_residuals)),
            "num_observations": final_residuals.size,
            "num_iterations": num_iterations,
            "parameter_norm": float(np.linalg.norm(estimation_output.final_parameters)),
            "covariance_condition": float(np.linalg.cond(estimation_output.covariance)),
        },
        context={"metric_type": "summary"},
    )
    logger.info("Logged summary metrics to Aim.")


    #################################################################
    ######################## POST-FIT RESIDUALS #####################
    #################################################################
    logger.info("Propagating final estimated state to generate post-fit residuals...")
    parameters = estimation_output.parameter_history[:, estimation_output.best_iteration]
    prop.initial_states = parameters
    final_result = sim.create_dynamics_simulator(bodies, prop)
    Residuals(cfg, "postfit_residuals", observations).plot()


    #############################################################################
    ################################## Figures ##################################
    #############################################################################
    logger.info("Plotting figures...")

    ResidualQQ(cfg, observations).plot()
    ResidualScatter(cfg, observations).plot()
    ResidualHistogram(cfg, observations).plot()
    ResidualRMSPerIteration(cfg, estimation_output).plot()
    ParameterCorrelationHeatmap(cfg, estimation_output).plot()
    ParameterHistoryPerIteration(cfg, estimation_output).plot()
    CovarianceEllipses(cfg, estimation_output, bodies, ctx).plot()
    RSWDistance(
        cfg,
        final_result.propagation_results,
        dep_vars[0],
        central_body="Neptune",
    ).plot()

    # ====================================================================
    # Propagate covariance and plot formal errors
    # ====================================================================
    logger.info("Propagating covariance over the full time arc ...")
    state_transition_interface = estimator.state_transition_interface
    start_epoch = float(ctx.start_epoch.to_float())
    end_epoch = float(ctx.end_epoch.to_float())
    step_days = OmegaConf.select(cfg, "propagation.step_days", default=10.0)
    step_seconds = step_days * 86400.0
    output_times = np.arange(start_epoch, end_epoch, step_seconds)

    propagated_covariances = est_an.propagate_covariance(
        estimation_output.covariance, state_transition_interface, output_times
    )
    propagated_formal_errors = est_an.propagate_formal_errors(
        initial_covariance=estimation_output.covariance,
        state_transition_interface=state_transition_interface,
        output_times=output_times,
    )

    epochs = np.array(list(propagated_formal_errors.keys()))
    formal_errors = np.array(list(propagated_formal_errors.values()))

    # RSW rotation
    n_epochs = len(epochs)
    fe_rsw = np.zeros((n_epochs, 6))
    for i, epoch in enumerate(epochs):
        state_est = bodies.get("Triton").ephemeris.cartesian_state(epoch)
        rot_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(state_est)
        full_rot = np.block([[rot_matrix, np.zeros((3, 3))], [np.zeros((3, 3)), rot_matrix]])
        cov = propagated_covariances[epoch]
        cov_rsw = full_rot @ cov @ full_rot.T
        fe_rsw[i] = np.sqrt(np.diag(cov_rsw))

    logger.info("Plotting propagated formal errors ...")
    PropagatedFormalErrorsCartesian(cfg, epochs, formal_errors).plot()
    PropagatedFormalErrorsRSW(cfg, epochs, fe_rsw).plot()
    logger.info("Propagated formal errors plotted.")

    # Compute RSW position differences at covariance epochs for uncertainty-vs-distance plot
    logger.info("Computing RSW distance at covariance propagation epochs ...")
    rsw_at_cov_epochs = np.zeros((n_epochs, 3))
    for i, epoch in enumerate(epochs):
        spice_state = bodies.get("Triton Spice").ephemeris.cartesian_state(epoch)
        triton_state = bodies.get("Triton").ephemeris.cartesian_state(epoch)
        rel_pos = spice_state[:3] - triton_state[:3]
        rot_matrix = frame_conversion.inertial_to_rsw_rotation_matrix(triton_state)
        rsw_at_cov_epochs[i] = rot_matrix @ rel_pos
    rsw_sigma = fe_rsw[:, :3]

    RSWDistanceWithUncertainty(cfg, epochs, rsw_at_cov_epochs, rsw_sigma).plot()
    logger.info("RSW distance with uncertainty envelopes plotted.")
    from matplotlib import pyplot as plt
    plt.show()
    # Save propagated formal errors as NumPy archive
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    np.savez(
        output_dir / "propagated_formal_errors.npz",
        epochs=epochs,
        formal_errors=formal_errors,
        formal_errors_rsw=fe_rsw,
    )
    logger.info("Propagated formal errors saved to %s", output_dir / "propagated_formal_errors.npz")

    # Log artifacts to Aim
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        aim_log_artifact_reference(config_path)
    aim_log_artifact_reference(output_dir.with_name("observations.tudat"))
    aim_log_artifact_reference(output_dir.with_name("estimation_output.tudat"))
    aim_log_artifact_reference(output_dir.with_name("estimation_log.tudat"))
    logger.info("Attached artifacts to Aim.")


if __name__ == "__main__":
    main()
