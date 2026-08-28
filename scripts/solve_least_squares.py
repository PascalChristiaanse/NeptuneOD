import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tudatpy.dynamics.propagation_setup as prop_setup
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.dynamics import simulator as sim
from tudatpy.estimation import estimation_analysis as est_an
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup
from tudatpy.util import redirect_std

from orbitdet.data import KernelManager
from orbitdet.estimation import get_apriori_covariance_matrix, get_estimatable_parameters
from orbitdet.observations import create_observation_collection
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

logger = logging.getLogger(__name__)


def _build_timestamp_series(
    dataframe: pd.DataFrame, year_col: str, month_col: str, day_col: str
) -> pd.Series:
    """Combine year/month/day columns into a pandas timestamp series.

    The day column may contain a fractional part (e.g. ``24.583229``), which is
    interpreted as the fraction of the day elapsed.

    Args:
        dataframe: The DataFrame with the observation data.
        year_col: Name of the year column.
        month_col: Name of the month column.
        day_col: Name of the day column (may include a fractional part).

    Returns:
        A pandas Series with datetime64 values; missing/invalid rows become NaT.
    """
    day = pd.to_numeric(dataframe[day_col], errors="coerce")
    day_integer = np.floor(day)
    day_fraction_seconds = (day - day_integer) * 86400.0

    timestamps = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(dataframe[year_col], errors="coerce"),
                "month": pd.to_numeric(dataframe[month_col], errors="coerce"),
                "day": day_integer,
            }
        ),
        errors="coerce",
    )
    return timestamps + pd.to_timedelta(day_fraction_seconds, unit="s")


def _dataset_time_columns(dataset_cfg: DictConfig) -> tuple[str, str, str] | None:
    """Find the year, month and day column names for a dataset config.

    The NSDB dataset configs describe the columns of the associated data file via
    ``format_columns`` (a mapping of 1-based column index to column name). This
    function locates the column names matching the year, month and day of the
    moment of observation. Alternative spellings ("Day of the moment of
    observation" without "with decimals") are also accepted, and month may be
    written as "Month" or "Month of the moment of observation".

    Args:
        dataset_cfg: The dataset config containing the ``format_columns`` mapping.

    Returns:
        A tuple of (year, month, day) column names, or None when not all three
        columns could be identified (e.g. datasets that only contain a Julian
        date, or micrometric relative observations).
    """
    fmt = dataset_cfg.get("format_columns", {})
    if not fmt:
        return None

    def _find(candidates: list[str]) -> str | None:
        for index, name in fmt.items():
            normalized = str(name).strip().lower()
            if any(candidate in normalized for candidate in candidates):
                return str(name)
        return None

    year_col = _find(["year"])
    month_col = _find(["month"])
    day_col = _find(["day"]) if year_col is not None else None

    if year_col is None or month_col is None or day_col is None:
        return None
    return year_col, month_col, day_col


def detect_date_bounds_from_datasets(cfg: DictConfig) -> tuple[str | None, str | None]:
    """Detect the observation date bounds from the configured datasets.

    Walks through all datasets listed in the experiment configuration and, for
    each one that references a data file with year/month/day columns, reads the
    file to find the earliest and latest observation time. The overall bounds
    across all datasets are returned as ISO-8601 strings.

    Datasets that do not expose year/month/day columns (e.g. relative position
    angle and separation micrometric observations, or data with a single Julian
    date column) are skipped.

    Args:
        cfg: The Hydra experiment configuration containing the ``datasets`` list.

    Returns:
        A tuple of (start_date, end_date) as ISO-8601 strings, or (None, None)
        when no dates could be detected.
    """
    datasets = OmegaConf.select(cfg, "datasets")
    if datasets is None:
        return None, None

    min_timestamp = None
    max_timestamp = None
    for set_name, dataset_cfg in datasets.items():
        columns = _dataset_time_columns(dataset_cfg)
        if columns is None:
            logger.debug(
                "Skipping dataset %s: no year/month/day columns found in format_columns.",
                set_name,
            )
            continue

        file_path = Path(dataset_cfg.file)
        if not file_path.exists():
            logger.warning("Skipping dataset %s: data file %s does not exist.", set_name, file_path)
            continue

        try:
            dataframe = pd.read_csv(
                file_path, sep=r"\s+", header=None, comment="#", engine="python"
            )
        except Exception as exc:
            logger.warning("Skipping dataset %s: could not read data file: %s", set_name, exc)
            continue

        # Map the format_columns indices (1-based) to the positional column names.
        fmt = dict(dataset_cfg.format_columns)
        col_names = list(dataframe.columns)

        def _keyfunc(k):
            try:
                return int(k)
            except Exception:
                return str(k)

        for index in sorted(fmt.keys(), key=_keyfunc):
            name = fmt.get(index, fmt.get(str(index), None))
            pos = None
            try:
                pos = int(index) - 1
            except Exception:
                try:
                    pos = int(index)
                except Exception:
                    pos = None
            if pos is not None and 0 <= pos < len(col_names):
                col_names[pos] = name if name is not None else col_names[pos]

        dataframe.columns = col_names
        timestamps = _build_timestamp_series(dataframe, *columns)
        timestamps = timestamps.dropna()
        if timestamps.empty:
            logger.warning("Skipping dataset %s: no valid timestamps found.", set_name)
            continue

        dataset_min = timestamps.min()
        dataset_max = timestamps.max()
        logger.debug(
            "Dataset %s: observation dates from %s to %s", set_name, dataset_min, dataset_max
        )
        if min_timestamp is None or dataset_min < min_timestamp:
            min_timestamp = dataset_min
        if max_timestamp is None or dataset_max > max_timestamp:
            max_timestamp = dataset_max

    if min_timestamp is None or max_timestamp is None:
        return None, None
    return min_timestamp.isoformat(), max_timestamp.isoformat()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config",
)
# @enforce_initialization Disabled to support submitit multiprocessing
def main(cfg: DictConfig):
    import os
    logger.info(f"Starting main() with on process PID {os.getpid()}")

    # Inject start and end epochs into the runtime context
    ctx: RuntimeContext = initialize(cfg) 
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

    # Detect the actual observation date bounds from the configured datasets
    detected_start, detected_end = detect_date_bounds_from_datasets(cfg)
    if detected_start is not None and detected_end is not None:
        ctx.start_epoch = iso_string_to_epoch_time_object(detected_start)
        ctx.end_epoch = iso_string_to_epoch_time_object(detected_end)
        logger.info(
            "Detected observation date bounds from datasets: %s to %s.",
            detected_start,
            detected_end,
        )

        # Add a buffer around the observation dates to cover the propagation
        # arc before the first and after the last observation.
        ctx.start_epoch = ctx.start_epoch - 365.25 * 24 * 3600
        ctx.end_epoch = ctx.end_epoch + 365.25 * 24 * 360
    else:
        logger.warning(
            "Could not detect observation date bounds from datasets; "
            "using configured start_date/end_date instead."
        )

    from tudatpy.astro.time_representation import DateTime

    logger.info(
        "Detected start epoch from datasets: "
        f"{DateTime.from_epoch_time_object(ctx.start_epoch).to_iso_string()}"
        " (with one year buffer)."
    )
    logger.info(
        "Detected end epoch from datasets: "
        f"{DateTime.from_epoch_time_object(ctx.end_epoch).to_iso_string()}"
        " (with one year buffer)."
    )

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

    # Create observation simulators for pre-fit residuals
    ephemeris_observation_simulators = obs_sim_setup.create_observation_simulators(
        observation_models, bodies
    )
    logger.info("Observation simulators created successfully.")

    if prop.processing_settings.set_integrated_result:
        logger.info(
            "Prefit residuals will be computed using the integrated result from the propagator."
        )
        sim.create_dynamics_simulator(bodies, prop)

    # Populate residuals in SingleObservationSets
    obs.compute_residuals_and_dependent_variables(
        observations, ephemeris_observation_simulators, bodies
    )

    # Plot and save pre-fit residuals before estimation modifies them
    from orbitdet.visualization import Residuals

    fig_prefit_residuals, ax_prefit_residuals = Residuals(cfg, observations).plot()
    logger.info("Pre-fit residuals computed successfully.")

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

    if inverse_apriori_covariance is not None:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            inverse_apriori_covariance=inverse_apriori_covariance,
            convergence_checker=convergence_settings,
        )
    else:
        estimation_input = est_an.EstimationInput(
            observations_and_times=observations,
            convergence_checker=convergence_settings,
        )
    # Set methodological options
    estimation_input.define_estimation_settings(
        save_state_history_per_iteration=True, save_residuals_and_parameters_per_iteration=True
    )
    from hydra.core.hydra_config import HydraConfig

    # estimation_input.save_to_binary(HydraConfig.get().runtime.output_dir + "/estimation_input")
    logger.info("Starting estimation...")

    estimation_log_path = Path(HydraConfig.get().runtime.output_dir) / "estimation_progression.log"
    try:
        with redirect_std(str(estimation_log_path)):
            estimation_output = estimator.perform_estimation(estimation_input)
    except Exception as e:
        logger.error("Estimation failed: %s", e)
        logger.info("Estimation progression logged to %s", estimation_log_path)
        # write estimation log file to logger
        if estimation_log_path.exists():
            with open(estimation_log_path) as f:
                for line in f:
                    logger.info("Estimation: %s", line.rstrip("\n"))
        else:
            logger.warning("Unable to find estimation log file at %s", estimation_log_path)
        # Exit program cleanly with error code
        import sys

        sys.exit(1)

    logger.info("Estimation progression logged to %s", estimation_log_path)
    save_tudat_object(estimation_output, estimation_log_path.with_suffix(".tudat"))
    save_tudat_object(observations, estimation_log_path.with_name("observations.tudat"))
    logger.info("Estimation output saved to %s", estimation_log_path.with_suffix(".tudat"))
    logger.info("Observations saved to %s", estimation_log_path.with_name("observations.tudat"))
    logger.info("Estimation completed successfully.")

    # Also log the estimation progress to the regular logger
    if estimation_log_path.exists():
        with open(estimation_log_path) as f:
            for line in f:
                logger.info("Estimation: %s", line.rstrip("\n"))

    # Log residual RMS per iteration to Aim
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

    logger.info("Estimation completed successfully.")

    # Plot post-fit residuals
    from orbitdet.visualization import Residuals, ResidualsPSD

    fig_residuals, ax_residuals = Residuals(cfg, observations).plot()

    # Plot residual PSD
    residuals_psd_cfg = cfg.get("residuals_psd", {})
    window_length_days = residuals_psd_cfg.get("window_length_days", 30.0)
    fig_psd, ax_psd = ResidualsPSD(
        cfg, observations, window_length_days, cfg.figures.get("residuals_psd", {})
    ).plot()

    # Plot residual RMS per iteration
    from orbitdet.visualization import ResidualRMSPerIteration

    fig_rms, ax_rms = ResidualRMSPerIteration(cfg, estimation_output).plot()

    # Plot parameter correlation heatmap
    from orbitdet.visualization import ParameterCorrelationHeatmap

    fig_corr, ax_corr = ParameterCorrelationHeatmap(cfg, estimation_output).plot()

    # Plot parameter history per iteration
    from orbitdet.visualization import ParameterHistoryPerIteration

    fig_param, ax_param = ParameterHistoryPerIteration(cfg, estimation_output).plot()

    # Plot covariance ellipses
    from orbitdet.visualization import CovarianceEllipses

    fig_ellipses, axes_ellipses = CovarianceEllipses(cfg, estimation_output, bodies, ctx).plot()

    from orbitdet.visualization import DifferencedDependentVariables

    fig_diff, axes_diff = DifferencedDependentVariables(
        cfg,
        reference_result=estimation_output.simulation_results_per_iteration[0].dynamics_results,
        comparison_results=[estimation_output.simulation_results_per_iteration[0].dynamics_results],
        reference_dependent_variable=dep_vars[2],
        comparison_dependent_variables=[dep_vars[1]],
    ).plot()

    # Plot RSW decomposition of relative position (Triton Spice vs Triton)
    from orbitdet.visualization import RSWDistance

    fig_rsw, axes_rsw = RSWDistance(
        cfg,
        estimation_output.simulation_results_per_iteration[-1].dynamics_results,
        dep_vars[0],
        central_body="Neptune",
    ).plot()

    # Plot dependent variable (Triton Spice relative position, Keplerian states)
    from orbitdet.visualization import DependentVariable

    fig_dep_relpos, axes_dep_relpos = DependentVariable(
        cfg, estimation_output.simulation_results_per_iteration[-1].dynamics_results, dep_vars[0]
    ).plot()
    fig_dep_triton_kep, axes_dep_triton_kep = DependentVariable(
        cfg, estimation_output.simulation_results_per_iteration[-1].dynamics_results, dep_vars[1]
    ).plot()
    fig_dep_spice_kep, axes_dep_spice_kep = DependentVariable(
        cfg, estimation_output.simulation_results_per_iteration[-1].dynamics_results, dep_vars[2]
    ).plot()

    # Plot residual histogram, Q-Q, and scatter
    from orbitdet.visualization import ResidualHistogram, ResidualQQ, ResidualScatter

    fig_hist, axes_hist = ResidualHistogram(cfg, observations).plot()
    fig_qq, axes_qq = ResidualQQ(cfg, observations).plot()
    fig_scatter, ax_scatter = ResidualScatter(cfg, observations).plot()

    # Save all figures to the output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save TudatPy objects to binary .tudat files
    logger.info("Saving TudatPy objects to disk...")
    observations_path = save_tudat_object(observations, output_dir / "observations")
    logger.info("Observation collection saved to %s", observations_path)

    estimation_output_path = save_tudat_object(estimation_output, output_dir / "estimation_output")
    logger.info("Estimation output saved to %s", estimation_output_path)

    # The Plot base class already saved each figure as a PDF, logged it to Aim
    # as a static image, and attached the PDF as an artifact reference. Only
    # the config and binary TudatPy objects still need explicit references.
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        aim_log_artifact_reference(config_path)
    aim_log_artifact_reference(observations_path.with_suffix(".tudat"))
    aim_log_artifact_reference(estimation_output_path.with_suffix(".tudat"))
    aim_log_artifact_reference(estimation_log_path.with_suffix(".tudat"))
    logger.info("Attached artifacts to Aim.")

    # fig_traj_path = output_dir / "triton_trajectory.pdf"
    # fig_traj.savefig(fig_traj_path)
    # logger.info(f"Triton trajectory plot saved to {fig_traj_path}")

    # plt.show()


if __name__ == "__main__":
    main()
