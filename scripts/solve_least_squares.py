import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tudatpy.dynamics.propagation_setup as prop_setup
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import estimation_analysis as est_an
from tudatpy.util import redirect_std

from orbitdet.data import KernelManager
from orbitdet.estimation import get_apriori_covariance_matrix, get_estimatable_parameters
from orbitdet.observations import create_observation_collection
from orbitdet.reproducibility import (
    RuntimeContext,
    aim_log_artifact,
    aim_log_figure,
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
    config_name="experiments/minimal_experiment",
)
@enforce_initialization
def main(cfg: DictConfig):
    ctx: RuntimeContext = initialize(cfg)

    # Inject start and end epochs into the runtime context
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

    # import tudatpy.estimation.observations as obs
    # observations = obs.ObservationCollection.load_from_binary("atanasObservations")
    # logger.warning("Observations loaded from binary file 'atanasObservations'.")

    logger.info("Observations generated successfully.")

    # Plot and save pre-fit residuals before estimation modifies them
    from orbitdet.visualization import plot_residuals

    fig_prefit_residuals, ax_prefit_residuals = plot_residuals(cfg, observations)

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
    with redirect_std(str(estimation_log_path)):
        estimation_output = estimator.perform_estimation(estimation_input)
    logger.info("Estimation progression logged to %s", estimation_log_path)

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

    # Plot dependent variable (Triton Spice relative position, Keplerian states)
    from orbitdet.visualization.dependent_variable import plot_dependent_variable

    fig_dep_relpos, axes_dep_relpos = plot_dependent_variable(
        cfg, estimation_output.simulation_results_per_iteration[-1].dynamics_results, dep_vars[0]
    )
    fig_dep_triton_kep, axes_dep_triton_kep = plot_dependent_variable(
        cfg, estimation_output.simulation_results_per_iteration[-1].dynamics_results, dep_vars[1]
    )
    fig_dep_spice_kep, axes_dep_spice_kep = plot_dependent_variable(
        cfg, estimation_output.simulation_results_per_iteration[-1].dynamics_results, dep_vars[2]
    )

    # Plot residual histogram, Q-Q, and scatter
    from orbitdet.visualization import (
        plot_residual_histogram,
        plot_residual_qq,
        plot_residual_scatter,
    )

    fig_hist, axes_hist = plot_residual_histogram(cfg, observations)
    fig_qq, axes_qq = plot_residual_qq(cfg, observations)
    fig_scatter, ax_scatter = plot_residual_scatter(cfg, observations)

    # Save all figures to the output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save TudatPy objects to binary .tudat files
    logger.info("Saving TudatPy objects to disk...")
    observations_path = save_tudat_object(observations, output_dir / "observations")
    logger.info("Observation collection saved to %s", observations_path)

    estimation_output_path = save_tudat_object(estimation_output, output_dir / "estimation_output")
    logger.info("Estimation output saved to %s", estimation_output_path)

    fig_prefit_path = output_dir / "prefit_residuals.pdf"
    fig_prefit_residuals.savefig(fig_prefit_path)
    logger.info(f"Pre-fit residuals plot saved to {fig_prefit_path}")

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

    fig_diff_path = output_dir / "differenced_dependent_variables.pdf"
    fig_diff.savefig(fig_diff_path)
    logger.info(f"Differenced dependent variables plot saved to {fig_diff_path}")

    fig_dep_relpos_path = output_dir / "dependent_variable_relative_position.pdf"
    fig_dep_relpos.savefig(fig_dep_relpos_path)
    logger.info(f"Dependent variable relative position plot saved to {fig_dep_relpos_path}")

    fig_dep_triton_kep_path = output_dir / "dependent_variable_triton_keplerian.pdf"
    fig_dep_triton_kep.savefig(fig_dep_triton_kep_path)
    logger.info(f"Dependent variable Triton Keplerian plot saved to {fig_dep_triton_kep_path}")

    fig_dep_spice_kep_path = output_dir / "dependent_variable_spice_keplerian.pdf"
    fig_dep_spice_kep.savefig(fig_dep_spice_kep_path)
    logger.info(f"Dependent variable Spice Keplerian plot saved to {fig_dep_spice_kep_path}")

    fig_hist_path = output_dir / "residual_histogram.pdf"
    fig_hist.savefig(fig_hist_path)
    logger.info(f"Residual histogram plot saved to {fig_hist_path}")

    fig_qq_path = output_dir / "residual_qq.pdf"
    fig_qq.savefig(fig_qq_path)
    logger.info(f"Residual Q-Q plot saved to {fig_qq_path}")

    fig_scatter_path = output_dir / "residual_scatter.pdf"
    fig_scatter.savefig(fig_scatter_path)
    logger.info(f"Residual scatter plot saved to {fig_scatter_path}")

    # Log all figures to Aim (interactive Figures + static Images)
    logger.info("Logging figures to Aim...")
    aim_log_figure(fig_prefit_residuals, name="prefit_residuals")
    aim_log_figure(fig_residuals, name="postfit_residuals")
    aim_log_figure(fig_psd, name="postfit_residuals_psd")
    aim_log_figure(fig_rms, name="residual_rms_per_iteration")
    aim_log_figure(fig_corr, name="parameter_correlation_heatmap")
    aim_log_figure(fig_param, name="parameter_history_per_iteration")
    aim_log_figure(fig_ellipses, name="covariance_ellipses")
    aim_log_figure(fig_rsw, name="rsw_distance")
    aim_log_figure(fig_diff, name="differenced_dependent_variables")
    aim_log_figure(fig_dep_relpos, name="dependent_variable_relative_position")
    aim_log_figure(fig_dep_triton_kep, name="dependent_variable_triton_keplerian")
    aim_log_figure(fig_dep_spice_kep, name="dependent_variable_spice_keplerian")
    aim_log_figure(fig_hist, name="residual_histogram")
    aim_log_figure(fig_qq, name="residual_qq")
    aim_log_figure(fig_scatter, name="residual_scatter")
    logger.info("Logged figures to Aim.")

    # Attach saved PDFs as artifacts
    logger.info("Attaching artifacts to Aim...")
    aim_log_artifact(fig_prefit_path)
    aim_log_artifact(fig_residuals_path)
    aim_log_artifact(fig_psd_path)
    aim_log_artifact(fig_rms_path)
    aim_log_artifact(fig_corr_path)
    aim_log_artifact(fig_param_path)
    aim_log_artifact(fig_ellipses_path)
    aim_log_artifact(fig_rsw_path)
    aim_log_artifact(fig_diff_path)
    aim_log_artifact(fig_dep_relpos_path)
    aim_log_artifact(fig_dep_triton_kep_path)
    aim_log_artifact(fig_dep_spice_kep_path)
    aim_log_artifact(fig_hist_path)
    aim_log_artifact(fig_qq_path)
    aim_log_artifact(fig_scatter_path)
    # Also log the config as an artifact
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        aim_log_artifact(config_path)
    # Also log the saved TudatPy objects and estimation log as artifacts
    aim_log_artifact(observations_path.with_suffix(".tudat"))
    aim_log_artifact(estimation_output_path.with_suffix(".tudat"))
    aim_log_artifact(estimation_log_path.with_suffix(".tudat"))
    logger.info("Attached artifacts to Aim.")

    # fig_traj_path = output_dir / "triton_trajectory.pdf"
    # fig_traj.savefig(fig_traj_path)
    # logger.info(f"Triton trajectory plot saved to {fig_traj_path}")

    # plt.show()


if __name__ == "__main__":
    main()
