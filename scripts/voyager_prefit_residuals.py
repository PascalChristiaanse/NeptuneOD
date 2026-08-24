import logging
import os
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.data import KernelManager
from orbitdet.observations.collection import create_observation_collection
from orbitdet.reproducibility import (
    RuntimeContext,
    aim_log_artifact,
    aim_log_metrics,
    enforce_initialization,
    initialize,
)
from orbitdet.simulation import (
    get_environment,
)
from orbitdet.visualization import Residuals, ResidualsPSD

display = os.environ.get("DISPLAY")
is_headless_display = display == ":99" or display == "localhost:99" or display == "127.0.0.1:99"

matplotlib.rcParams["webagg.port"] = 8988
matplotlib.rcParams["webagg.open_in_browser"] = False

if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or is_headless_display:
    matplotlib.use("WebAgg", force=True)
elif display or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("QtAgg", force=True)
    except Exception:
        matplotlib.use("TkAgg", force=True)
else:
    matplotlib.use("WebAgg", force=True)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/voyager_prefits",
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
    km.download_all_data_files()
    logger.info("Configuration loaded and runtime initialized successfully.")

    bodies = get_environment(cfg, ctx)
    logger.info("Environment created successfully.")

    # Create observations
    observations, observation_models = create_observation_collection(cfg, bodies)
    logger.info("Observations generated successfully.")

    # Add range dependent variable to compute lighttime post simulation
    # range_setting = obs_dep_var.target_range_between_link_ends_dependent_variable()
    # observations.add_dependent_variable(range_setting)
    # logger.info("Added dependent variables (target_range_between_link_ends) succesfully")

    # Create observation simulators for pre-fit residuals
    ephemeris_observation_simulators = obs_sim_setup.create_observation_simulators(
        observation_models, bodies
    )
    logger.info("Observation simulators created successfully.")

    # Populate residuals in SingleObservationSets
    obs.compute_residuals_and_dependent_variables(
        observations, ephemeris_observation_simulators, bodies
    )
    logger.info("Pre-fit residuals computed successfully.")

    # Log metric summary of residuals to Aim
    rms = np.sqrt(
        np.mean(np.square(observations.get_single_observation_sets()[0].concatenated_residuals))
    )
    if rms.size > 0:
        aim_log_metrics(
            {
                "residuals_rms": float(rms.std()),
                "residuals_mean": float(rms.mean()),
                "residuals_max": float(abs(rms).max()),
                "num_observations": rms.size,
            }
        )
        logger.info("Logged residual summary metrics to Aim.")

    residuals_psd = ResidualsPSD(cfg, observations, 30, cfg.figures.residuals_psd)
    fig_psd, ax_psd = residuals_psd._make_figure()
    residuals = Residuals(cfg, observations)
    fig, ax = residuals._make_figure()

    # add line on y axis at date of voyager 2 closest approach to neptune,
    # which is 1989-8-25T16:00:00
    import datetime

    closest_approach_time = datetime.datetime(1989, 8, 25, 16, 0, 0)

    ax[0].axvline(x=closest_approach_time, color="red", linestyle="--")
    ax[1].axvline(x=closest_approach_time, color="red", linestyle="--", label="Closest Approach")
    ax[1].legend()
    logger.info("Pre-fit residuals plotted successfully.")

    # Save, log to Aim, and attach the PDFs via the Plot base class after the
    # closest-approach lines were added.
    residuals.publish(fig, ax)
    residuals_psd.publish(fig_psd, ax_psd)

    # Also log config.yaml as artifact for this run
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        aim_log_artifact(config_path)

    backend = plt.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        logger.info("Skipping interactive display because matplotlib backend is %s.", backend)
    else:
        if backend == "webagg":
            logger.info("Open the interactive plot at http://localhost:8988")
        plt.show(block=True)

    logger.info("Pre-fit residuals script completed.")


if __name__ == "__main__":
    main()
