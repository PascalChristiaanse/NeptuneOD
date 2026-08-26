"""Compute and visualise pre-fit residuals for Gaia astrometric observations.

Usage:
    python scripts/gaia_prefit_residuals.py
"""

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
from orbitdet.simulation import get_environment
from orbitdet.visualization import Residuals
from orbitdet.visualization import ResidualsPSD
from orbitdet.visualization import ResidualsScan
from orbitdet.visualization import ResidualScanHistogram

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
    config_name="experiments/gaia_prefits",
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
    try:
        residual_sets = observations.get_single_observation_sets()
        concatenated = np.concatenate(
            [np.asarray(obs_set.residuals).flatten() for obs_set in residual_sets]
        )
        if concatenated.size > 0:
            aim_log_metrics(
                {
                    "residuals_rms": float(np.sqrt(np.mean(np.square(concatenated)))),
                    "residuals_mean": float(np.mean(concatenated)),
                    "residuals_max": float(np.abs(concatenated).max()),
                    "num_observations": int(concatenated.size),
                }
            )
            logger.info("Logged residual summary metrics to Aim.")
    except Exception as exc:
        logger.warning("Could not log residual summary metrics: %s", exc)

    Residuals(cfg, observations).plot()
    ResidualsPSD(cfg, observations, 20, cfg.figures.residuals_psd).plot()
    ResidualsScan(cfg.figures, observations).plot()
    ResidualScanHistogram(cfg.figures, observations).plot()

    logger.info("Pre-fit residuals plotted successfully.")

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

    logger.info("Gaia pre-fit residuals script completed.")


    # Observations get loaded correctly (manually verified)
    # Observation times get loaded correctly (manually verified)
    # Gaia ephemeris gets loaded correctly (see aim run, manually verified, J2000 from SSB 
    # (as per config/gaia docs)https://gea.esac.esa.int/archive/documentation/FPR/chap_datamodel/
    # sec_dm_focused_product_release/ssec_dm_sso_observation.html) 

    # Ephemeris doesnt seem to match up with literature (see aim runs favorites/gaiaprefitsresiduals 59fe499)
    # "systematics and refinement... yuan2025"



if __name__ == "__main__":
    main()
