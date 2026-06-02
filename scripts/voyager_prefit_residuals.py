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
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.dynamics import simulator as sim
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.data import KernelManager
from orbitdet.observations.collection import create_observation_collection
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)
from orbitdet.visualization import plot_residuals, plot_residuals_psd

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
    integrator = get_integrator_settings(cfg, ctx)
    model = get_dynamical_model(cfg, ctx, bodies)
    dep_vars = [
        prop_setup.dependent_variable.relative_position("Voyager 2", "Neptune Barycenter"),
        prop_setup.dependent_variable.relative_position("Voyager 2 spice", "Neptune Barycenter"),
    ]
    propagator = get_propagator_settings(cfg, ctx, model, integrator, dep_vars)
    logger.info("Environment created successfully.")

    result: sim.SingleArcSimulator = sim.create_dynamics_simulator(bodies, propagator)

    # Plot uninterpolated state history extracted from ancillary files to inspect for
    # outliersdataset_name = str(getattr(settings.ephemeris, "source_dataset", "voyager"))
    voyager_cfg = getattr(getattr(cfg, "datasets", None), "voyager", None)
    from orbitdet.data.voyager_data import build_voyager_tabulated_state_history

    state_history = build_voyager_tabulated_state_history(cfg, voyager_cfg)

    # plot log of x y and z over time for voyager 2 and voyager 2 spice to verify they are the same
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].scatter(
        list(state_history.keys()),
        np.array(list(state_history.values()))[:, 0],
        label="Voyager 2 x",
        s=10,
        marker="x",
    )
    ax[0].scatter(
        list(state_history.keys()),
        np.array(list(state_history.values()))[:, 1],
        label="Voyager 2 y",
        s=10,
        marker="x",
    )
    ax[0].scatter(
        list(state_history.keys()),
        np.array(list(state_history.values()))[:, 2],
        label="Voyager 2 z",
        s=10,
        marker="x",
    )
    ax[0].plot(
        list(result.dependent_variable_history.keys()),
        np.array(list(result.dependent_variable_history.values()))[:, 0],
        label="Voyager 2 x",
    )
    ax[0].plot(
        list(result.dependent_variable_history.keys()),
        np.array(list(result.dependent_variable_history.values()))[:, 1],
        label="Voyager 2 y",
    )
    ax[0].plot(
        list(result.dependent_variable_history.keys()),
        np.array(list(result.dependent_variable_history.values()))[:, 2],
        label="Voyager 2 z",
    )
    ax[0].plot(
        list(result.dependent_variable_history.keys()),
        np.array(list(result.dependent_variable_history.values()))[:, 3],
        label="Voyager 2 spice x",
        linestyle="dashed",
    )
    ax[0].plot(
        list(result.dependent_variable_history.keys()),
        np.array(list(result.dependent_variable_history.values()))[:, 4],
        label="Voyager 2 spice y",
        linestyle="dashed",
    )
    ax[0].plot(
        list(result.dependent_variable_history.keys()),
        np.array(list(result.dependent_variable_history.values()))[:, 5],
        label="Voyager 2 spice z",
        linestyle="dashed",
    )
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (m)")
    ax[0].set_title("Position of Voyager 2 and Voyager 2 spice over time")

    ax[1].plot(
        list(result.dependent_variable_history.keys()),
        np.linalg.norm(
            np.array(list(result.dependent_variable_history.values()))[:, :3]
            - np.array(list(result.dependent_variable_history.values()))[:, 3:],
            axis=1,
        ),
        label="Difference in position between Voyager 2 and Voyager 2 spice",
    )
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Position difference (m)")
    ax[1].set_yscale("log")
    ax[1].set_title("Difference in position between Voyager 2 and Voyager 2 spice over time")
    ax[1].legend()
    # plt.show()

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

    fig, ax = plot_residuals(cfg, observations)
    fig, ax = plot_residuals_psd(cfg, observations, 30, cfg.figures.residuals_psd)
    # add line on y axis at date of voyager 2 closest approach to neptune,
    # which is 1989-8-25T16:00:00
    import datetime

    closest_approach_time = datetime.datetime(1989, 8, 25, 16, 0, 0)

    ax[0].axvline(x=closest_approach_time, color="red", linestyle="--")
    ax[1].axvline(x=closest_approach_time, color="red", linestyle="--", label="Closest Approach")
    ax[1].legend()
    logger.info("Pre-fit residuals plotted successfully.")

    # Save the figure to the output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "prefit_residuals.pdf"
    fig.savefig(fig_path)
    logger.info(f"Pre-fit residuals plot saved to {fig_path}")

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
