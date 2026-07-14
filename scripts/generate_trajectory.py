import logging

import hydra
import tudatpy.dynamics.propagation_setup as prop_setup
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object

from orbitdet.data import KernelManager
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
        prop_setup.dependent_variable.keplerian_state("Triton Spice", "Neptune"),
        prop_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
    ]
    prop = get_propagator_settings(cfg, ctx, acc, integ, dependent_variables_to_save=dep_vars)
    logger.info("Propagator settings created successfully.")

    logger.info("Simulation setup complete. Ready for propagation and estimation.")

    from tudatpy.dynamics import simulator

    result = simulator.create_dynamics_simulator(bodies, prop)

    from orbitdet.visualization import plot_differenced_dependent_variables

    fig, data = plot_differenced_dependent_variables(
        cfg,
        result.propagation_results,
        [result.propagation_results],
        dep_vars[0],
        [dep_vars[1]],
    )
    # Save the figure to the output directory
    from pathlib import Path

    from hydra.core.hydra_config import HydraConfig

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_psd_path = output_dir / "prefit_residuals_psd.pdf"
    # fig_psd.savefig(fig_psd_path)
    logger.info(f"Pre-fit residual PSD plot saved to {fig_psd_path}")

    fig_path = output_dir / "prefit_residuals.pdf"
    fig.savefig(fig_path)

    fig.show()

    import matplotlib.pyplot as plt
    import numpy as np

    state_history: dict[float, np.ndarray] = result.state_history

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
