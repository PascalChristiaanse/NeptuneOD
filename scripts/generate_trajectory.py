import logging
from pathlib import Path

import hydra
import tudatpy.dynamics.propagation_setup as prop_setup
from hydra.core.hydra_config import HydraConfig
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
    config_name="experiments/classic_triton_state",
)
@enforce_initialization
def main(cfg: DictConfig):
    ctx: RuntimeContext = initialize(cfg)

    # Inject start and end epochs into the runtime context
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

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
        prop_setup.dependent_variable.relative_position("Triton Spice", "Triton"),
        prop_setup.dependent_variable.relative_velocity("Triton Spice", "Triton"),
    ]
    prop = get_propagator_settings(cfg, ctx, acc, integ, dependent_variables_to_save=dep_vars)
    logger.info("Propagator settings created successfully.")

    logger.info("Simulation setup complete. Ready for propagation and estimation.")

    from tudatpy.dynamics import simulator

    result = simulator.create_dynamics_simulator(bodies, prop)
    result.propagation_results.save_to_binary("myresult_atanas_triton_state")

    # ── Plot Keplerian differences (Triton Spice vs Triton) ──
    from orbitdet.visualization import plot_differenced_dependent_variables

    fig_diff_kep, data_diff_kep = plot_differenced_dependent_variables(
        cfg,
        result.propagation_results,
        [result.propagation_results],
        dep_vars[0],
        [dep_vars[1]],
    )

    # ── Plot RSW decomposition ──
    from orbitdet.visualization import plot_RSW_distance

    fig_rsw, data_rsw = plot_RSW_distance(
        cfg,
        result.propagation_results,
        dep_vars[2],
    )

    # ── Plot dependent variables ──
    from orbitdet.visualization import plot_dependent_variable

    fig_dep_spice_kep, axes_dep_spice_kep = plot_dependent_variable(
        cfg, result.propagation_results, dep_vars[0]
    )
    fig_dep_triton_kep, axes_dep_triton_kep = plot_dependent_variable(
        cfg, result.propagation_results, dep_vars[1]
    )
    fig_dep_relpos, axes_dep_relpos = plot_dependent_variable(
        cfg, result.propagation_results, dep_vars[2]
    )
    fig_dep_relvel, axes_dep_relvel = plot_dependent_variable(
        cfg, result.propagation_results, dep_vars[3]
    )

    # Save all figures to the output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_diff_kep_path = output_dir / "keplerian_difference.pdf"
    fig_diff_kep.savefig(fig_diff_kep_path)
    logger.info(f"Keplerian difference plot saved to {fig_diff_kep_path}")

    fig_rsw_path = output_dir / "rsw_distance.pdf"
    fig_rsw.savefig(fig_rsw_path)
    logger.info(f"RSW distance plot saved to {fig_rsw_path}")

    fig_dep_spice_kep_path = output_dir / "dependent_variable_spice_keplerian.pdf"
    fig_dep_spice_kep.savefig(fig_dep_spice_kep_path)
    logger.info(f"Spice Keplerian plot saved to {fig_dep_spice_kep_path}")

    fig_dep_triton_kep_path = output_dir / "dependent_variable_triton_keplerian.pdf"
    fig_dep_triton_kep.savefig(fig_dep_triton_kep_path)
    logger.info(f"Triton Keplerian plot saved to {fig_dep_triton_kep_path}")

    fig_dep_relpos_path = output_dir / "dependent_variable_relative_position.pdf"
    fig_dep_relpos.savefig(fig_dep_relpos_path)
    logger.info(f"Relative position plot saved to {fig_dep_relpos_path}")

    fig_dep_relvel_path = output_dir / "dependent_variable_relative_velocity.pdf"
    fig_dep_relvel.savefig(fig_dep_relvel_path)
    logger.info(f"Relative velocity plot saved to {fig_dep_relvel_path}")


if __name__ == "__main__":
    main()
