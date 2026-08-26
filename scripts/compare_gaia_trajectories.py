import logging
from pathlib import Path

import hydra
import pandas as pd
import tudatpy.dynamics.propagation_setup as prop_setup
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object

from orbitdet.data import KernelManager
from orbitdet.reproducibility import (
    RuntimeContext,
    enforce_initialization,
    initialize,
)
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
    config_name="experiments/gaia_eph_check",
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
        prop_setup.dependent_variable.keplerian_state("Gaia", "Sun"),
        prop_setup.dependent_variable.keplerian_state("Gaia Spice", "Sun"),
        prop_setup.dependent_variable.relative_position("Gaia Spice", "Gaia"),
        prop_setup.dependent_variable.relative_velocity("Gaia Spice", "Gaia"),
    ]
    prop = get_propagator_settings(cfg, ctx, acc, integ, dependent_variables_to_save=dep_vars)
    logger.info("Propagator settings created successfully.")

    logger.info("Simulation setup complete. Ready for propagation and estimation.")

    from tudatpy.dynamics import simulator

    result = simulator.create_dynamics_simulator(bodies, prop)
    result.propagation_results.save_to_binary("myresult_atanas_triton_state")

    # ── Plot Keplerian differences (Triton Spice vs Triton) ──
    from orbitdet.visualization import DifferencedDependentVariables

    # Only evaluate the difference at the epochs at which we actually have Gaia
    # observations. The archived Gaia kernel is a tabulated ephemeris with sparse
    # support points; evaluating the difference continuously vs. the (interpolated)
    # ESA kernel shows mainly interpolator error rather than the physical
    # discrepancy at the observation epochs. The differences are interpolated
    # onto the observation epochs and plotted as a scatter.
    gaia_cache = Path(cfg.bodies_to_create.Gaia.ephemeris.cache_file)
    gaia_table = pd.read_pickle(gaia_cache)
    observation_epochs = gaia_table["epoch"].to_numpy()

    fig_diff_kep, data_diff_kep = DifferencedDependentVariables(
        cfg,
        result.propagation_results,
        [result.propagation_results],
        dep_vars[0],
        [dep_vars[1]],
        epochs=observation_epochs,
        scatter=True,
    ).plot()

    # ── Plot RSW decomposition ──
    from orbitdet.visualization import RSWDistance

    fig_rsw, data_rsw = RSWDistance(
        cfg,
        result.propagation_results,
        dep_vars[2],
        epochs=observation_epochs,
        scatter=True,
    ).plot()

    # ── Plot dependent variables ──
    from orbitdet.visualization import DependentVariable

    fig_dep_spice_kep, axes_dep_spice_kep = DependentVariable(
        cfg, result.propagation_results, dep_vars[0]
    ).plot()
    fig_dep_triton_kep, axes_dep_triton_kep = DependentVariable(
        cfg, result.propagation_results, dep_vars[1]
    ).plot()
    fig_dep_relpos, axes_dep_relpos = DependentVariable(
        cfg, result.propagation_results, dep_vars[2]
    ).plot()
    fig_dep_relvel, axes_dep_relvel = DependentVariable(
        cfg, result.propagation_results, dep_vars[3]
    ).plot()

    # -- Plot RSW distance
    from orbitdet.visualization import RSWDistance

    RSWDistance(
        cfg,
        result.propagation_results,
        dep_vars[2],
        central_body="SSB",
        epochs=observation_epochs,
        scatter=True,
    ).plot()

    # Save all figures to the output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The Plot base class already saved each figure as a PDF, logged it to Aim
    # as a static image, and attached the PDF as an artifact reference.
    logger.info("Figures saved, logged to Aim, and artifacts attached by Plot base class.")

    # show figures
    import matplotlib.pyplot as plt
    plt.show(block=True)

if __name__ == "__main__":
    main()
