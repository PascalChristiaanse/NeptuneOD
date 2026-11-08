"""Pipeline stage: propagate a trajectory and save the result.

This stage does the expensive work and nothing else.  It propagates the
configured bodies over the configured time window, saves the resulting
:class:`~tudatpy.dynamics.propagation.SingleArcSimulationResults`, and draws no
figures, so a change to a figure title never re-runs it.

Runnable in three ways:

* standalone, with the default config::

      python scripts/propagate_trajectory.py

* standalone, against a different config::

      python scripts/propagate_trajectory.py --config-name=propagate_trajectory

* as a Snakemake rule, which additionally pins the output directory::

      python scripts/propagate_trajectory.py hydra.run.dir=results/<stage>

Artifact written into the Hydra run directory:

``propagation_result.tudat``
    The :class:`~tudatpy.dynamics.propagation.SingleArcSimulationResults`,
    carrying the state history and the dependent-variable history.  Every
    trajectory figure rule consumes this.

The dependent variables are declared in the config, in order, so the figure
rules can select one by index without the script hard-coding body names.  The
order is preserved in the saved result's ``ordered_dependent_variable_settings``.

The stage is self-contained: it furnishes SPICE itself, because kernel state is
process-local and cannot be handed to another process.
"""

import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.dynamics import simulator

from orbitdet.data import KernelManager
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)
from orbitdet.utility import save_tudat_object

logger = logging.getLogger(__name__)


def _build_dependent_variable(entry: DictConfig):
    """Build one dependent-variable save setting from a config entry.

    The config names the variable with a ``type`` key and the bodies it refers
    to with ``body`` / ``central_body`` / ``secondary_body``.  Keeping the
    mapping here means the config stays declarative and the script stays free of
    hard-coded body names.
    """
    dependent_variable = prop_setup.dependent_variable
    variable_type = entry.type

    match variable_type:
        case "keplerian_state":
            return dependent_variable.keplerian_state(entry.body, entry.central_body)
        case "modified_equinoctial_state":
            return dependent_variable.modified_equinoctial_state(
                entry.body, entry.central_body
            )
        case "relative_position":
            return dependent_variable.relative_position(entry.body, entry.secondary_body)
        case "relative_velocity":
            return dependent_variable.relative_velocity(entry.body, entry.secondary_body)
        case _:
            raise ValueError(
                f"Unknown dependent variable type '{variable_type}'. Known: "
                "keplerian_state, modified_equinoctial_state, relative_position, "
                "relative_velocity."
            )


def _build_dependent_variables(cfg: DictConfig) -> list:
    """Build every dependent-variable save setting, preserving config order."""
    entries = cfg.get("dependent_variables") or []
    return [_build_dependent_variable(entry) for entry in entries]


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="propagate_trajectory",
)
@enforce_initialization
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    ctx: RuntimeContext = initialize(cfg)

    # Inject the epochs the propagator needs.
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

    # Kernel state is process-local, so every stage furnishes SPICE itself.
    logger.info("Loading kernels...")
    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    km.download_all_data_files()
    logger.info("Kernels loaded and furnished successfully.")

    bodies = get_environment(cfg, ctx)
    accelerations = get_dynamical_model(cfg, ctx, bodies)
    integrator = get_integrator_settings(cfg, ctx)

    dependent_variables = _build_dependent_variables(cfg)
    logger.info("Saving %d dependent variable(s).", len(dependent_variables))

    propagator_settings = get_propagator_settings(
        cfg, ctx, accelerations, integrator, dependent_variables_to_save=dependent_variables
    )
    logger.info("Environment and propagator settings created successfully.")

    logger.info("Propagating trajectory...")
    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
    result = dynamics_simulator.propagation_results
    logger.info("Propagation completed with %d state(s).", len(result.state_history))

    # -- Artifact ------------------------------------------------------
    result_path = save_tudat_object(result, os.path.join(output_dir, "propagation_result"))
    logger.info("Propagation result saved to %s", result_path)

    logger.info("propagate_trajectory stage completed.")


if __name__ == "__main__":
    main()
