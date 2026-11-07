"""Pipeline stage: build the observation collection and its pre-fit residuals.

This stage does the expensive work and nothing else.  It produces two artifacts
and draws no figures, so a change to a figure title never re-runs it.

Runnable in three ways:

* standalone, with the default config::

      python scripts/prefit_residuals.py

* standalone, against a different config::

      python scripts/prefit_residuals.py --config-name=prefit_residuals

* as a Snakemake rule, which additionally pins the output directory::

      python scripts/prefit_residuals.py hydra.run.dir=results/<stage>

Artifacts written into the Hydra run directory:

``observations.tudat``
    The :class:`~tudatpy.estimation.observations.ObservationCollection` with
    the pre-fit residuals already populated.  Every figure rule and every
    downstream stage consumes this.
``standard_frame.parquet``
    The standard observation frame the collection was built from, carrying the
    ``observation_set_index`` column that links every row back to its set.

Together these two files are enough to rebuild an
:class:`~orbitdet.analysis.ObservationAccess`, which is what the figure rules
in ``scripts/plot_residuals_figure.py`` do.  Nothing is recomputed there.

The stage is self-contained: it furnishes SPICE itself, because kernel state is
process-local and cannot be handed to another process.
"""

import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.analysis import ObservationAccess
from orbitdet.data import KernelManager
from orbitdet.observations import create_observation_collection
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_integrator_settings,
    get_propagator_settings,
)
from orbitdet.utility import save_tudat_object

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="prefit_residuals",
)
@enforce_initialization
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    ctx: RuntimeContext = initialize(cfg)

    # Inject the epochs the propagator and the observation models need.
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
    get_propagator_settings(cfg, ctx, accelerations, integrator, dependent_variables_to_save=[])
    logger.info("Environment and propagator settings created successfully.")

    logger.info("Creating observation collection...")
    observations, observation_models, standard_frame = create_observation_collection(cfg, bodies)
    access = ObservationAccess.build(observations, standard_frame)
    logger.info(
        "Observation collection created with %d observation(s).",
        len(observations.concatenated_times),
    )

    logger.info("Computing pre-fit residuals with respect to SPICE...")
    simulators = obs_sim_setup.create_observation_simulators(observation_models, bodies)
    obs.compute_residuals_and_dependent_variables(observations, simulators, bodies)
    access.refresh()
    logger.info("Pre-fit residuals computed successfully.")

    # -- Artifacts -----------------------------------------------------
    observations_path = save_tudat_object(observations, os.path.join(output_dir, "observations"))
    logger.info("Observations saved to %s", observations_path)

    frame_path = os.path.join(output_dir, "standard_frame.parquet")
    standard_frame.to_parquet(frame_path, index=False)
    logger.info("Standard frame saved to %s", frame_path)

    logger.info("prefit_residuals stage completed.")


if __name__ == "__main__":
    main()
