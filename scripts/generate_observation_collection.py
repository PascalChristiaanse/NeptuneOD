"""Pipeline stage: build an observation collection and its standard frame.

This stage turns a collection config into the two artifacts every later stage
consumes.  It computes no residuals and draws no figures, so a change to a
residual model or a figure title never re-runs it.

Runnable in three ways:

* standalone, with the default config::

      python scripts/generate_observation_collection.py

* standalone, against a different config::

      python scripts/generate_observation_collection.py \\
          --config-name=generate_observation_collection

* as a Snakemake rule, which additionally pins the output directory::

      python scripts/generate_observation_collection.py hydra.run.dir=results/<stage>

Artifacts written into the Hydra run directory:

``observations.tudat``
    The :class:`~tudatpy.estimation.observations.ObservationCollection`, with
    its observations but no residuals.  The residual stage loads it, computes
    the residuals in place and writes it back.
``standard_frame.parquet``
    The standard observation frame the collection was built from, carrying the
    ``observation_set_index`` column that links every row back to its set.

Both names are configurable through ``output.observations`` and
``output.standard_frame``, so a stage can label its artifacts for what they are
- the same collection is a pre-fit input in one stage and a post-fit input in
another::

      python scripts/generate_observation_collection.py \\
          output.observations=prefit_observations \\
          output.standard_frame=prefit_standard_frame.parquet

Together the two files are enough to rebuild an
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

from orbitdet.data import KernelManager
from orbitdet.observations import create_observation_collection
from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.simulation import get_environment
from orbitdet.utility import save_tudat_object

logger = logging.getLogger(__name__)

# Default artifact names, used when the config does not override them.
DEFAULT_OBSERVATIONS_NAME = "observations"
DEFAULT_STANDARD_FRAME_NAME = "standard_frame.parquet"


def _output_paths(cfg: DictConfig, output_dir: str) -> tuple[str, str]:
    """Resolve the two artifact paths from the config.

    ``output.observations`` and ``output.standard_frame`` name the files written
    into the Hydra run directory.  A trailing ``.tudat`` on the observations name
    is accepted and stripped, because TudatPy's ``save_to_binary`` appends that
    extension itself - so ``prefit_observations`` and
    ``prefit_observations.tudat`` both produce ``prefit_observations.tudat``.

    Args:
        cfg: The run config.
        output_dir: The Hydra run directory the artifacts are written into.

    Returns:
        ``(observations_path, standard_frame_path)``, both absolute.
    """
    output_cfg = cfg.get("output") or {}
    observations_name = str(output_cfg.get("observations", DEFAULT_OBSERVATIONS_NAME))
    if observations_name.endswith(".tudat"):
        observations_name = observations_name[: -len(".tudat")]
    standard_frame_name = str(
        output_cfg.get("standard_frame", DEFAULT_STANDARD_FRAME_NAME)
    )
    return (
        os.path.join(output_dir, observations_name),
        os.path.join(output_dir, standard_frame_name),
    )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="generate_observation_collection",
)
@enforce_initialization
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir

    ctx: RuntimeContext = initialize(cfg)

    # Inject the epochs the observation models need.
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
    logger.info("Environment built.")

    logger.info("Creating observation collection...")
    observations, _, standard_frame = create_observation_collection(cfg, bodies)
    logger.info(
        "Observation collection created with %d observation(s) in %d set(s).",
        len(observations.concatenated_times),
        len(observations.get_single_observation_sets()),
    )

    # -- Artifacts -----------------------------------------------------
    observations_path, frame_path = _output_paths(cfg, output_dir)

    saved_path = save_tudat_object(observations, observations_path)
    logger.info("Observations saved to %s", saved_path)

    standard_frame.to_parquet(frame_path, index=False)
    logger.info("Standard frame saved to %s", frame_path)

    logger.info("generate_observation_collection stage completed.")


if __name__ == "__main__":
    main()
