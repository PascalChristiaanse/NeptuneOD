"""Pipeline stage: draw one residual figure from a saved observation collection.

This is the cheap half of the pipeline.  It loads the artifacts a previous stage
wrote, rebuilds the analysis access layer, and draws exactly one figure.  No
propagation, no residual computation, no SPICE beyond what the observation
models need to be rebuilt.

Runnable in three ways:

* standalone, with the default config and the first registered figure::

      python scripts/plot_residuals_figure.py

* standalone, naming the figure, its config, and the artifacts to read::

      python scripts/plot_residuals_figure.py \\
          figure=prefit_residuals_spice \\
          figure_config=conf/figures/prefit_residuals_spice.yaml \\
          observations=results/pipeline/chapter_3/prefit_residuals/observations.tudat \\
          standard_frame=results/pipeline/chapter_3/prefit_residuals/standard_frame.parquet

* as a Snakemake rule, which passes all four as overrides.

The figure is written to ``<hydra.run.dir>/<figure name>.pdf`` plus a ``.fig``
pickle for interactive inspection, matching every other plotter in the project.

Why one figure per process
--------------------------
Splitting the figures into separate rules means editing a title, a colour map or
a legend position re-runs one cheap job instead of the whole generation stage.
The cost is that each figure pays for loading the collection and rebuilding the
access layer, which is milliseconds against a propagation that takes minutes.

Why a registry
--------------
Each figure config names its plotter class with a ``plotter`` key (e.g.
``plotter: "Residuals"``).  The registry maps that class name to the class, so
one config file per figure instance (``prefit_residuals_spice``,
``postfit_residuals``, ...) can share a single plotter without a hand-maintained
name -> class table.  Every residual plotter shares the same
``(figure_cfg, access)`` constructor.  The estimation and simulation families
live in their own scripts (``plot_estimation_figure.py`` and
``plot_simulation_figure.py``) because they load different artifacts.
"""

import logging

import hydra
import pandas as pd
from _figure_common import (
    attach_figure_config,
    output_dir,
    resolve_builder,
    select_matplotlib_backend,
    show_or_serve,
)
from omegaconf import DictConfig
from tudatpy.estimation import observations as obs

from orbitdet.analysis import ObservationAccess
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.utility import load_tudat_object
from orbitdet.visualization import (
    GaiaTransitErrorComponents,
    MahalanobisResiduals,
    NormalizedResiduals,
    NormalizedResidualScan,
    ResidualHistogram,
    ResidualQQ,
    Residuals,
    ResidualScanHistogram,
    ResidualScatter,
    ResidualsPSD,
    ResidualsScan,
    SymmetricWhitenedResiduals,
    WhitenedResidualScatter,
)

logger = logging.getLogger(__name__)


#: Every residual plotter this script can draw, keyed by the class name a figure
#: config declares with its ``plotter`` key.  Each class takes
#: ``(figure_cfg, access)`` - the figure's own config block and the analysis
#: access layer - and its ``.plot()`` publishes the figure.
FIGURE_BUILDERS = {
    "Residuals": Residuals,
    "ResidualsScan": ResidualsScan,
    "NormalizedResiduals": NormalizedResiduals,
    "NormalizedResidualScan": NormalizedResidualScan,
    "ResidualScatter": ResidualScatter,
    "ResidualHistogram": ResidualHistogram,
    "ResidualScanHistogram": ResidualScanHistogram,
    "ResidualQQ": ResidualQQ,
    "ResidualsPSD": ResidualsPSD,
    "WhitenedResidualScatter": WhitenedResidualScatter,
    "SymmetricWhitenedResiduals": SymmetricWhitenedResiduals,
    "MahalanobisResiduals": MahalanobisResiduals,
    "GaiaTransitErrorComponents": GaiaTransitErrorComponents,
}


def _load_access(cfg: DictConfig) -> ObservationAccess:
    """Rebuild the analysis access layer from the artifacts on disk.

    The collection carries the residuals; the standard frame carries the
    provenance columns and the ``observation_set_index`` link between them.
    Rebuilding the access layer from these two files is what lets a figure rule
    run without repeating the generation stage.
    """
    observations_path = cfg.observations
    standard_frame_path = cfg.standard_frame

    logger.info("Loading observation collection from %s", observations_path)
    collection = load_tudat_object(obs.ObservationCollection, observations_path)

    logger.info("Loading standard frame from %s", standard_frame_path)
    standard_frame = pd.read_parquet(standard_frame_path)

    return ObservationAccess.build(collection, standard_frame)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="plot_residuals_figure",
)
@enforce_initialization
def main(cfg: DictConfig):
    select_matplotlib_backend()

    figure_name = cfg.figure

    cfg = attach_figure_config(cfg, figure_name)
    figure_cfg = cfg.figures[figure_name]
    builder = resolve_builder(FIGURE_BUILDERS, figure_cfg)

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the figure config has been merged in.
    initialize(cfg)

    access = _load_access(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    builder(figure_cfg, access).plot()
    logger.info("Figure '%s' written to %s", figure_name, output_dir())

    show_or_serve()


if __name__ == "__main__":
    main()
