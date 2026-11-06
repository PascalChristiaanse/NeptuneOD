"""Pipeline stage: draw one figure from a saved observation collection.

This is the cheap half of the pipeline.  It loads the artifacts a previous stage
wrote, rebuilds the analysis access layer, and draws exactly one figure.  No
propagation, no residual computation, no SPICE beyond what the observation
models need to be rebuilt.

Runnable in three ways:

* standalone, with the default config and the first registered figure::

      python scripts/plot_figure.py

* standalone, naming the figure, its config, and the artifacts to read::

      python scripts/plot_figure.py \\
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
"""

"""
This script is dumb as fuck.
Lets also decentralize the plot figure script. again, one change and everything gets rebuilt. 
his is due to the registry, which is a poor solution. the figure configuration should specify which 
class to use, what kwargs to use to call it, and what parameters are actually 
passed through with cfg. 
Fix later
"""

import logging
import os

import hydra
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tudatpy.estimation import observations as obs

from orbitdet.analysis import ObservationAccess
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.utility import load_tudat_object
from orbitdet.visualization import ResidualHistogram, Residuals, ResidualScatter

logger = logging.getLogger(__name__)


def _select_matplotlib_backend() -> None:
    """Pick an interactive backend when a display exists, else WebAgg.

    Mirrors the selection used by the other plotting scripts so a figure run
    from a terminal behaves the same as one run from the debugger.
    """
    display = os.environ.get("DISPLAY")
    is_headless_display = display in (":99", "localhost:99", "127.0.0.1:99")

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


#: Every figure this script can draw, keyed by the name used in ``cfg.figures``
#: and on the command line.  Each entry is a callable taking ``(cfg, access)``
#: and returning a plotter whose ``.plot()`` publishes the figure.
FIGURE_BUILDERS = {
    "prefit_residuals_spice": lambda cfg, access: Residuals(
        cfg, access, "prefit_residuals_spice", group_by="source_set_id"
    ),
    "residual_scatter": lambda cfg, access: ResidualScatter(
        cfg, access, group_by="observatory_code"
    ),
    "residual_histogram": lambda cfg, access: ResidualHistogram(
        cfg, access, group_by="observable_type"
    ),
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


def _attach_figure_config(cfg: DictConfig, figure_name: str) -> DictConfig:
    """Merge the figure's own config file into ``cfg.figures.<figure_name>``.

    The plotters read their styling from ``cfg.figures.<name>``.  Rather than
    compose every figure config into one experiment file, each figure rule
    passes the path of the single file it needs and it is merged in here.  That
    keeps a figure's styling next to the rule that draws it, and means editing
    one figure cannot invalidate another.
    """
    path = cfg.figure_config
    logger.info("Loading figure config from %s", path)

    figure_cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    cfg.figures = {figure_name: figure_cfg}
    OmegaConf.set_struct(cfg, True)

    return cfg


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="plot_figure",
)
@enforce_initialization
def main(cfg: DictConfig):
    _select_matplotlib_backend()

    figure_name = cfg.figure
    builder = FIGURE_BUILDERS.get(figure_name)
    if builder is None:
        raise ValueError(f"Unknown figure '{figure_name}'. Available: {sorted(FIGURE_BUILDERS)}.")

    cfg = _attach_figure_config(cfg, figure_name)

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the figure config has been merged in.
    initialize(cfg)

    access = _load_access(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    builder(cfg, access).plot()
    logger.info("Figure '%s' written to %s", figure_name, HydraConfig.get().runtime.output_dir)

    backend = plt.get_backend().lower()
    if backend == "webagg":
        logger.info("Open the interactive plot at http://localhost:8988")
    elif backend not in ("agg",) and "inline" not in backend:
        plt.show(block=True)


if __name__ == "__main__":
    main()
