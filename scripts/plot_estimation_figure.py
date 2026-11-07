"""Pipeline stage: draw one estimation figure from a saved estimation output.

Loads a serialized :class:`~tudatpy.estimation.estimation_analysis.EstimationOutput`
and draws exactly one figure from it.  No estimation is re-run; the covariance,
parameter history and residual history are read straight off the artifact.

Runnable standalone::

    python scripts/plot_estimation_figure.py \\
        figure=parameter_correlation_heatmap \\
        figure_config=conf/figures/parameter_correlation_heatmap.yaml \\
        estimation_output=results/pipeline/chapter_3/estimation_output.tudat

or as a Snakemake rule, which passes the same overrides.

The figure is written to ``<hydra.run.dir>/<figure name>.pdf`` plus a ``.fig``
pickle for interactive inspection.

Figures that need the system of bodies (``covariance_ellipses`` rotates the
covariance into the RSW frame) rebuild the environment from the experiment
config, exactly as the propagation stages do.
"""

import logging

import hydra
from _figure_common import (
    attach_figure_config,
    output_dir,
    resolve_builder,
    select_matplotlib_backend,
    show_or_serve,
)
from omegaconf import DictConfig
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.simulation import get_environment
from orbitdet.utility import load_tudat_object
from orbitdet.visualization import (
    CovarianceEllipses,
    ParameterCorrelationHeatmap,
    ParameterHistoryPerIteration,
    ResidualRMSPerIteration,
)

logger = logging.getLogger(__name__)


#: Every estimation plotter this script can draw, keyed by the class name a
#: figure config declares with its ``plotter`` key.  Each class takes
#: ``(figure_cfg, estimation_output)`` and its ``.plot()`` publishes the figure.
#: ``CovarianceEllipses`` additionally needs the system of bodies and runtime
#: context; the script supplies them when ``requires_environment`` is set.
FIGURE_BUILDERS = {
    "ResidualRMSPerIteration": ResidualRMSPerIteration,
    "ParameterCorrelationHeatmap": ParameterCorrelationHeatmap,
    "ParameterHistoryPerIteration": ParameterHistoryPerIteration,
    "CovarianceEllipses": CovarianceEllipses,
}


def _load_estimation_output(cfg: DictConfig) -> est_an.EstimationOutput:
    """Load the serialized estimation output named by ``cfg.estimation_output``."""
    path = cfg.estimation_output
    logger.info("Loading estimation output from %s", path)
    return load_tudat_object(est_an.EstimationOutput, path)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="plot_estimation_figure",
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
    ctx = initialize(cfg)

    estimation_output = _load_estimation_output(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    if getattr(builder, "requires_environment", False):
        bodies = get_environment(cfg, ctx)
        builder(figure_cfg, estimation_output, bodies, ctx).plot()
    else:
        builder(figure_cfg, estimation_output).plot()
    logger.info("Figure '%s' written to %s", figure_name, output_dir())

    show_or_serve()


if __name__ == "__main__":
    main()
