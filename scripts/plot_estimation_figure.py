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
from pathlib import Path

import hydra
from _figure_common import (
    apply_figure_style,
    attach_figure_config,
    output_dir,
    resolve_builder,
    select_matplotlib_backend,
    show_or_serve,
)
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.data import KernelManager
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.simulation import get_environment
from orbitdet.utility import load_tudat_object
from orbitdet.visualization import (
    CovarianceCorrelationPairGrid,
    CovarianceEllipses,
    ParameterCorrelationHeatmap,
    ParameterHistoryPerIteration,
    ResidualRMSPerIteration,
)

logger = logging.getLogger(__name__)


#: Every estimation plotter this script can draw, keyed by the class name a
#: figure config declares with its ``plotter`` key.  Each class takes
#: ``(figure_cfg, estimation_output)`` and its ``.plot()`` publishes the figure.
#: ``CovarianceEllipses`` and ``CovarianceCorrelationPairGrid`` additionally
#: need the system of bodies and runtime context; the script supplies them when
#: ``requires_environment`` is set.
FIGURE_BUILDERS = {
    "ResidualRMSPerIteration": ResidualRMSPerIteration,
    "ParameterCorrelationHeatmap": ParameterCorrelationHeatmap,
    "ParameterHistoryPerIteration": ParameterHistoryPerIteration,
    "CovarianceEllipses": CovarianceEllipses,
    "CovarianceCorrelationPairGrid": CovarianceCorrelationPairGrid,
}


def _load_estimation_output(cfg: DictConfig) -> est_an.EstimationOutput:
    """Load the serialized estimation output named by ``cfg.estimation_output``."""
    path = cfg.estimation_output
    logger.info("Loading estimation output from %s", path)
    return load_tudat_object(est_an.EstimationOutput, path)


def _inject_epochs(cfg: DictConfig, ctx) -> None:
    """Attach the run's epochs to the runtime context.

    :func:`orbitdet.reproducibility.initialize` returns a context that only
    carries the reproducibility bookkeeping (commit, output directory, seed).
    The epochs live in the run configuration, so every stage that needs them
    copies them across; the environment builders and the RSW rotation both read
    ``ctx.start_epoch``.
    """
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)


def _furnish_kernels(cfg: DictConfig) -> None:
    """Furnish SPICE kernels before the environment is built.

    A figure that rotates its covariance into the RSW frame rebuilds the system
    of bodies, which resolves ephemerides from the kernel pool.  Loading is
    cheap and idempotent, so it is done for every environment-needing figure.
    """
    logger.info("Furnishing SPICE kernels...")
    km = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="plot_estimation_figure",
)
@enforce_initialization
def main(cfg: DictConfig):
    select_matplotlib_backend()

    # The figure's name is the stem of its config file, so a rule only has to
    # name the file.  There is no global ``figure`` key to keep in sync.
    figure_name = Path(cfg.figure_config).stem

    cfg = attach_figure_config(cfg, figure_name)
    figure_cfg = cfg.figures[figure_name]
    builder = resolve_builder(FIGURE_BUILDERS, figure_cfg)

    # The journal style must be applied before any figure is built, so it runs
    # once the figure's own config (which may name a style) is merged in.
    apply_figure_style(cfg, figure_cfg)

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the figure config has been merged in.
    ctx = initialize(cfg)

    estimation_output = _load_estimation_output(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    if getattr(builder, "requires_environment", False):
        # The RSW-frame plotters rebuild the system of bodies, so they need the
        # epochs on the context and the kernels in the pool first.
        _inject_epochs(cfg, ctx)
        _furnish_kernels(cfg)
        bodies = get_environment(cfg, ctx)
        builder(figure_cfg, estimation_output, bodies, ctx).plot()
    else:
        builder(figure_cfg, estimation_output).plot()
    logger.info("Figure '%s' written to %s", figure_name, output_dir())

    show_or_serve(cfg)


if __name__ == "__main__":
    main()
