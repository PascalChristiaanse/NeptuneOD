"""Pipeline stage: draw the specification curve of one parameter-combination sweep.

The figure is the truth table and the outcome of a sweep, aligned row by row:
the top panel marks which parameters each fit estimated, the bottom panel shows
that fit's residual RMS, sorted so the curve is monotone.  Fits that did not
converge are kept and marked, because with a sweep where most combinations fail,
*which* combinations fail is itself a result.

The input is the comparison table written by
``scripts/analyse_parameter_sweep.py``, not the estimation outputs: the table
already carries the truth-table columns, the convergence verdict and the RMS, so
the figure and the table cannot disagree about what a fit did.

Runnable standalone::

    python scripts/plot_sweep_figure.py \\
        figure_config=conf/figures/specification_curve_gravity.yaml \\
        sweep_table=results/chapter_3/parameter_sweeps/gravity/gravity_sweep_comparison.csv

or as a Snakemake rule, which passes the same overrides.

The figure is written to ``<hydra.run.dir>/<output_file stem>.pdf`` plus a
``.fig`` pickle for interactive inspection.  The stem comes from the figure
config's ``output_file``, so the rule's declared output and the file the plotter
writes cannot drift apart.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
from _figure_common import (
    apply_figure_style,
    attach_figure_config,
    output_dir,
    resolve_builder,
    select_matplotlib_backend,
    show_or_serve,
)
from omegaconf import DictConfig

from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.visualization import SpecificationCurve

logger = logging.getLogger(__name__)


#: Every sweep plotter this script can draw, keyed by the class name a figure
#: config declares with its ``plotter`` key.  Each class takes
#: ``(figure_cfg, frame)`` and its ``.plot()`` publishes the figure.
FIGURE_BUILDERS = {
    "SpecificationCurve": SpecificationCurve,
}


def _load_sweep_table(cfg: DictConfig) -> pd.DataFrame:
    """Load the comparison table named by ``cfg.sweep_table``.

    The truth-table columns are read back as booleans: a CSV round-trip turns
    them into strings, and the plotter draws a dot where a toggle is on.
    """
    path = Path(str(cfg.sweep_table))
    logger.info("Loading sweep comparison table from %s", path)
    frame = pd.read_csv(path)

    for column in cfg.get("boolean_columns") or []:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.lower().isin(("true", "1", "yes"))
    return frame


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="plot_sweep_figure",
)
@enforce_initialization
def main(cfg: DictConfig):
    select_matplotlib_backend()

    # The figure's name is the stem of its config file, so a rule only has to
    # name the file.  There is no global ``figure`` key to keep in sync.
    figure_name = Path(str(cfg.figure_config)).stem

    cfg = attach_figure_config(cfg, figure_name)
    figure_cfg = cfg.figures[figure_name]
    builder = resolve_builder(FIGURE_BUILDERS, figure_cfg)

    # The journal style must be applied before any figure is built, so it runs
    # once the figure's own config (which may name a style) is merged in.
    apply_figure_style(cfg, figure_cfg)

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the figure config has been merged in.
    initialize(cfg)

    frame = _load_sweep_table(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    builder(figure_cfg, frame).plot()
    logger.info("Figure '%s' written to %s", figure_name, output_dir())

    show_or_serve(cfg)


if __name__ == "__main__":
    main()
