"""Pipeline stage: write one table from a saved estimation output.

This is the table counterpart of ``scripts/plot_estimation_figure.py``.  It
loads a serialized :class:`~tudatpy.estimation.estimation_analysis.EstimationOutput`,
builds the full per-parameter dataframe from it, and publishes the column
selection the table config asks for.  No estimation is re-run.

Runnable standalone::

    python scripts/write_estimation_table.py \\
        table_config=conf/tables/parameter_change.yaml \\
        estimation_output=results/chapter_3/initial_state_estimation/IAU2015/estimation_output.tudat

or as a Snakemake rule, which passes the same overrides.

The table is written to ``<hydra.run.dir>/<output_file stem>`` as CSV and
Markdown, plus one LaTeX file per style the table config's ``writer`` block
names (``booktabs`` and ``aa`` by default in the shipped configs).

Why one wide frame
------------------
The artifact carries the parameter history, the final parameters, the formal
errors and the covariance.  Rather than have every table re-read the artifact
and recompute the same derived quantities, the script builds **one** dataframe
with everything derivable from it, and the table config selects the columns it
shows.  A new table is therefore a config file, not a class.

Table selection
---------------
Each table config names its producer class with a ``table`` key (e.g.
``table: "EstimationParameterTable"``).  The registry maps that class name to
the class, so one config file per table instance can share a single producer.

Single-fit and multi-fit tables
-------------------------------
Most tables describe one fit and read ``estimation_output``.  A comparison
table describes several, so it reads ``estimation_outputs`` - a mapping of
label to path - and ``observations``, the collection the fits were run against.
Which of the two a producer needs is declared in :data:`MULTI_FIT_TABLES`, so
the script stays free of per-table knowledge.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.analysis.tables import (
    EstimationParameterTable,
    GoodnessOfFitTable,
    TableView,
    attach_table_config,
    build_writers,
    resolve_table_builder,
)
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.utility import load_tudat_object

logger = logging.getLogger(__name__)


#: Every table producer this script can write, keyed by the class name a table
#: config declares with its ``table`` key.  Each class takes
#: ``(estimation_output, **options)`` and its ``._build()`` returns the wide
#: dataframe the config selects from.
TABLE_BUILDERS = {
    "EstimationParameterTable": EstimationParameterTable,
    "GoodnessOfFitTable": GoodnessOfFitTable,
}

#: Producers that compare several fits.  They take a label -> output mapping
#: plus the observation collection, instead of a single estimation output.
MULTI_FIT_TABLES = {"GoodnessOfFitTable"}

#: Keys of a table config that describe the table rather than the producer.
_TABLE_KEYS = ("table", "output_file", "writer", "columns")


def _load_estimation_output(cfg: DictConfig) -> est_an.EstimationOutput:
    """Load the serialized estimation output named by ``cfg.estimation_output``."""
    path = cfg.estimation_output
    logger.info("Loading estimation output from %s", path)
    return load_tudat_object(est_an.EstimationOutput, path)


def _load_estimation_outputs(cfg: DictConfig) -> dict[str, est_an.EstimationOutput]:
    """Load every fit named by ``cfg.estimation_outputs``, keeping the labels."""
    outputs: dict[str, est_an.EstimationOutput] = {}
    for label, path in cfg.estimation_outputs.items():
        logger.info("Loading estimation output '%s' from %s", label, path)
        outputs[str(label)] = load_tudat_object(est_an.EstimationOutput, str(path))
    if not outputs:
        raise ValueError("cfg.estimation_outputs is empty; nothing to compare.")
    return outputs


def _load_observations(cfg: DictConfig):
    """Load the observation collection the fits were run against."""
    from tudatpy.estimation import observations as obs

    path = cfg.observations
    logger.info("Loading observation collection from %s", path)
    return load_tudat_object(obs.ObservationCollection, path)


def _producer_options(table_cfg: DictConfig) -> dict:
    """Return the producer keyword arguments declared by the table config.

    Everything that describes the table itself is removed; the rest is passed
    through, which keeps the script free of per-table knowledge.
    """
    options = dict(table_cfg)
    for key in _TABLE_KEYS:
        options.pop(key, None)
    return options


def _output_stem(cfg: DictConfig, table_cfg: DictConfig, table_name: str) -> str:
    """Return the stem the table's files are written under.

    ``output_name`` on the run config wins, so a rule that writes the same table
    once per model can keep the files apart.  Otherwise the table config's
    ``output_file`` names the output, and the config's own stem is the fallback.
    """
    override = cfg.get("output_name")
    if override:
        return Path(str(override)).stem
    return Path(str(table_cfg.get("output_file", table_name))).stem


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="write_estimation_table",
)
@enforce_initialization
def main(cfg: DictConfig):
    # The table's name is the stem of its config file, so a rule only has to
    # name the file.  There is no global ``table`` key to keep in sync.
    table_name = Path(cfg.table_config).stem

    cfg = attach_table_config(cfg, table_name)
    table_cfg = cfg.tables[table_name]
    builder = resolve_table_builder(TABLE_BUILDERS, table_cfg)

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the table config has been merged in.
    initialize(cfg)

    options = _producer_options(table_cfg)
    if table_cfg.get("table") in MULTI_FIT_TABLES:
        logger.info("Building the comparison frame...")
        frame = builder(
            _load_estimation_outputs(cfg),
            observations=_load_observations(cfg),
            **options,
        )._build()
    else:
        logger.info("Building the estimation parameter frame...")
        frame = builder(_load_estimation_output(cfg), **options)._build()

    logger.info("Writing table '%s'...", table_name)
    TableView(
        frame,
        name=_output_stem(cfg, table_cfg, table_name),
        columns=table_cfg.get("columns"),
        writers=build_writers(table_cfg.get("writer")),
    ).publish()
    logger.info("Table '%s' written to %s", table_name, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
