"""Pipeline stage: compare the fits of one parameter-combination sweep.

Reads the results folder of one sweep group (``gravity`` or ``pole``), one
sub-directory per specification, and writes the group's comparison table: one
row per specification, with the truth table, the convergence verdict and the
residual statistics.

The truth table comes from the sweep's CSV, the same file the rule enumerates
with.  A fit directory with no output is still listed, with ``converged`` false
and NaN statistics, so a missing fit is visible rather than absent.

Runnable standalone::

    python scripts/analyse_parameter_sweep.py \\
        sweep_group=gravity \\
        sweep_dir=results/chapter_3/parameter_sweeps/gravity \\
        sweep_csv=conf/sweeps/gravity.csv

or as a Snakemake rule, which passes the same overrides.

Artifacts written into the Hydra run directory:

``<group>_sweep_comparison.csv`` / ``.md``
    The comparison table, one row per specification.
``<group>_sweep_comparison_<style>.tex``
    The same table in each LaTeX style the table config names.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.analysis.tables import (
    SweepComparisonTable,
    TableView,
    attach_table_config,
    build_writers,
    summarise_fit,
)
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.utility import load_tudat_object

logger = logging.getLogger(__name__)


def _read_specifications(sweep_csv: Path) -> list[dict]:
    """Return the sweep's specifications, one per CSV row.

    ``slug`` is the row's position, which is what the table shows; ``directory``
    is the fit's folder, built the way the rule's ``Paramspace`` builds it.
    """
    frame = pd.read_csv(sweep_csv)
    return [
        {
            "slug": f"{index:02d}",
            "directory": "_".join(f"{column}-{value}" for column, value in row.items()),
            **{column: bool(value) for column, value in row.items()},
        }
        for index, row in enumerate(frame.to_dict("records"))
    ]


def _load_fit(sweep_dir: Path, slug: str):
    """Load one specification's estimation output and solver log.

    Returns
    -------
    tuple[est_an.EstimationOutput, str] | None
        The output and the log text, or ``None`` when the fit produced no
        output.  A missing log is not fatal: the fit is then reported as not
        converged, which is the conservative reading.
    """
    output_path = sweep_dir / slug / "estimation_output.tudat"
    if not output_path.exists():
        logger.warning("No estimation output for '%s' at %s", slug, output_path)
        return None

    logger.info("Loading estimation output for '%s' from %s", slug, output_path)
    output = load_tudat_object(est_an.EstimationOutput, str(output_path))

    log_path = sweep_dir / slug / "estimation_log.log"
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
    else:
        logger.warning(
            "No solver log for '%s' at %s; reporting it as not converged", slug, log_path
        )
        log_text = ""
    return output, log_text


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="analyse_parameter_sweep",
)
@enforce_initialization
def main(cfg: DictConfig):
    group_name = str(cfg.sweep_group)
    sweep_dir = Path(str(cfg.sweep_dir))

    table_name = Path(str(cfg.table_config)).stem
    cfg = attach_table_config(cfg, table_name)
    table_cfg = cfg.tables[table_name]

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the table config has been merged in.
    initialize(cfg)

    specifications = _read_specifications(Path(str(cfg.sweep_csv)))
    rows = []
    for specification in specifications:
        loaded = _load_fit(sweep_dir, specification["directory"])
        if loaded is None:
            continue
        output, log_text = loaded
        row = summarise_fit(specification, output, log_text)
        logger.info(
            "%s: converged=%s rms=%.4e",
            specification["slug"],
            row["converged"],
            row["rms"],
        )
        rows.append(row)

    table = SweepComparisonTable(specifications, rows)
    frame = table._build()
    n_converged = int(frame["converged"].sum())
    logger.info(
        "%d of %d specifications converged in the '%s' sweep.",
        n_converged,
        len(frame),
        group_name,
    )

    output_name = cfg.get("output_name") or table.name

    TableView(
        frame,
        name=str(output_name),
        columns=table_cfg.get("columns"),
        writers=build_writers(table_cfg.get("writer")),
    ).publish()
    logger.info("Sweep comparison written to %s", HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
