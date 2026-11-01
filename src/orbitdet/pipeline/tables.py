"""Stage: weight summary tables.

Turns the per-observation weights CSV into the per-source and per-group summary
tables that appear in the report, in three formats (CSV, LaTeX, Excel). This is
a pure function of the weights CSV, so changing a table's formatting re-runs
this rule alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from orbitdet.visualization.weight_summary_table import (
    GROUP_COLUMNS as _GROUP_COLUMNS,
)
from orbitdet.visualization.weight_summary_table import (
    SOURCE_COLUMNS as _SOURCE_COLUMNS,
)
from orbitdet.visualization.weight_summary_table import WeightSummaryTable

logger = logging.getLogger(__name__)

def run_weight_tables(
    cfg: DictConfig,
    weights_path: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Build the per-source and per-group weight summary tables.

    Parameters
    ----------
    cfg:
        The ``figures`` config slice (the table builder reads no styling, but
        takes the config for consistency with the other stages).
    weights_path:
        ``observation_weights.csv`` from the weighting stage. When weighting is
        disabled the file has headers but no rows.
    output_dir:
        Directory this stage writes into.

    Returns
    -------
    list[Path]
        Every file written.

    Notes
    -----
    An empty weights table is not an error. Weighting is configuration, and an
    experiment may legitimately disable it -- the ``weight_tables`` rule still
    runs, because the DAG is unconditional. The stage then writes empty tables
    with the correct schema, so downstream report steps always find parseable
    files and a reader can tell that weighting was off rather than that a step
    was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_df = pd.read_csv(weights_path)

    if weights_df.empty:
        logger.info(
            "Weights table at %s has no rows (weighting disabled for this "
            "experiment); writing empty summary tables.",
            weights_path,
        )
        return _write_empty_tables(output_dir)

    builder = WeightSummaryTable(cfg, weights_df)
    source_table = builder.build_source()
    group_table = builder.build_group()

    written: list[Path] = []

    for table, name in ((source_table, "source"), (group_table, "group")):
        csv_path = output_dir / f"weight_summary_{name}.csv"
        table.to_csv(csv_path, index=False)
        written.append(csv_path)

        tex_path = output_dir / f"weight_summary_{name}.tex"
        tex_path.write_text(
            builder.to_latex_source(table)
            if name == "source"
            else builder.to_latex_group(table)
        )
        written.append(tex_path)

    xlsx_path = output_dir / "weight_summary.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        source_table.to_excel(writer, sheet_name="Per source", index=False)
        group_table.to_excel(writer, sheet_name="Per group", index=False)
    written.append(xlsx_path)

    logger.info("Weight summary tables written: %s", ", ".join(p.name for p in written))
    return written


def _write_empty_tables(output_dir: Path) -> list[Path]:
    """Write schema-correct empty tables, for when weighting is disabled."""
    written: list[Path] = []

    for name, columns in (
        ("source", _SOURCE_COLUMNS),
        ("group", _GROUP_COLUMNS),
    ):
        csv_path = output_dir / f"weight_summary_{name}.csv"
        # Header only: a 0-byte file is *not* an empty DataFrame, it is an
        # unparseable one. ``pd.read_csv`` raises ``EmptyDataError`` on it, so
        # write the real column names and no rows.
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
        written.append(csv_path)

        tex_path = output_dir / f"weight_summary_{name}.tex"
        tex_path.write_text(
            "% Weighting is disabled for this experiment; no summary available.\n"
        )
        written.append(tex_path)

    xlsx_path = output_dir / "weight_summary.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame(columns=_SOURCE_COLUMNS).to_excel(
            writer, sheet_name="Per source", index=False
        )
        pd.DataFrame(columns=_GROUP_COLUMNS).to_excel(
            writer, sheet_name="Per group", index=False
        )
    written.append(xlsx_path)

    logger.info("Empty weight summary tables written (weighting disabled).")
    return written
