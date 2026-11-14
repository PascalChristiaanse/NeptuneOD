"""A configured selection of columns from a wider dataframe.

The estimation tables build one wide dataframe holding everything derivable from
an artifact, and a table config then says which of those columns to show, under
which headers, in which order.  :class:`TableView` is that selection: it holds
the frame, the chosen columns and the writer, and publishes the result.

Keeping the selection separate from the producer means a new table is a config
file, not a class: the same wide frame can be published as a parameter-change
table, a formal-error table or a correlation table without recomputing anything.

Example
-------
::

    from orbitdet.analysis.tables import TableView

    view = TableView(
        frame,
        name="parameter_change",
        columns=["parameter", "initial", "final", "relative_change_percent"],
        writers={"booktabs": LatexTableWriter(caption="Parameter changes.")},
    )
    view.publish()
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from hydra.core.hydra_config import HydraConfig

from orbitdet.analysis.tables.latex import LatexTableWriter

logger = logging.getLogger(__name__)

__all__ = ["TableView"]


class TableView:
    """Publish a configured column selection of a dataframe.

    Parameters
    ----------
    frame : pd.DataFrame
        The wide dataframe the view selects from.
    name : str
        Base name for the published files.
    columns : Sequence[str], optional
        Columns to show, in this order.  Defaults to every column of the frame.
        Names the frame does not carry are dropped with a warning, so a config
        can name a column an older artifact does not produce.
    writers : Mapping[str, LatexTableWriter], optional
        LaTeX renderers keyed by style name.  Each style is written to its own
        file, so one selection can be published in several journal formats.
        Empty publishes only CSV and Markdown.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        name: str,
        columns: Sequence[str] | None = None,
        writers: Mapping[str, LatexTableWriter] | None = None,
    ):
        self.frame = frame
        self.name = name
        self.columns = list(columns) if columns is not None else None
        self.writers = dict(writers or {})

    def select(self) -> pd.DataFrame:
        """Return the configured column selection of the frame.

        Returns
        -------
        pd.DataFrame
            The selected columns, in the configured order.
        """
        if self.columns is None:
            return self.frame

        missing = [column for column in self.columns if column not in self.frame.columns]
        if missing:
            logger.warning(
                "TableView '%s': dropping columns the frame does not carry: %s",
                self.name,
                ", ".join(missing),
            )
        present = [column for column in self.columns if column in self.frame.columns]
        return self.frame[present]

    def publish(self) -> pd.DataFrame:
        """Write the selection as CSV, Markdown and one LaTeX file per style.

        A single style writes ``<name>.tex``; several write
        ``<name>_<style>.tex``, so the files cannot overwrite each other.

        Returns
        -------
        pd.DataFrame
            The published selection.
        """
        table = self.select()

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        table.to_csv(output_dir / f"{self.name}.csv", index=False)
        (output_dir / f"{self.name}.md").write_text(_to_markdown(table))

        for style, writer in self.writers.items():
            (output_dir / f"{self._latex_stem(style)}.tex").write_text(writer.render(table))

        return table

    def _latex_stem(self, style: str) -> str:
        """Return the file stem for one style's LaTeX output."""
        if len(self.writers) == 1:
            return self.name
        return f"{self.name}_{style}"


def _to_markdown(frame: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub-flavoured markdown table."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(c) for c in frame.columns]
        lines = ["| " + " | ".join(columns) + " |"]
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines) + "\n"
