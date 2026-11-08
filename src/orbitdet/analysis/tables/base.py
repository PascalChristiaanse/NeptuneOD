"""Base class for table producers in the analysis module.

A table class mirrors :class:`orbitdet.visualization.base.Plot`: where a plotter
builds a matplotlib figure and publishes it as a PDF, a table builds a
:class:`pandas.DataFrame` and publishes it as CSV and Markdown.  Both write the
saved file to the Hydra output directory, so a run's figures and its tables are
bookkept the same way.

Subclasses implement :meth:`Table._build`, which returns the dataframe (or a
mapping of name -> dataframe).  The public :meth:`Table.build` wraps that with
the publishing below, so callers only need::

    frame = SourceResidualSummary(access).build()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)

__all__ = ["Table"]


class Table(ABC):
    """Base class for all table-producing classes.

    Parameters
    ----------
    name : str, optional
        Base name used for output files.  Defaults to the snake_case class name
        (set by subclasses that need a different name).
    """

    def __init__(self, name: str | None = None):
        self.name = name or _snake_case(self.__class__.__name__)

    @abstractmethod
    def _build(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Build and return the table dataframe(s) for this producer."""

    def build(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Build the table(s) and publish them to the run output directory.

        Returns
        -------
        pd.DataFrame | dict[str, pd.DataFrame]
            The table(s) as returned by :meth:`_build`.
        """
        result = self._build()
        self.publish(result)
        return result

    def publish(self, result: pd.DataFrame | dict[str, pd.DataFrame]) -> None:
        """Write the table(s) as CSV and Markdown.

        Parameters
        ----------
        result : pd.DataFrame | dict[str, pd.DataFrame]
            One table, or a mapping of table name to table.  A mapping writes
            ``<name>_<key>.csv`` / ``.md``; a single table writes
            ``<name>.csv`` / ``.md``.
        """
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = result if isinstance(result, dict) else {self.name: result}
        for table_name, frame in tables.items():
            csv_path = output_dir / f"{table_name}.csv"
            frame.to_csv(csv_path, index=False)

            markdown_path = output_dir / f"{table_name}.md"
            markdown_path.write_text(_to_markdown(frame))


def _to_markdown(frame: pd.DataFrame) -> str:
    """Render a dataframe as a GitHub-flavoured markdown table.

    Prefers ``DataFrame.to_markdown`` when the optional ``tabulate`` dependency
    is installed, falling back to a hand-rolled renderer otherwise so the table
    layer never hard-depends on it.
    """
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return _to_markdown_fallback(frame)


def _to_markdown_fallback(frame: pd.DataFrame) -> str:
    """Minimal markdown table renderer used when ``tabulate`` is unavailable."""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _snake_case(name: str) -> str:
    """Convert a CamelCase class name to snake_case (e.g. ``ResidualSummary``)."""
    import re

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
