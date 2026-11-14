"""Config plumbing shared by the table producers and the table script.

A table is described by its own config file, exactly as a figure is: the file
names the producer, the columns it shows and the writer that typesets it.  The
script that runs a table therefore only has to name the file, and editing one
table's caption cannot invalidate another.

The writer is declared as a nested block so a table can be re-typeset for a
different journal without touching the producer.  The block describes the
*content* (caption, headers, formats) once, and ``styles`` names every journal
format to emit from it::

    table: "EstimationParameterTable"
    output_file: "parameter_change"
    writer:
      styles: ["booktabs", "aa"]
      caption: "Initial and final parameter values."
      label: "tab:parameter_change"
      headers:
        relative_change_percent: "$\\Delta$ [\\%]"
      formats:
        initial: ".6e"
      style_options:
        aa:
          typesize: "\\scriptsize"

That writes ``parameter_change_booktabs.tex`` and ``parameter_change_aa.tex``
from one description, so the two journals cannot drift apart.  A single style
writes ``parameter_change.tex`` with no suffix.

Example
-------
::

    from orbitdet.analysis.tables.config import attach_table_config, build_writers

    cfg = attach_table_config(cfg, "tab_parameter_change")
    writers = build_writers(cfg.tables["tab_parameter_change"].writer)
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from orbitdet.analysis.tables.latex import AATableWriter, LatexTableWriter

logger = logging.getLogger(__name__)

__all__ = [
    "WRITER_STYLES",
    "DEFAULT_STYLE",
    "attach_table_config",
    "build_writer",
    "build_writers",
    "resolve_table_builder",
]

#: LaTeX styles a table config can request with its ``writer.styles`` list.
#: ``booktabs`` is the plain journal-neutral table; ``aa`` is the A&A / AAS
#: ``deluxetable`` environment.
WRITER_STYLES: dict[str, type[LatexTableWriter]] = {
    "booktabs": LatexTableWriter,
    "aa": AATableWriter,
}

#: Style used when a writer block names none.
DEFAULT_STYLE = "booktabs"


def attach_table_config(cfg: DictConfig, table_name: str) -> DictConfig:
    """Merge the table's own config file into ``cfg.tables.<table_name>``.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration, carrying the path of the table config in
        ``cfg.table_config``.
    table_name : str
        Name the table config is stored under, normally the stem of its file.

    Returns
    -------
    DictConfig
        The same configuration, with ``cfg.tables`` replaced by the single
        table block.
    """
    path = cfg.table_config
    logger.info("Loading table config from %s", path)

    table_cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    cfg.tables = {table_name: table_cfg}
    OmegaConf.set_struct(cfg, True)

    return cfg


def build_writers(writer_cfg: DictConfig | None) -> dict[str, LatexTableWriter]:
    """Construct one LaTeX writer per style named by *writer_cfg*.

    The block's content keys (``caption``, ``label``, ``headers``, ``formats``,
    ``alignments``, ...) are shared by every style, so the formats cannot drift
    apart.  ``style_options`` adds or overrides keys for one style only, which is
    where a journal-specific option such as the A&A ``typesize`` belongs.

    Parameters
    ----------
    writer_cfg : DictConfig, optional
        The table config's ``writer`` block.  ``None`` means the table publishes
        no LaTeX, which is what a table with no writer block asks for.

    Returns
    -------
    dict[str, LatexTableWriter]
        Style name to writer, in the order the config lists them.  Empty when no
        block was given.

    Raises
    ------
    ValueError
        If the block names a style that is not in :data:`WRITER_STYLES`.
    """
    if writer_cfg is None:
        return {}

    options = OmegaConf.to_container(writer_cfg, resolve=True) or {}
    styles = options.pop("styles", None) or [DEFAULT_STYLE]
    style_options = options.pop("style_options", None) or {}

    writers: dict[str, LatexTableWriter] = {}
    for style in styles:
        writer_class = WRITER_STYLES.get(style)
        if writer_class is None:
            raise ValueError(
                f"Unknown table style '{style}'. Available: {sorted(WRITER_STYLES)}."
            )
        per_style = dict(options)
        per_style.update(style_options.get(style) or {})
        writers[style] = writer_class(**per_style)
    return writers


def build_writer(writer_cfg: DictConfig | None) -> LatexTableWriter | None:
    """Construct the writer for the first style named by *writer_cfg*.

    A convenience for callers that only need one format; :func:`build_writers`
    is what the table script uses so every configured style is emitted.

    Parameters
    ----------
    writer_cfg : DictConfig, optional
        The table config's ``writer`` block.

    Returns
    -------
    LatexTableWriter | None
        The first style's writer, or ``None`` when no block was given.
    """
    writers = build_writers(writer_cfg)
    return next(iter(writers.values()), None)


def resolve_table_builder(registry: dict, table_cfg: DictConfig):
    """Return the table class declared by the table config.

    Parameters
    ----------
    registry : dict
        Maps table class name to the class.
    table_cfg : DictConfig
        The table's own config block, which must carry a ``table`` key.

    Returns
    -------
    type
        The table class to instantiate.
    """
    table_name = table_cfg.get("table")
    if table_name is None:
        raise ValueError(
            "Table config has no 'table' key. Add e.g. 'table: \"ParameterChangeTable\"' "
            f"to the config. Available tables: {sorted(registry)}."
        )
    builder = registry.get(table_name)
    if builder is None:
        raise ValueError(f"Unknown table '{table_name}'. Available: {sorted(registry)}.")
    return builder
