"""LaTeX renderers for the analysis table layer.

A table producer (:class:`orbitdet.analysis.tables.base.Table`) owns *what* the
numbers are; a writer owns *how* they are typeset.  Splitting the two means one
dataframe can be published as a plain ``booktabs`` table for a report and as a
journal-specific table for a manuscript without duplicating the reduction.

:class:`LatexTableWriter` is both the shared base and a usable writer: it renders
a ``booktabs`` table (``\\toprule`` / ``\\midrule`` / ``\\bottomrule``) and
exposes the four structural hooks a subclass overrides to change the surrounding
environment.  :class:`AATableWriter` is the worked example: it keeps the same
cell formatting and only swaps the environment for the ``deluxetable`` macros
that Astronomy & Astrophysics and the AAS journals use.

Example
-------
::

    from orbitdet.analysis.tables import LatexTableWriter

    writer = LatexTableWriter(
        caption="Parameter changes.",
        label="tab:parameter_change",
        headers={"relative_change_percent": r"$\\Delta$ [\\%]"},
        formats={"final": ".6e", "relative_change_percent": ".4f"},
        alignments={"parameter": "l"},
    )
    print(writer.render(frame))

Subclassing
-----------
Override any of :meth:`LatexTableWriter._begin`, :meth:`_header`, :meth:`_body`
or :meth:`_end`; the cell formatting (:meth:`format_cell`), header lookup
(:meth:`header_for`) and alignment (:meth:`alignment_for`) are inherited, so a
journal writer only has to describe its own preamble and row syntax.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import pandas as pd

__all__ = ["LatexTableWriter", "AATableWriter", "NAMED_FORMATS", "escape_latex"]

#: Characters that must be escaped in text cells.  Deliberately minimal: a
#: header such as ``$\\Delta$`` is already LaTeX and must survive untouched, so
#: only the characters that would otherwise break the table are replaced.
_ESCAPES: dict[str, str] = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
}

#: A per-column formatter: either a ``format()`` spec (``".4f"``), the name of
#: an entry in :data:`NAMED_FORMATS`, or a callable taking the raw value and
#: returning the cell text.
CellFormat = str | Callable[[object], str]


def _math(value: object) -> str:
    """Wrap a value in math mode, so ``v_x`` renders as a subscript."""
    return f"${value}$"


def _text(value: object) -> str:
    """Render a value as escaped text."""
    return escape_latex(str(value))


def _toggle(value: object) -> str:
    """Render a boolean truth-table cell as a check mark or a dash."""
    if isinstance(value, str):
        return r"\checkmark" if value.lower() in ("true", "1", "yes") else "--"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return r"\checkmark" if bool(value) else "--"


def _integer(value: object) -> str:
    """Render a count, with a dash for a missing value.

    A column that is missing for some rows is stored as float by pandas, so a
    plain ``"d"`` format would raise on the NaN.  This formatter accepts both
    and renders the gap as ``--``.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return str(int(value))


#: Formatters a config can name by string, for cases a ``format()`` spec cannot
#: express.  ``"math"`` is the useful one: a parameter column holding ``v_x``
#: becomes ``$v_x$`` without the producer having to store LaTeX in the dataframe.
#: ``"toggle"`` renders truth-table cells, where a false toggle is a dash rather
#: than the word ``False``.  ``"int"`` renders a count column that may carry a
#: missing value.
NAMED_FORMATS: dict[str, Callable[[object], str]] = {
    "math": _math,
    "text": _text,
    "toggle": _toggle,
    "int": _integer,
}


def escape_latex(text: str) -> str:
    """Escape the characters that would break a LaTeX table cell.

    Parameters
    ----------
    text : str
        Plain text to escape.

    Returns
    -------
    str
        The text with ``&``, ``%``, ``#`` and ``_`` escaped.
    """
    for character, replacement in _ESCAPES.items():
        text = text.replace(character, replacement)
    return text


class LatexTableWriter:
    """Render a dataframe as a LaTeX ``booktabs`` table.

    Parameters
    ----------
    caption : str, optional
        Table caption.  Omitted from the output when not given.
    label : str, optional
        ``\\label{...}`` value, e.g. ``"tab:parameter_change"``.
    headers : Mapping[str, str], optional
        Display header per column.  Columns not named here use their own name.
    formats : Mapping[str, CellFormat], optional
        Per-column cell format: a ``format()`` spec or a callable.  Columns not
        named here are formatted by :meth:`format_cell`'s default rule.
    alignments : Mapping[str, str], optional
        Per-column alignment (``"l"``, ``"c"`` or ``"r"``).  Columns not named
        here are right-aligned when numeric and left-aligned otherwise.
    columns : Sequence[str], optional
        Columns to render, in this order.  Defaults to every column of the
        frame, so a producer can compute more than one table shows.
    raw : Sequence[str], optional
        Columns whose text is already LaTeX and must not be escaped, e.g. a
        parameter column holding ``$v_x$``.
    footer : Sequence[Sequence[str]], optional
        Extra rows rendered after the data, separated by a rule.  Cells are
        emitted verbatim, so a caller can put ``"Total"`` in the first column.
    notes : Sequence[str], optional
        Notes rendered below the table.
    empty_comment : str, optional
        Text returned instead of a table when the frame is empty.
    """

    def __init__(
        self,
        caption: str | None = None,
        label: str | None = None,
        headers: Mapping[str, str] | None = None,
        formats: Mapping[str, CellFormat] | None = None,
        alignments: Mapping[str, str] | None = None,
        columns: Sequence[str] | None = None,
        raw: Sequence[str] | None = None,
        footer: Sequence[Sequence[str]] | None = None,
        notes: Sequence[str] | None = None,
        empty_comment: str = "% No data.",
    ):
        self.caption = caption
        self.label = label
        self.headers = dict(headers or {})
        self.formats = dict(formats or {})
        self.alignments = dict(alignments or {})
        self.columns = list(columns) if columns is not None else None
        self.raw = set(raw or ())
        self.footer = [list(row) for row in (footer or [])]
        self.notes = list(notes or [])
        self.empty_comment = empty_comment

    # -- Public API -----------------------------------------------------

    def render(self, frame: pd.DataFrame) -> str:
        """Render *frame* as a LaTeX table.

        Parameters
        ----------
        frame : pd.DataFrame
            The table to render.  Only :attr:`columns` are emitted when set.

        Returns
        -------
        str
            The LaTeX source, or :attr:`empty_comment` when the frame is empty.
        """
        frame = self._select(frame)
        if frame.empty:
            return self.empty_comment

        lines: list[str] = []
        lines.extend(self._begin(frame))
        lines.extend(self._header(frame))
        lines.extend(self._body(frame))
        lines.extend(self._end(frame))
        return "\n".join(lines)

    # -- Structural hooks -----------------------------------------------

    def _begin(self, frame: pd.DataFrame) -> list[str]:
        """Open the table environment and emit the caption and label."""
        lines = [r"\begin{table}[htbp]", r"\centering"]
        if self.caption:
            lines.append(f"\\caption{{{self.caption}}}")
        if self.label:
            lines.append(f"\\label{{{self.label}}}")
        lines.append(f"\\begin{{tabular}}{{{self._alignment(frame)}}}")
        lines.append(r"\toprule")
        return lines

    def _header(self, frame: pd.DataFrame) -> list[str]:
        """Emit the header row and the rule below it."""
        cells = " & ".join(self.header_for(column) for column in frame.columns)
        return [f"{cells} \\\\", r"\midrule"]

    def _body(self, frame: pd.DataFrame) -> list[str]:
        """Emit one row per record, then the footer rows."""
        lines = [
            " & ".join(self.format_cell(row[column], column) for column in frame.columns) + r" \\"
            for _, row in frame.iterrows()
        ]
        if self.footer:
            lines.append(r"\midrule")
            lines.extend(" & ".join(str(cell) for cell in row) + r" \\" for row in self.footer)
        return lines

    def _end(self, frame: pd.DataFrame) -> list[str]:
        """Close the table environment and emit the notes."""
        lines = [r"\bottomrule", r"\end{tabular}"]
        if self.notes:
            lines.append(r"\par\smallskip")
            lines.extend(f"\\footnotesize {note}" for note in self.notes)
        lines.append(r"\end{table}")
        return lines

    # -- Shared formatting ----------------------------------------------

    def header_for(self, column: str) -> str:
        """Return the display header for *column*.

        An explicit header is emitted verbatim, because a header is where a
        caller puts markup such as ``$\\Delta$ [\\%]``.  A column with no
        explicit header falls back to its own name, escaped, so a name like
        ``relative_change_percent`` does not break the build.
        """
        explicit = self.headers.get(column)
        if explicit is not None:
            return str(explicit)
        return escape_latex(str(column))

    def alignment_for(self, column: str, frame: pd.DataFrame | None = None) -> str:
        """Return the column alignment, defaulting on the column's dtype."""
        explicit = self.alignments.get(column)
        if explicit is not None:
            return explicit
        if frame is not None and column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]):
                return "r"
        return "l"

    def format_cell(self, value: object, column: str) -> str:
        """Format one cell for LaTeX.

        A configured :attr:`formats` entry wins; otherwise a non-finite float
        becomes ``--``, a finite float is rendered with six significant digits,
        and anything else is escaped text.  Columns listed in :attr:`raw` are
        emitted verbatim, so a cell can carry its own markup.
        """
        spec = self.formats.get(column)
        if callable(spec):
            return spec(value)
        if isinstance(value, float) and not math.isfinite(value):
            return "--"
        if spec is not None:
            named = NAMED_FORMATS.get(spec)
            if named is not None:
                return named(value)
            return format(value, spec)
        if isinstance(value, float):
            return f"{value:.6g}"
        if column in self.raw:
            return str(value)
        return escape_latex(str(value))

    # -- Internals ------------------------------------------------------

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Restrict *frame* to :attr:`columns`, ignoring names it lacks."""
        if self.columns is None:
            return frame
        present = [column for column in self.columns if column in frame.columns]
        return frame[present]

    def _alignment(self, frame: pd.DataFrame) -> str:
        """Return the ``tabular`` column specification."""
        return " ".join(self.alignment_for(column, frame) for column in frame.columns)


class AATableWriter(LatexTableWriter):
    """Render a dataframe with the A&A / AAS ``deluxetable`` macros.

    Only the environment changes: the caption becomes ``\\tablecaption`` (which
    carries the label), the header row is wrapped in ``\\tablehead`` with one
    ``\\colhead`` per column, and the data sits between ``\\startdata`` and
    ``\\enddata``.  Cell formatting is inherited unchanged, so a table renders
    identically in both writers apart from the surrounding markup.

    Parameters
    ----------
    tablewidth : str, optional
        ``\\tablewidth`` argument.  ``"0pt"`` (the default) lets the table size
        itself to its contents.
    typesize : str, optional
        ``\\tabletypesize`` argument, e.g. ``"\\scriptsize"``.  Omitted when not
        given.
    """

    def __init__(self, *args, tablewidth: str = "0pt", typesize: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tablewidth = tablewidth
        self.typesize = typesize

    def _begin(self, frame: pd.DataFrame) -> list[str]:
        lines = [f"\\begin{{deluxetable}}{{{self._alignment(frame)}}}"]
        if self.typesize:
            lines.append(f"\\tabletypesize{{{self.typesize}}}")
        lines.append(f"\\tablewidth{{{self.tablewidth}}}")

        caption = self.caption or ""
        if self.label:
            caption = f"{caption}\\label{{{self.label}}}"
        lines.append(f"\\tablecaption{{{caption}}}")
        return lines

    def _header(self, frame: pd.DataFrame) -> list[str]:
        cells = " & ".join(f"\\colhead{{{self.header_for(column)}}}" for column in frame.columns)
        return ["\\tablehead{", cells, "}", "\\startdata"]

    def _body(self, frame: pd.DataFrame) -> list[str]:
        lines = [
            " & ".join(self.format_cell(row[column], column) for column in frame.columns) + r" \\"
            for _, row in frame.iterrows()
        ]
        lines.extend(" & ".join(str(cell) for cell in row) + r" \\" for row in self.footer)
        return lines

    def _end(self, frame: pd.DataFrame) -> list[str]:
        lines = ["\\enddata"]
        lines.extend(f"\\tablecomments{{{note}}}" for note in self.notes)
        lines.append("\\end{deluxetable}")
        return lines
