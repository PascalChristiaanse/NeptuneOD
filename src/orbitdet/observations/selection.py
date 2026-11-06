"""Row selection over the standard observation frame.

A *selection* answers one question: **which observations does this rule apply
to?**  It is the "where" half of a weighting or outlier-rejection plan, and it
is deliberately separate from *clustering*, which answers "how are those
observations partitioned?".

Selections are expressed as pandas query strings evaluated against the standard
frame, so any column the frame carries is selectable::

    select: "data_source == 'nsdb' and observatory_code != '689'"

That is the whole language.  It is not a bespoke DSL: it is :meth:`pandas.DataFrame.query`,
which means the usual pandas operators (``==``, ``!=``, ``<``, ``in``, ``and``,
``or``, ``not``, ``.str.contains(...)``) all work, and a typo in a column name
raises at load time rather than silently matching nothing.

Why a query string rather than a structured predicate tree
----------------------------------------------------------
A structured form (``{column: ..., op: ..., value: ...}``) needs its own parser,
its own error messages and its own documentation, and it still cannot express
``a and (b or c)`` without nesting.  A query string is one line, is already
familiar, and gets pandas' error messages for free.  The cost is that the
expression is evaluated, so :func:`compile_selection` validates it against the
schema's column names up front and rejects anything that reaches outside the
frame.

Selections compose with clustering
----------------------------------
A plan pairs a selection with a strategy and a clustering::

    weighting:
      default:
        strategy: {type: id_v2}
        clustering: {levels: [{type: timeframe, gap_threshold_hours: 8.0}]}
      overrides:
        - select: "data_source == 'gaia_archive'"
          strategy: {type: gaia_archive, covariance_mode: full_matrix}
        - select: "observatory_code == 'Voyager 2'"
          strategy: {type: layered, ...}
          clustering: {levels: [{type: section, ...}, {type: timeframe, ...}]}

The first matching override wins; ``default`` catches everything else.  This is
the same "default plus overrides" shape as the router, but keyed on *any* frame
column rather than only the set identifier, and with the clustering attached to
the same entry instead of split across two config sections.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from orbitdet.preprocessing.schema import STANDARD_DATAFRAME_COLUMNS

logger = logging.getLogger(__name__)

__all__ = [
    "Selection",
    "compile_selection",
    "STANDARD_COLUMNS",
]

#: Every column a selection may reference, in canonical order.
STANDARD_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in STANDARD_DATAFRAME_COLUMNS)

#: Columns the analysis layer adds on top of the standard schema.  A selection
#: may reference these too, but only once the frame has been derived.
_DERIVED_COLUMNS: tuple[str, ...] = (
    "observation_set_index",
    "rejected",
    "residual_first_rad",
    "residual_second_rad",
    "weight_first",
    "weight_second",
    "residual_along_scan_rad",
    "residual_across_scan_rad",
    "z_first",
    "z_second",
)


@dataclass(frozen=True)
class Selection:
    """A row predicate over the standard observation frame.

    Instances are produced by :func:`compile_selection`, which validates the
    expression against the known column names.  Constructing one directly skips
    that check, so prefer the factory.

    Attributes
    ----------
    query : str | None
        A :meth:`pandas.DataFrame.query` expression, or ``None`` for "every
        row".  ``None`` is the default selection and matches everything.
    description : str
        The original config text, kept for logging and error messages.
    """

    query: str | None = None
    description: str = ""

    # -- Evaluation -----------------------------------------------------

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        """Return a boolean mask of the rows this selection matches.

        Parameters
        ----------
        frame : pd.DataFrame
            The standard (or derived) observation frame.

        Returns
        -------
        np.ndarray
            A boolean array aligned with ``frame``'s rows.

        Raises
        ------
        ValueError
            If the query references a column the frame does not have, or if
            pandas cannot evaluate it.
        """
        if self.query is None:
            return np.ones(len(frame), dtype=bool)
        try:
            result = frame.eval(self.query)
        except Exception as exc:  # pandas raises several unrelated types
            raise ValueError(
                f"Selection {self.description!r} could not be evaluated against the "
                f"observation frame: {exc}"
            ) from exc
        mask = np.asarray(result, dtype=bool)
        if mask.shape != (len(frame),):
            raise ValueError(
                f"Selection {self.description!r} produced a mask of shape {mask.shape}, "
                f"expected ({len(frame)},)."
            )
        return mask

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the rows of *frame* this selection matches."""
        return frame.loc[self.mask(frame)]

    def matches(self, frame: pd.DataFrame) -> bool:
        """Whether any row of *frame* matches this selection."""
        return bool(self.mask(frame).any())

    # -- Introspection --------------------------------------------------

    @property
    def is_universal(self) -> bool:
        """Whether this selection matches every row."""
        return self.query is None

    def __repr__(self) -> str:
        return f"Selection({self.description or self.query or '<all rows>'!r})"


def compile_selection(spec: Any) -> Selection:
    """Build a :class:`Selection` from a config value, validating it.

    Parameters
    ----------
    spec : str | Mapping | None
        One of:

        - ``None`` — match every row.
        - a query string, e.g. ``"data_source == 'nsdb'"``.
        - a mapping with a ``where`` key holding that string, e.g.
          ``{where: "data_source == 'nsdb'"}``.  The mapping form exists so a
          selection can grow extra keys later without breaking configs.

    Returns
    -------
    Selection
        The compiled selection.

    Raises
    ------
    TypeError
        If *spec* is not a string, mapping or ``None``.
    ValueError
        If the expression is empty, references an unknown column, or is not a
        valid pandas expression.
    """
    if spec is None:
        return Selection(query=None, description="")

    if isinstance(spec, (DictConfig, Mapping)):
        plain = OmegaConf.to_container(spec, resolve=True) if isinstance(spec, DictConfig) else spec
        unknown = set(plain) - {"where"}
        if unknown:
            raise ValueError(
                f"Unknown selection key(s) {sorted(unknown)}; expected 'where'."
            )
        if "where" not in plain:
            raise ValueError("A selection mapping must have a 'where' key.")
        return compile_selection(plain["where"])

    if not isinstance(spec, str):
        raise TypeError(
            f"A selection must be a query string, a mapping with 'where', or None; "
            f"got {type(spec).__name__}."
        )

    query = spec.strip()
    if not query:
        raise ValueError("A selection query string must not be empty.")

    _validate_query(query)
    return Selection(query=query, description=query)


def _validate_query(query: str) -> None:
    """Check that *query* parses and only references known columns.

    Evaluated against an empty frame carrying every standard column, so a typo
    in a column name fails here — at config load — rather than silently matching
    nothing at run time.
    """
    from orbitdet.preprocessing.schema import empty_frame

    probe = empty_frame()
    for name in _DERIVED_COLUMNS:
        if name not in probe.columns:
            probe[name] = pd.Series(dtype="float64")

    try:
        probe.eval(query)
    except pd.errors.UndefinedVariableError as exc:
        raise ValueError(
            f"Selection {query!r} references an unknown column: {exc}. "
            f"Available columns: {sorted(probe.columns)}."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Selection {query!r} is not a valid expression: {exc}") from exc
