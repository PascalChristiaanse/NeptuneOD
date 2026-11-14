"""Comparison table for a parameter-combination sweep.

A sweep runs one fit per *specification* — a set of parameters to estimate — and
this module turns those fits into one table: the truth table (which parameters
each fit estimated), the convergence verdict, and the residual statistics.

Where the specifications come from
----------------------------------
They are passed in, not declared here.  The sweep is a property of the
experiment, so the experiment's Snakefile owns the enumeration and hands the
specifications to this module as plain mappings::

    [{"slug": "gravity_nep_GM", "label": "gravity_nep_GM",
      "neptune_GM": True, "triton_GM": False, "neptune_j2_j4": False}, ...]

Every key other than ``slug`` and ``label`` is a truth-table column.  Keeping
the enumeration out of the library means a new sweep is a rule change, not a
library change, and the table cannot describe a different set of fits than the
rule ran.

Convergence
-----------
The estimator does not report convergence as a flag.  It stops either because
the convergence checker fired or because it ran out of iterations, and the only
record of which is the solver's own log, whose last lines are::

    Maximum number of iterations reached
    Best iteration: 4 out of 4
    Rms residual from best iteration: 5.07106e-05

A fit that stopped on the iteration limit is *not* converged, however small its
residual: the residual is still falling and the parameters are still moving.
:func:`read_convergence` therefore reads the log rather than inferring from the
residual, and a fit whose log cannot be read is reported as not converged rather
than silently counted as a success.

Example
-------
::

    from orbitdet.analysis.tables import SweepComparisonTable, summarise_fit

    rows = [summarise_fit(spec, output, log_text) for spec, output, log_text in fits]
    frame = SweepComparisonTable(specifications, rows)._build()
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from orbitdet.analysis.tables.base import Table

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    from tudatpy.estimation import estimation_analysis as est_an

__all__ = [
    "COMPARISON_COLUMNS",
    "SweepComparisonTable",
    "build_sweep_frame",
    "read_convergence",
    "summarise_fit",
]

#: Keys of a specification mapping that are not truth-table columns.
_NON_CELL_KEYS: tuple[str, ...] = ("slug", "label", "directory")

#: Column order of the comparison table.
COMPARISON_COLUMNS: tuple[str, ...] = (
    "specification",
    "converged",
    "rms",
    "n_iterations",
    "best_iteration",
    "n_parameters",
    "n_residuals",
    "max_abs",
    "rms_R",
    "rms_S",
    "rms_W",
)

#: Solver log line naming the best iteration, e.g. ``Best iteration: 4 out of 4``.
_BEST_ITERATION_PATTERN = re.compile(r"Best iteration:\s*(\d+)\s*out of\s*(\d+)")

#: Solver log line reporting the residual of one iteration.
_RESIDUAL_PATTERN = re.compile(r"Current residual:\s*([0-9.eE+-]+)")

#: Solver log line reporting the residual of the best iteration.
_BEST_RESIDUAL_PATTERN = re.compile(r"Rms residual from best iteration:\s*([0-9.eE+-]+)")

#: Solver log line printed when the iteration limit, not the convergence
#: checker, ended the estimation.
_ITERATION_LIMIT_MARKER = "Maximum number of iterations reached"


def specification_cells(specification: Mapping[str, object]) -> dict[str, object]:
    """Return the truth-table cells of *specification*.

    Parameters
    ----------
    specification : Mapping[str, object]
        One specification, as the sweep rule declares it.  ``slug`` and
        ``label`` are metadata; every other key is a truth-table column.

    Returns
    -------
    dict[str, object]
        The truth-table columns, in the order the mapping lists them.
    """
    return {key: value for key, value in specification.items() if key not in _NON_CELL_KEYS}


def specification_columns(specifications: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Return the truth-table columns of *specifications*, in first-seen order.

    Taking the union rather than the first specification's keys means a group
    whose specifications do not all carry the same columns still produces a
    complete table, with the missing cells left empty.
    """
    columns: dict[str, None] = {}
    for specification in specifications:
        columns.update(dict.fromkeys(specification_cells(specification)))
    return tuple(columns)


def read_convergence(log_text: str) -> tuple[bool, int | None, int | None]:
    """Read the convergence verdict from the solver's log.

    A fit is converged when both hold:

    * the solver's own termination reason is not the iteration cap.  The solver
      stops either because its convergence checker fired or because it ran out
      of iterations, and it prints the latter; a fit that stopped on the cap is
      not converged, however small its residual, because the parameters were
      still moving.  This is only meaningful when ``minimum_residual_change`` is
      set: with the solver's default of ``0.0`` the checker can never fire, so
      every fit ends on the cap.
    * the best iteration is better than the initial one.  A fit that never
      improved on its starting parameters has not converged on anything, even if
      the checker fired because the residual stopped moving.

    Parameters
    ----------
    log_text : str
        The contents of the fit's ``estimation_log.log``.

    Returns
    -------
    tuple[bool, int | None, int | None]
        ``(converged, best_iteration, n_iterations)``.  A log with no
        recognisable summary is reported as not converged, with ``None`` for
        both iteration counts, so an unreadable log is never mistaken for a
        success.
    """
    match = _BEST_ITERATION_PATTERN.search(log_text)
    if match is None:
        return False, None, None
    best_iteration = int(match.group(1))
    n_iterations = int(match.group(2))

    stopped_on_cap = _ITERATION_LIMIT_MARKER in log_text
    converged = not stopped_on_cap and _improved_on_the_start(log_text)
    return converged, best_iteration, n_iterations


def _improved_on_the_start(log_text: str) -> bool:
    """Return whether the best iteration beat the initial one.

    The first ``Current residual`` line is the residual at the starting
    parameters, and the summary reports the residual of the best iteration.  A
    fit whose best is no better than its start has not converged on anything,
    however the solver terminated.

    A log missing either value is reported as not improved, so an unreadable log
    is never mistaken for a success.
    """
    residuals = _RESIDUAL_PATTERN.findall(log_text)
    best = _BEST_RESIDUAL_PATTERN.search(log_text)
    if not residuals or best is None:
        return False
    initial = float(residuals[0])
    best_residual = float(best.group(1))
    return best_residual < initial


def _rms(values: np.ndarray) -> float:
    """Root-mean-square of the finite entries of *values*, or NaN when none."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def _residual_components(output: est_an.EstimationOutput) -> np.ndarray | None:
    """Return the residuals as an ``(n_observations, 3)`` array, or ``None``.

    The synthetic collection measures a relative Cartesian position, so each
    observation contributes three scalars.  A residual vector that does not
    divide by three cannot be laid out per observation, and the per-component
    columns are then left NaN rather than guessed at.
    """
    residuals = np.asarray(output.final_residuals, dtype=float).ravel()
    if residuals.size == 0 or residuals.size % 3 != 0:
        return None
    return residuals.reshape(-1, 3)


def summarise_fit(
    specification: Mapping[str, object],
    output: est_an.EstimationOutput,
    log_text: str,
) -> dict[str, object]:
    """Return one fit's row of the comparison table.

    Parameters
    ----------
    specification : Mapping[str, object]
        The specification the fit ran, as the sweep rule declares it.
    output : est_an.EstimationOutput
        The fit's saved estimation output.
    log_text : str
        The fit's solver log, used for the convergence verdict.

    Returns
    -------
    dict[str, object]
        The truth-table cells, the convergence verdict, and the residual
        statistics.  The residual RMS is the estimator's own best-iteration
        residual, so it describes exactly the fit that was saved.
    """
    converged, best_iteration, n_iterations = read_convergence(log_text)
    residuals = np.asarray(output.final_residuals, dtype=float).ravel()

    row: dict[str, object] = {
        "specification": specification["slug"],
        "label": specification.get("label", specification["slug"]),
        **specification_cells(specification),
        "converged": converged,
        "best_iteration": best_iteration,
        "n_iterations": n_iterations,
        "n_parameters": int(np.asarray(output.final_parameters).size),
        "n_residuals": int(residuals.size),
        "rms": _rms(residuals),
        "max_abs": float(np.max(np.abs(residuals[np.isfinite(residuals)])))
        if np.isfinite(residuals).any()
        else float("nan"),
    }

    components = _residual_components(output)
    for index, name in enumerate(("R", "S", "W")):
        row[f"rms_{name}"] = _rms(components[:, index]) if components is not None else np.nan
    return row


def build_sweep_frame(
    specifications: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Assemble the comparison frame from per-fit rows.

    The frame is ordered by *specifications*, not by the order the rows arrive
    in, so the table reads as the truth table.  A specification with no row (a
    fit that produced no output) is still listed, with NaN statistics and
    ``converged`` false, so a missing fit is visible rather than absent.

    Parameters
    ----------
    specifications : Sequence[Mapping[str, object]]
        The specifications the sweep ran, in table order.
    rows : Sequence[Mapping[str, object]]
        Rows as returned by :func:`summarise_fit`.

    Returns
    -------
    pd.DataFrame
        One row per specification, with the truth-table columns first.
    """
    by_slug = {str(row["specification"]): row for row in rows}
    ordered = []
    for specification in specifications:
        slug = str(specification["slug"])
        row = by_slug.get(slug)
        if row is None:
            row = {
                "specification": slug,
                "label": specification.get("label", slug),
                **specification_cells(specification),
                "converged": False,
            }
        ordered.append(row)

    frame = pd.DataFrame(ordered)
    leading = [
        "specification",
        "label",
        *specification_columns(specifications),
        "converged",
    ]
    trailing = [column for column in COMPARISON_COLUMNS if column not in leading]
    columns = [*leading, *trailing]
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns]


class SweepComparisonTable(Table):
    """The comparison table of one parameter-combination sweep.

    Parameters
    ----------
    specifications : Sequence[Mapping[str, object]]
        The specifications the sweep ran, in table order.
    rows : Sequence[Mapping[str, object]]
        Per-fit rows, as returned by :func:`summarise_fit`.
    name : str, optional
        Base name for the published files.  Defaults to
        ``sweep_comparison``.
    writers : Mapping[str, LatexTableWriter], optional
        LaTeX renderers keyed by style name.
    """

    def __init__(self, specifications, rows, name: str | None = None, writers=None):
        super().__init__(name=name or "sweep_comparison", writers=writers)
        self.specifications = list(specifications)
        self.rows = list(rows)

    def _build(self) -> pd.DataFrame:
        return build_sweep_frame(self.specifications, self.rows)
