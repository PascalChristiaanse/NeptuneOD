"""Estimation-output tables: one wide dataframe, many configured views.

The estimator's artifact carries far more than any single table shows: the
parameter history, the final parameters, the formal errors and the covariance.
Rather than have each table re-read the artifact and recompute the same derived
quantities, this module builds **one** dataframe holding everything that can be
derived per parameter from an
:class:`~tudatpy.estimation.estimation_analysis.EstimationOutput`, and lets a
table config select and rename the columns it wants.

That split is what makes a new table cheap: adding a column to the dataframe
makes it available to every table, and a table is then just a column list, a
header map and a writer.

The dataframe
-------------
One row per estimated parameter, with these columns:

``parameter``
    Display name, from the config or the default Cartesian state names.
``index``
    Position in the parameter vector.
``initial``, ``final``
    A priori and estimated value.
``absolute_change``
    ``final - initial``.
``relative_change_percent``
    ``(final - initial) / initial * 100``; NaN when ``initial`` is zero.
``change_factor``
    ``final / initial``; NaN when ``initial`` is zero.
``formal_error``
    The 1σ formal error from the covariance diagonal.
``relative_formal_error_percent``
    ``formal_error / |final| * 100``; NaN when ``final`` is zero.
``shift_sigma``
    How far the estimate moved, in units of its own formal error.
``correlation_with_previous``
    Correlation with the preceding parameter, from the covariance.

Example
-------
::

    from orbitdet.analysis.tables import EstimationParameterTable

    table = EstimationParameterTable(estimation_output)
    frame = table._build()         # the full dataframe, nothing selected
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from orbitdet.analysis.tables.base import Table

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    from tudatpy.estimation import estimation_analysis as est_an

__all__ = ["EstimationParameterTable", "DEFAULT_PARAMETER_NAMES", "build_parameter_frame"]

#: Names used when the caller supplies none: the Cartesian state of the body
#: whose initial state is estimated, in the inertial frame.
DEFAULT_PARAMETER_NAMES: tuple[str, ...] = ("x", "y", "z", "v_x", "v_y", "v_z")

#: Column order of the produced dataframe.
COLUMNS: tuple[str, ...] = (
    "parameter",
    "index",
    "initial",
    "final",
    "absolute_change",
    "relative_change_percent",
    "change_factor",
    "formal_error",
    "relative_formal_error_percent",
    "shift_sigma",
    "correlation_with_previous",
)


class EstimationParameterTable(Table):
    """The full per-parameter dataframe of a solved estimation.

    Parameters
    ----------
    estimation_output : est_an.EstimationOutput
        The solved estimation problem.
    parameter_names : Sequence[str], optional
        Display name per parameter, in parameter-vector order.  Defaults to
        :data:`DEFAULT_PARAMETER_NAMES`, padded with ``p<i>`` when the vector is
        longer.
    name : str, optional
        Base name for the published files.  Defaults to
        ``estimation_parameters``.
    writers : Mapping[str, LatexTableWriter], optional
        LaTeX renderers keyed by style name.  Empty publishes only CSV and
        Markdown, which is the right default for the wide dataframe: a table
        config selects the columns it wants and supplies its own writers.
    """

    def __init__(
        self,
        estimation_output: est_an.EstimationOutput,
        parameter_names: list[str] | tuple[str, ...] | None = None,
        name: str = "estimation_parameters",
        writers=None,
    ):
        super().__init__(name=name, writers=writers)
        self.estimation_output = estimation_output
        self.parameter_names = list(parameter_names or DEFAULT_PARAMETER_NAMES)

    def _build(self) -> pd.DataFrame:
        return build_parameter_frame(self.estimation_output, self.parameter_names)


def build_parameter_frame(
    estimation_output: est_an.EstimationOutput,
    parameter_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build the full per-parameter dataframe from an estimation output.

    Parameters
    ----------
    estimation_output : est_an.EstimationOutput
        The solved estimation problem.
    parameter_names : Sequence[str], optional
        Display name per parameter, in parameter-vector order.

    Returns
    -------
    pd.DataFrame
        One row per parameter, with the columns listed in :data:`COLUMNS`.
    """
    initial = _initial_parameters(estimation_output)
    final = np.asarray(estimation_output.final_parameters, dtype=float)
    if initial.shape != final.shape:
        raise ValueError(
            "The a priori and final parameter vectors differ in length "
            f"({initial.size} vs {final.size}); the estimation output is inconsistent."
        )

    n_params = initial.size
    names = _names_for(n_params, list(parameter_names or DEFAULT_PARAMETER_NAMES))
    formal_error = _formal_errors(estimation_output, n_params)
    correlations = _correlations(estimation_output, n_params)

    return pd.DataFrame(
        {
            "parameter": names,
            "index": np.arange(n_params),
            "initial": initial,
            "final": final,
            "absolute_change": final - initial,
            "relative_change_percent": _relative_change_percent(initial, final),
            "change_factor": _change_factor(initial, final),
            "formal_error": formal_error,
            "relative_formal_error_percent": _relative_formal_error_percent(
                formal_error, final
            ),
            "shift_sigma": _shift_sigma(initial, final, formal_error),
            "correlation_with_previous": correlations,
        },
        columns=list(COLUMNS),
    )


def _initial_parameters(estimation_output: est_an.EstimationOutput) -> np.ndarray:
    """Return the a priori parameter vector from the estimation output.

    ``parameter_history`` is stored parameter-major (one row per parameter, one
    column per iteration), so the a priori vector is its first column.  An
    output with no history at all cannot describe a change, which is an error
    rather than an empty table.
    """
    history = np.asarray(estimation_output.parameter_history, dtype=float)
    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError(
            "The estimation output carries no parameter history, so the a priori "
            "parameter vector is unknown. Re-run the estimation with "
            "save_residuals_and_parameters_per_iteration=True."
        )
    return history[:, 0]


def _formal_errors(estimation_output: est_an.EstimationOutput, n_params: int) -> np.ndarray:
    """Return the 1σ formal errors, or NaN when the output carries none."""
    try:
        errors = np.asarray(estimation_output.formal_errors, dtype=float)
    except Exception:  # tudatpy raises a plain RuntimeError when absent
        return np.full(n_params, np.nan)
    if errors.shape != (n_params,):
        return np.full(n_params, np.nan)
    return errors


def _correlations(estimation_output: est_an.EstimationOutput, n_params: int) -> np.ndarray:
    """Return each parameter's correlation with the one before it.

    The first parameter has no predecessor, so its entry is NaN.  A missing or
    degenerate covariance yields NaN rather than an error: the column is
    informative, not essential.
    """
    correlations = np.full(n_params, np.nan)
    try:
        covariance = np.asarray(estimation_output.covariance, dtype=float)
    except Exception:  # tudatpy raises a plain RuntimeError when absent
        return correlations
    if covariance.shape != (n_params, n_params):
        return correlations

    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    for i in range(1, n_params):
        denominator = sigma[i] * sigma[i - 1]
        if denominator > 0.0:
            correlations[i] = covariance[i, i - 1] / denominator
    return correlations


def _relative_change_percent(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Percentage change from *initial* to *final*, NaN where undefined.

    A parameter that starts at zero has no meaningful percentage change, so it
    is reported as NaN (rendered ``--``) rather than as an infinite value.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.where(initial != 0.0, (final - initial) / initial * 100.0, np.nan)
    return change


def _change_factor(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Multiplicative change ``final / initial``, NaN where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(initial != 0.0, final / initial, np.nan)
    return factor


def _relative_formal_error_percent(formal_error: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Formal error as a percentage of the estimated value, NaN where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(final != 0.0, formal_error / np.abs(final) * 100.0, np.nan)
    return relative


def _shift_sigma(initial: np.ndarray, final: np.ndarray, formal_error: np.ndarray) -> np.ndarray:
    """How far the estimate moved, in units of its own formal error.

    A shift of one sigma means the update is the size of the reported
    uncertainty, which is the scale a reader compares against.  A parameter with
    no formal error has no such scale, so it is NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sigmas = np.where(formal_error > 0.0, (final - initial) / formal_error, np.nan)
    return sigmas


def _names_for(size: int, names: list[str]) -> list[str]:
    """Pad or truncate *names* to exactly *size* entries."""
    if len(names) < size:
        names = names + [f"p{i}" for i in range(len(names), size)]
    return names[:size]
