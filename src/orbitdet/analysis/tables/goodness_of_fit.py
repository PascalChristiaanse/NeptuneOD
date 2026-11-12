"""Goodness-of-fit table for a set of solved estimations.

Comparing fits means comparing their residuals, but a single RMS hides the
structure: a fit can have a small RMS and still be wrong in one RSW component,
and the component that is wrong is what tells a reader which part of the model
is inadequate.  This table therefore reports, per fit, the RMS of the residuals
and the largest residual in each RSW component and in the norm.

Where the residuals come from
-----------------------------
The residuals are the estimator's own, read off
:attr:`~tudatpy.estimation.estimation_analysis.EstimationOutput.final_residuals`
- the residuals of the best iteration, in the observation order the estimator
used.  They are not recomputed, so the table describes exactly the fit that was
saved.

Rotating into RSW
-----------------
The estimator stores each residual in the frame its observation model measures
in, which for a relative Cartesian position is the global inertial frame.  The
RSW frame is defined by the propagated body's own state, so the rotation needs
that state at every observation epoch.

The best iteration's state history is on the propagator's own grid, not the
observation grid, so it is interpolated onto the observation epochs.  The
interpolation is linear and only used to orient the frame; the residual values
themselves are never interpolated.

Example
-------
::

    from orbitdet.analysis.tables import GoodnessOfFitTable

    table = GoodnessOfFitTable(
        {"IAU 2015 pole": output_a, "Jacobson (2009) pole": output_b},
        observations=collection,
    )
    frame = table._build()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tudatpy.astro import frame_conversion as fc

from orbitdet.analysis.tables.base import Table

if TYPE_CHECKING:  # pragma: no cover - import for annotations only
    from tudatpy.estimation import estimation_analysis as est_an
    from tudatpy.estimation import observations as obs

__all__ = ["GoodnessOfFitTable", "build_goodness_of_fit_frame", "RSW_COMPONENTS"]

#: The RSW component names, in the order the rotation matrix returns them.
RSW_COMPONENTS: tuple[str, ...] = ("R", "S", "W")

#: Column order of the produced dataframe.
COLUMNS: tuple[str, ...] = (
    "model",
    "n_residuals",
    "rms",
    "max_abs",
    "rms_R",
    "rms_S",
    "rms_W",
    "max_abs_R",
    "max_abs_S",
    "max_abs_W",
    "rms_norm",
    "max_abs_norm",
)


class GoodnessOfFitTable(Table):
    """Per-fit residual RMS and per-RSW-component maximum error.

    Parameters
    ----------
    outputs : Mapping[str, est_an.EstimationOutput]
        Fit label to estimation output, in the order the rows should appear.
    observations : obs.ObservationCollection
        The collection the fits were run against.  Its epochs are the grid the
        residuals live on, and its observable size says how many scalars each
        observation contributes.
    labels : Mapping[str, str], optional
        Display label per fit key.  A key with no entry is shown as-is.  This
        exists because a rule passes the fits as Hydra overrides, whose grammar
        forbids spaces in a key, so the keys are slugs and the labels live here.
    name : str, optional
        Base name for the published files.  Defaults to ``goodness_of_fit``.
    writers : Mapping[str, LatexTableWriter], optional
        LaTeX renderers keyed by style name.
    """

    def __init__(
        self,
        outputs: Mapping[str, est_an.EstimationOutput],
        observations: obs.ObservationCollection,
        labels: Mapping[str, str] | None = None,
        name: str = "goodness_of_fit",
        writers=None,
    ):
        super().__init__(name=name, writers=writers)
        self.outputs = dict(outputs)
        self.observations = observations
        self.labels = dict(labels or {})

    def _build(self) -> pd.DataFrame:
        return build_goodness_of_fit_frame(self.outputs, self.observations, labels=self.labels)


def build_goodness_of_fit_frame(
    outputs: Mapping[str, est_an.EstimationOutput],
    observations: obs.ObservationCollection,
    labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build the per-fit goodness-of-fit dataframe.

    Parameters
    ----------
    outputs : Mapping[str, est_an.EstimationOutput]
        Fit key to estimation output, in row order.
    observations : obs.ObservationCollection
        The collection the fits were run against.
    labels : Mapping[str, str], optional
        Display label per fit key.  A key with no entry is shown as-is.

    Returns
    -------
    pd.DataFrame
        One row per fit, with the columns listed in :data:`COLUMNS`.
    """
    labels = dict(labels or {})
    epochs = np.asarray(observations.concatenated_times, dtype=float)
    rows = [_row_for(labels.get(key, key), output, epochs) for key, output in outputs.items()]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _row_for(label: str, output: est_an.EstimationOutput, epochs: np.ndarray) -> dict:
    """Return one fit's row of the goodness-of-fit table."""
    residuals = np.asarray(output.final_residuals, dtype=float).ravel()
    finite = residuals[np.isfinite(residuals)]

    row = {
        "model": label,
        "n_residuals": int(residuals.size),
        "rms": _rms(finite),
        "max_abs": float(np.max(np.abs(finite))) if finite.size else np.nan,
    }

    rsw = _residuals_in_rsw(output, residuals, epochs)
    if rsw is None:
        for name in RSW_COMPONENTS:
            row[f"rms_{name}"] = np.nan
            row[f"max_abs_{name}"] = np.nan
        row["rms_norm"] = np.nan
        row["max_abs_norm"] = np.nan
        return row

    for index, name in enumerate(RSW_COMPONENTS):
        column = rsw[:, index]
        column = column[np.isfinite(column)]
        row[f"rms_{name}"] = _rms(column)
        row[f"max_abs_{name}"] = float(np.max(np.abs(column))) if column.size else np.nan

    norms = np.linalg.norm(rsw, axis=1)
    norms = norms[np.isfinite(norms)]
    row["rms_norm"] = _rms(norms)
    row["max_abs_norm"] = float(np.max(norms)) if norms.size else np.nan
    return row


def _residuals_in_rsw(
    output: est_an.EstimationOutput, residuals: np.ndarray, epochs: np.ndarray
) -> np.ndarray | None:
    """Rotate the residual vector into the RSW frame.

    Parameters
    ----------
    output : est_an.EstimationOutput
        The fit whose best-iteration state history defines the frame.
    residuals : np.ndarray
        The flat residual vector, three scalars per observation.
    epochs : np.ndarray
        The observation epochs, one per scalar residual.  A collection expands
        its epochs per scalar, so an observation measuring three components
        appears three times; the block's first epoch is the observation's.

    Returns
    -------
    np.ndarray | None
        An ``(n_observations, 3)`` array of RSW residuals, or ``None`` when the
        residuals cannot be laid out per observation or the state history is
        missing.  A ``None`` result leaves the RSW columns NaN rather than
        guessing at a frame.
    """
    n_components = len(RSW_COMPONENTS)
    if epochs.size == 0 or residuals.size != epochs.size:
        return None
    if residuals.size % n_components != 0:
        return None

    blocks = residuals.reshape(-1, n_components)
    # One epoch per observation: the scalars of an observation share it.
    observation_epochs = epochs.reshape(-1, n_components)[:, 0]

    states = _interpolated_states(output, observation_epochs)
    if states is None:
        return None

    rsw = np.full_like(blocks, np.nan)
    for index, state in enumerate(states):
        radius, velocity = state[:3], state[3:]
        # A degenerate state (position parallel to velocity) has no RSW frame.
        if np.linalg.norm(np.cross(radius, velocity)) == 0.0:
            continue
        rotation = np.asarray(fc.inertial_to_rsw_rotation_matrix(state))
        rsw[index, :] = rotation @ blocks[index, :]
    return rsw


def _interpolated_states(output: est_an.EstimationOutput, epochs: np.ndarray) -> np.ndarray | None:
    """Interpolate the best iteration's state history onto *epochs*.

    The propagator's grid is not the observation grid, so the state is
    interpolated linearly.  Only the frame orientation is taken from it, so the
    interpolation error does not enter the residual values.
    """
    try:
        results = output.simulation_results_per_iteration[output.best_iteration]
    except (AttributeError, IndexError, KeyError):
        return None

    dynamics = getattr(results, "dynamics_results", results)
    history = getattr(dynamics, "state_history", None)
    if not history:
        return None

    grid = np.array(sorted(history.keys()), dtype=float)
    values = np.array([np.asarray(history[key]).ravel() for key in sorted(history.keys())])
    if grid.size < 2 or values.shape[1] < 6:
        return None

    # Outside the propagated window there is no state to orient the frame with.
    if epochs.min() < grid[0] or epochs.max() > grid[-1]:
        return None

    return np.column_stack([np.interp(epochs, grid, values[:, index]) for index in range(6)])


def _rms(values: np.ndarray) -> float:
    """Root-mean-square of *values*, or NaN when there are none."""
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))
