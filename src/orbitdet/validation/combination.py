"""Comparison of two propagator/integrator combinations on one dynamical model.

The integrator validation in :mod:`orbitdet.simulation` answers "how accurate is
this integrator on its own" by comparing a propagation against the same
propagation at half the timestep (Richardson extrapolation). That estimate is
contaminated by the linear interpolation used to bring the two runs onto a
common epoch grid, which is of the same order as the integrator error itself.

This module instead compares two complete **combinations** — an integrator plus
a propagator formulation — against each other:

* a *reference* combination, which is trusted to be much more accurate
  (high-order integrator, small timestep);
* a *test* combination, e.g. the production setting under investigation.

Both runs start from the identical initial state, so the difference between the
two state histories is a direct measure of the test combination's error.

Epoch alignment
---------------
Both runs are compared on the **coarser** of the two epoch grids:

* epochs that exist in both histories are compared exactly, with no
  interpolation at all;
* epochs that exist only in the coarser history have the finer trajectory
  evaluated with a Lagrange interpolator of configurable order.

For commensurate timesteps (e.g. 3600 s vs 900 s) every compared epoch is exact,
and the result is a pure integration-error measure. The number of interpolated
epochs is reported so the reader can judge how much of the difference is an
interpolation artifact.

Note that TudatPy's fixed-step propagation emits its first output epoch one step
*before* the configured initial epoch, so the raw compared series begins before
``initial_epoch``. The default ``buffer`` discards those leading epochs along
with the boundary of the interval.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from tudatpy.astro import frame_conversion as fc
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.dynamics import simulator as sim
from tudatpy.math import interpolators as interp

from orbitdet.reproducibility.runtime import RuntimeContext
from orbitdet.simulation import build_integrator_settings, get_propagator_settings

logger = logging.getLogger(__name__)

#: Epochs are matched by rounding to this many decimal places (i.e. a 1 µs
#: tolerance), which is far below any integrator step but well above the
#: floating-point noise of the epoch arithmetic.
_EPOCH_KEY_SCALE = 1e6


@dataclass
class PropagationOutcome:
    """Result of a single propagation, in the form needed for comparison.

    Attributes
    ----------
    label : str
        Human-readable identifier, e.g. ``"RKF78 @ 3600 s (cowell)"``.
    integrator_type : str
        Integrator name used for this run.
    timestep : float
        Fixed integration step [s].
    propagator : str
        Translational propagator formulation, e.g. ``"cowell"``.
    epochs : np.ndarray
        Absolute epochs [s since J2000], sorted, shape ``(N,)``.
    states : np.ndarray
        Cartesian states, shape ``(N, 6)``.
    integration_time : float
        Wall-clock duration of the propagation [s].
    initial_state : np.ndarray
        The state the propagation started from, shape ``(6,)``.
    """

    label: str
    integrator_type: str
    timestep: float
    propagator: str
    epochs: np.ndarray
    states: np.ndarray
    integration_time: float
    initial_state: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)


@dataclass
class ComparisonResult:
    """Per-epoch difference between a test and a reference propagation.

    Attributes
    ----------
    mode : str
        ``"exact"`` when every compared epoch existed in both histories, else
        ``"mixed"`` (some epochs were Lagrange-interpolated).
    epochs : np.ndarray
        Absolute epochs [s since J2000] at which the comparison was made.
    pos_diff : np.ndarray
        Position difference (test - reference) [m], shape ``(N, 3)``.
    vel_diff : np.ndarray
        Velocity difference (test - reference) [m/s], shape ``(N, 3)``.
    rsw_diff : np.ndarray
        ``pos_diff`` rotated into the RSW frame defined by the reference
        state, shape ``(N, 3)``. Radial / along-track / cross-track components.
    pos_diff_norm : np.ndarray
        Euclidean norm of ``pos_diff`` [m].
    vel_diff_norm : np.ndarray
        Euclidean norm of ``vel_diff`` [m/s].
    fractional_pos_diff : np.ndarray
        ``pos_diff_norm`` divided by the reference body's distance to its
        central body (dimensionless).
    metrics : dict[str, float]
        Scalar summary statistics (RMS, max, endpoint values, counts).
    reference, test : PropagationOutcome | None
        The two outcomes that produced this comparison, kept for labelling.
    """

    mode: str
    epochs: np.ndarray
    pos_diff: np.ndarray
    vel_diff: np.ndarray
    rsw_diff: np.ndarray
    pos_diff_norm: np.ndarray
    vel_diff_norm: np.ndarray
    fractional_pos_diff: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    reference: PropagationOutcome | None = field(default=None, repr=False)
    test: PropagationOutcome | None = field(default=None, repr=False)


def run_propagation(
    bodies: env.SystemOfBodies,
    settings: prop_setup.propagator.TranslationalStatePropagatorSettings,
    label: str,
    integrator_type: str,
    timestep: float,
    propagator: str,
) -> PropagationOutcome:
    """Run one propagation and capture its state history and wall-clock time.

    Parameters
    ----------
    bodies : env.SystemOfBodies
        The system of bodies the propagation acts on.
    settings : prop_setup.propagator.TranslationalStatePropagatorSettings
        Propagator settings for this run.
    label, integrator_type, timestep, propagator
        Descriptive metadata stored on the returned outcome.

    Returns
    -------
    PropagationOutcome
        The (sorted) epoch grid and Cartesian state history.
    """
    logger.info("Running propagation: %s", label)
    t0 = time.perf_counter()
    dynamics_simulator = sim.create_dynamics_simulator(bodies, settings)
    integration_time = time.perf_counter() - t0

    state_history = dynamics_simulator.propagation_results.state_history
    epochs = np.array(sorted(state_history.keys()))
    states = np.array([np.asarray(state_history[e]).flatten() for e in epochs])

    logger.info(
        "%s: %d epochs (%.0f s ... %.0f s) in %.2f s.",
        label,
        len(epochs),
        epochs[0],
        epochs[-1],
        integration_time,
    )

    return PropagationOutcome(
        label=label,
        integrator_type=integrator_type,
        timestep=float(timestep),
        propagator=propagator,
        epochs=epochs,
        states=states,
        integration_time=integration_time,
        initial_state=np.array(settings.initial_states, dtype=float).flatten(),
    )


def build_combination_settings(
    cfg,
    ctx: RuntimeContext,
    acceleration_settings,
    integrator_type: str,
    timestep: float,
    propagator: str,
    initial_state,
    dependent_variables_to_save=(),
):
    """Build propagator settings for one integrator + propagator combination.

    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration (for ``integrator`` fallbacks and frames).
    ctx : RuntimeContext
        Runtime context holding the arc epochs.
    acceleration_settings
        Acceleration model map returned by :func:`get_dynamical_model`.
    integrator_type : str
        Integrator name, e.g. ``"RKF78"``.
    timestep : float
        Fixed integration step [s].
    propagator : str
        Translational propagator formulation, e.g. ``"cowell"``.
    initial_state : sequence[float] | None
        Explicit initial state shared by both combinations. When ``None``, the
        state is resolved from the configuration (falling back to SPICE), which
        is appropriate for the first of the two runs.
    dependent_variables_to_save : sequence
        Optional dependent variables to record.

    Returns
    -------
    prop_setup.propagator.TranslationalStatePropagatorSettings
        Propagator settings for the requested combination.
    """
    integrator_settings = build_integrator_settings(timestep, integrator_type, cfg)
    return get_propagator_settings(
        cfg,
        ctx,
        acceleration_settings,
        integrator_settings,
        list(dependent_variables_to_save),
        initial_state=None if initial_state is None else list(initial_state),
        propagator_type=propagator,
    )


def _epoch_keys(epochs: np.ndarray) -> np.ndarray:
    """Round epochs to integer microseconds for exact, robust matching."""
    return np.round(np.asarray(epochs, dtype=float) * _EPOCH_KEY_SCALE).astype(np.int64)


def _select_states(
    outcome: PropagationOutcome,
    query_epochs: np.ndarray,
    interpolator_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``outcome``'s states at ``query_epochs`` plus an "exact" mask.

    Epochs present in the outcome's own history are looked up exactly; the
    remaining epochs are evaluated with a Lagrange interpolator of the given
    order. This keeps the comparison free of interpolation error whenever the
    two timesteps are commensurate, and limits it to the epochs that genuinely
    need it otherwise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(states, exact_mask)`` where ``states`` has shape
        ``(len(query_epochs), 6)`` and ``exact_mask`` is a boolean array.
    """
    lookup = {key: idx for idx, key in enumerate(_epoch_keys(outcome.epochs))}
    query_keys = _epoch_keys(query_epochs)

    states = np.empty((len(query_epochs), outcome.states.shape[1]), dtype=float)
    exact = np.zeros(len(query_epochs), dtype=bool)

    missing = []
    for i, key in enumerate(query_keys):
        idx = lookup.get(int(key))
        if idx is None:
            missing.append(i)
        else:
            states[i] = outcome.states[idx]
            exact[i] = True

    if missing:
        # Boundary handling degrades near the ends of the interval, so the
        # caller is expected to buffer the comparison on both sides.
        settings = interp.lagrange_interpolation(interpolator_order)
        interpolator = interp.create_one_dimensional_vector_interpolator(
            {float(e): np.asarray(s, dtype=float) for e, s in zip(outcome.epochs, outcome.states)},
            settings,
        )
        for i in missing:
            states[i] = np.asarray(interpolator.interpolate(float(query_epochs[i]))).flatten()

    return states, exact


def _rsw_components(ref_states: np.ndarray, rel_pos: np.ndarray) -> np.ndarray:
    """Rotate a relative position into the RSW frame of each reference state.

    The RSW frame is defined by the reference body's own state; its radial axis
    points away from the central body, its along-track axis lies in the orbital
    plane in the direction of motion, and the cross-track axis completes the
    right-handed set. Degenerate states (position parallel to velocity) yield
    ``NaN`` rather than a bogus frame.
    """
    components = np.full_like(rel_pos, np.nan, dtype=float)
    for i, state in enumerate(ref_states):
        if np.linalg.norm(np.cross(state[:3], state[3:])) == 0.0:
            continue
        components[i] = fc.inertial_to_rsw_rotation_matrix(state) @ rel_pos[i]
    return components


def compare_outcomes(
    reference: PropagationOutcome,
    test: PropagationOutcome,
    buffer: int = 0,
    interpolator_order: int = 8,
) -> ComparisonResult:
    """Compare a test propagation against a reference propagation.

    The comparison is made on the coarser of the two epoch grids (see module
    docstring). ``buffer`` epochs are dropped from each end of the compared
    series to exclude boundary effects and the run-up transient of the coarser
    integrator.

    Parameters
    ----------
    reference : PropagationOutcome
        The trusted, high-accuracy trajectory.
    test : PropagationOutcome
        The trajectory under investigation.
    buffer : int, optional
        Number of epochs to discard at each end of the compared series.
    interpolator_order : int, optional
        Lagrange order used for epochs that are not common to both histories.

    Returns
    -------
    ComparisonResult
        Per-epoch differences and scalar summary metrics.

    Raises
    ------
    ValueError
        If the two histories do not support a comparison (empty overlap, or
        fewer epochs than the requested buffer).
    """
    if reference.n_epochs < 2 or test.n_epochs < 2:
        raise ValueError("Both propagations need at least two epochs to be compared.")

    # Compare on the coarser grid: fewer epochs, and the finer trajectory is
    # the one that gets interpolated (its smaller step keeps that error small).
    if reference.n_epochs <= test.n_epochs:
        query_epochs = reference.epochs
    else:
        query_epochs = test.epochs

    # Only keep query epochs that lie within both histories' common span.
    span_start = max(reference.epochs[0], test.epochs[0])
    span_end = min(reference.epochs[-1], test.epochs[-1])
    inside = (query_epochs >= span_start) & (query_epochs <= span_end)
    query_epochs = query_epochs[inside]

    if len(query_epochs) < 2:
        raise ValueError(
            "The two histories overlap in fewer than two epochs; check the arc settings."
        )

    ref_states, ref_exact = _select_states(reference, query_epochs, interpolator_order)
    test_states, test_exact = _select_states(test, query_epochs, interpolator_order)

    logger.info(
        "Comparing on %d epochs: %d exact, %d with a Lagrange-%d evaluation "
        "(%s trajectory interpolated).",
        len(query_epochs),
        int(np.count_nonzero(ref_exact & test_exact)),
        int(np.count_nonzero(~(ref_exact & test_exact))),
        interpolator_order,
        "reference" if reference.n_epochs > test.n_epochs else "test",
    )

    # --- Discard boundary epochs ---
    if buffer > 0:
        if len(query_epochs) <= 2 * buffer:
            raise ValueError(
                f"Buffer of {buffer} epochs leaves nothing to compare out of "
                f"{len(query_epochs)} epochs."
            )
        slice_ = slice(buffer, len(query_epochs) - buffer)
        query_epochs = query_epochs[slice_]
        ref_states = ref_states[slice_]
        test_states = test_states[slice_]
        ref_exact = ref_exact[slice_]
        test_exact = test_exact[slice_]

    # --- Differences ---
    pos_diff = test_states[:, :3] - ref_states[:, :3]
    vel_diff = test_states[:, 3:] - ref_states[:, 3:]

    pos_diff_norm = np.linalg.norm(pos_diff, axis=1)
    vel_diff_norm = np.linalg.norm(vel_diff, axis=1)

    ref_radius = np.linalg.norm(ref_states[:, :3], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fractional_pos_diff = np.where(ref_radius > 0.0, pos_diff_norm / ref_radius, np.nan)

    rsw_diff = _rsw_components(ref_states, pos_diff)

    # Count exactness after the buffer slice so it always describes the epochs
    # that were actually compared.
    both_exact = ref_exact & test_exact
    n_exact = int(np.count_nonzero(both_exact))
    mode = "exact" if n_exact == len(query_epochs) else "mixed"

    metrics = _summarize(
        epochs=query_epochs,
        pos_diff_norm=pos_diff_norm,
        vel_diff_norm=vel_diff_norm,
        fractional_pos_diff=fractional_pos_diff,
        n_epochs_total=max(reference.n_epochs, test.n_epochs),
        n_exact=n_exact,
        pos_diff_exact=pos_diff_norm[both_exact],
        vel_diff_exact=vel_diff_norm[both_exact],
    )
    metrics["reference_integration_time"] = reference.integration_time
    metrics["test_integration_time"] = test.integration_time
    metrics["integration_time_ratio"] = (
        test.integration_time / reference.integration_time
        if reference.integration_time > 0.0
        else float("nan")
    )
    metrics["mode_is_exact"] = float(mode == "exact")

    # Initial states must be identical, otherwise the difference is dominated
    # by an offset rather than by accumulated integration error.
    initial_offset = np.linalg.norm(test.initial_state[:3] - reference.initial_state[:3])
    metrics["initial_state_offset"] = float(initial_offset)
    if not np.isfinite(initial_offset) or initial_offset > 1e-6:
        logger.warning(
            "Initial states differ by %.6e m; the comparison includes that offset.",
            initial_offset,
        )

    result = ComparisonResult(
        mode=mode,
        epochs=query_epochs,
        pos_diff=pos_diff,
        vel_diff=vel_diff,
        rsw_diff=rsw_diff,
        pos_diff_norm=pos_diff_norm,
        vel_diff_norm=vel_diff_norm,
        fractional_pos_diff=fractional_pos_diff,
        metrics=metrics,
        reference=reference,
        test=test,
    )
    return result


def _summarize(
    epochs: np.ndarray,
    pos_diff_norm: np.ndarray,
    vel_diff_norm: np.ndarray,
    fractional_pos_diff: np.ndarray,
    n_epochs_total: int,
    n_exact: int,
    pos_diff_exact: np.ndarray | None = None,
    vel_diff_exact: np.ndarray | None = None,
) -> dict[str, float]:
    """Build the scalar summary metrics reported for every comparison.

    The overall RMS/max describe the whole compared series. When part of the
    series needed interpolation, the extra ``*_exact`` metrics give the same
    statistics restricted to the epochs compared without any interpolation, so
    the interpolation artifact can be separated from genuine integration drift.
    """

    def rms(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")

    def maximum(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else float("nan")

    metrics = {
        "n_epochs_total": float(n_epochs_total),
        "n_epochs_compared": float(len(epochs)),
        "n_epochs_exact": float(n_exact),
        "epoch_first": float(epochs[0]),
        "epoch_last": float(epochs[-1]),
        "pos_rms": rms(pos_diff_norm),
        "pos_max": maximum(pos_diff_norm),
        "pos_start": float(pos_diff_norm[0]),
        "pos_end": float(pos_diff_norm[-1]),
        "vel_rms": rms(vel_diff_norm),
        "vel_max": maximum(vel_diff_norm),
        "vel_end": float(vel_diff_norm[-1]),
        "fractional_pos_rms": rms(fractional_pos_diff),
        "fractional_pos_max": maximum(fractional_pos_diff),
    }

    if pos_diff_exact is not None:
        metrics["pos_rms_exact_only"] = rms(pos_diff_exact)
        metrics["pos_max_exact_only"] = maximum(pos_diff_exact)
    if vel_diff_exact is not None:
        metrics["vel_rms_exact_only"] = rms(vel_diff_exact)

    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    """Format a metrics mapping as an indented, multi-line string for logging."""
    return "\n".join(f"    {name}: {value:.6g}" for name, value in metrics.items())
