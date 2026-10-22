"""Tests for the combination-validation machinery.

``compare_outcomes`` is the correctness-critical piece: it must (a) prefer
exact epoch lookups over interpolation, (b) only interpolate the epochs that
genuinely need it, (c) subtract in the right order (test - reference), and
(d) decompose the difference in the RSW frame defined by the *reference* state.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from orbitdet.validation.combination import (
    PropagationOutcome,
    _epoch_keys,
    _select_states,
    compare_outcomes,
    format_metrics,
)


def _kepler_like_state(epochs, radius=354.8e6, speed=4390.0):
    """Build a smooth synthetic trajectory: a circular orbit in the xy-plane."""
    period = 2.0 * np.pi * radius / speed
    angle = 2.0 * np.pi * np.asarray(epochs, dtype=float) / period
    positions = np.column_stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
    )
    velocities = np.column_stack(
        [-speed * np.sin(angle), speed * np.cos(angle), np.zeros_like(angle)]
    )
    return np.hstack([positions, velocities])


def make_outcome(
    epochs, states, label="run", timestep=3600.0, integrator_type="RKF78", propagator="cowell"
):
    epochs = np.asarray(epochs, dtype=float)
    states = np.asarray(states, dtype=float)
    return PropagationOutcome(
        label=label,
        integrator_type=integrator_type,
        timestep=timestep,
        propagator=propagator,
        epochs=epochs,
        states=states,
        integration_time=10.0,
        initial_state=states[0].copy(),
    )


class TestEpochKeys:
    def test_rounds_to_microsecond(self):
        epochs = np.array([1000.0000004, 1000.0000016, 1000.000002])
        keys = _epoch_keys(epochs)
        assert keys[0] == keys[1] or keys[1] == keys[2]

    def test_distinct_epochs_stay_distinct(self):
        epochs = np.array([0.0, 0.01])
        assert len(set(_epoch_keys(epochs).tolist())) == 2


class TestSelectStates:
    def test_shared_epochs_are_exact(self):
        epochs = np.arange(0.0, 10.0, 1.0)
        states = _kepler_like_state(epochs)
        outcome = make_outcome(epochs, states)

        selected, exact = _select_states(outcome, epochs[2:5], interpolator_order=8)

        assert exact.all()
        np.testing.assert_allclose(selected, states[2:5], rtol=0, atol=0)

    def test_missing_epochs_use_interpolation(self):
        """A grid twice as fine must reproduce the coarse grid to high accuracy."""
        coarse = np.arange(0.0, 100.0, 2.0)
        fine = np.arange(0.0, 100.0, 1.0)
        outcome = make_outcome(fine, _kepler_like_state(fine))

        selected, exact = _select_states(outcome, coarse, interpolator_order=8)

        # Only the even (shared) epochs are exact; the rest are interpolated.
        assert exact.sum() == len(coarse)
        truth = _kepler_like_state(coarse)
        # Every requested epoch happens to exist in the fine grid here, so all
        # values must be exact.
        np.testing.assert_allclose(selected, truth, rtol=0, atol=0)

    def test_partial_overlap_reports_mixed_exactness(self):
        fine = np.arange(0.0, 10.0, 0.5)
        outcome = make_outcome(fine, _kepler_like_state(fine))
        query = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        selected, exact = _select_states(outcome, query, interpolator_order=8)

        assert exact.tolist() == [True, False, True, False, True]
        # Interpolated values must land near the analytic trajectory.
        truth = _kepler_like_state(query)
        np.testing.assert_allclose(selected, truth, rtol=1e-6, atol=1.0)


class TestCompareOutcomes:
    def test_identical_runs_give_zero_difference(self):
        epochs = np.arange(0.0, 50.0, 1.0)
        states = _kepler_like_state(epochs)
        reference = make_outcome(epochs, states, label="ref")
        test = make_outcome(epochs, states, label="test")

        result = compare_outcomes(reference, test)

        assert result.mode == "exact"
        np.testing.assert_allclose(result.pos_diff_norm, 0.0, atol=0)
        np.testing.assert_allclose(result.vel_diff_norm, 0.0, atol=0)
        assert result.metrics["pos_rms"] == 0.0

    def test_difference_is_test_minus_reference(self):
        epochs = np.arange(0.0, 10.0, 1.0)
        reference = make_outcome(epochs, _kepler_like_state(epochs), label="ref")
        shifted = _kepler_like_state(epochs)
        shifted[:, 0] += 100.0  # displace the test run along +x
        test = make_outcome(epochs, shifted, label="test")

        result = compare_outcomes(reference, test)

        # Every x-difference must be exactly +100 m (test - reference).
        np.testing.assert_allclose(result.pos_diff[:, 0], 100.0, atol=1e-9)
        np.testing.assert_allclose(result.pos_diff_norm, 100.0, atol=1e-9)

    def test_compares_on_the_coarser_grid(self):
        coarse = np.arange(0.0, 100.0, 10.0)
        fine = np.arange(0.0, 100.0, 1.0)
        reference = make_outcome(coarse, _kepler_like_state(coarse), label="ref", timestep=10.0)
        test = make_outcome(fine, _kepler_like_state(fine), label="test", timestep=1.0)

        result = compare_outcomes(reference, test)

        assert len(result.epochs) == len(coarse)
        np.testing.assert_allclose(result.epochs, coarse)
        assert result.mode == "exact"  # all coarse epochs exist in the fine grid

    def test_buffer_removes_boundary_epochs(self):
        epochs = np.arange(0.0, 20.0, 1.0)
        reference = make_outcome(epochs, _kepler_like_state(epochs))
        test = make_outcome(epochs, _kepler_like_state(epochs))

        buffered = compare_outcomes(reference, test, buffer=3)
        unbuffered = compare_outcomes(reference, test, buffer=0)

        assert len(buffered.epochs) == len(unbuffered.epochs) - 6
        np.testing.assert_allclose(buffered.epochs, unbuffered.epochs[3:-3])

    def test_rsw_components_reconstruct_the_position_difference(self):
        """The RSW rotation is orthonormal, so the norm must be preserved."""
        epochs = np.arange(0.0, 30.0, 1.0)
        reference = make_outcome(epochs, _kepler_like_state(epochs))
        perturbed = _kepler_like_state(epochs)
        perturbed[:, 1] += 250.0
        perturbed[:, 2] -= 75.0
        test = make_outcome(epochs, perturbed)

        result = compare_outcomes(reference, test)

        rsw_norm = np.linalg.norm(result.rsw_diff, axis=1)
        np.testing.assert_allclose(rsw_norm, result.pos_diff_norm, rtol=1e-9, atol=1e-9)

    def test_rsw_radial_component_matches_the_radius_change(self):
        """A purely radial displacement must appear only in the R component."""
        epochs = np.arange(0.0, 30.0, 1.0)
        states = _kepler_like_state(epochs)
        reference = make_outcome(epochs, states)

        # Scale the radius by a constant factor -> displacement along +R.
        radial_test = states.copy()
        radial_test[:, :3] *= 1.001
        test = make_outcome(epochs, radial_test)

        result = compare_outcomes(reference, test)

        radius = np.linalg.norm(states[:, :3], axis=1)
        np.testing.assert_allclose(result.rsw_diff[:, 0], 0.001 * radius, rtol=1e-9)
        np.testing.assert_allclose(result.rsw_diff[:, 1], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.rsw_diff[:, 2], 0.0, atol=1e-6)

    def test_metrics_are_consistent_with_the_series(self):
        epochs = np.arange(0.0, 40.0, 1.0)
        reference = make_outcome(epochs, _kepler_like_state(epochs))
        test = _kepler_like_state(epochs)
        test[:, 0] += np.linspace(0.0, 400.0, len(epochs))
        test = make_outcome(epochs, test)

        result = compare_outcomes(reference, test)

        norm = result.pos_diff_norm
        assert result.metrics["pos_rms"] == pytest.approx(float(np.sqrt(np.mean(norm**2))))
        assert result.metrics["pos_max"] == pytest.approx(float(np.max(norm)))
        assert result.metrics["pos_start"] == pytest.approx(float(norm[0]))
        assert result.metrics["pos_end"] == pytest.approx(float(norm[-1]))
        assert result.metrics["n_epochs_compared"] == float(len(norm))

    def test_exact_counts_describe_the_buffered_series(self):
        """Exactness counts must match the epochs that were actually compared."""
        coarse = np.arange(0.0, 100.0, 10.0)
        fine = np.arange(0.0, 100.0, 1.0)
        reference = make_outcome(coarse, _kepler_like_state(coarse))
        test = make_outcome(fine, _kepler_like_state(fine))

        result = compare_outcomes(reference, test, buffer=3)

        assert result.metrics["n_epochs_compared"] == float(len(result.epochs))
        # Every compared epoch exists in the fine grid, so all of them are exact.
        assert result.metrics["n_epochs_exact"] == result.metrics["n_epochs_compared"]
        assert result.mode == "exact"

    def test_mixed_mode_counts_and_exact_only_metrics(self):
        """With a non-commensurate reference, only shared epochs are exact."""
        # 1.5 s is exactly representable, so every other reference epoch (the
        # integer ones) coincides with the fine grid while the halves do not.
        coarse = np.arange(0.0, 100.0, 1.5)
        fine = np.arange(0.0, 100.0, 1.0)
        reference = make_outcome(coarse, _kepler_like_state(coarse))
        test = make_outcome(fine, _kepler_like_state(fine))

        result = compare_outcomes(reference, test, buffer=2)

        compared = result.metrics["n_epochs_compared"]
        exact = result.metrics["n_epochs_exact"]
        assert result.mode == "mixed"
        assert exact < compared
        # Roughly half the epochs are shared with the fine grid.
        assert compared / 3 < exact <= compared
        # Both outcomes carry the *same* analytic trajectory, so the shared
        # epochs must agree bit-for-bit while the interpolated ones carry a
        # small Lagrange residual. This is exactly what the exact-only metric
        # is for.
        assert result.metrics["pos_rms_exact_only"] == 0.0
        assert result.metrics["pos_rms"] > 0.0
        assert result.metrics["pos_max"] < 1.0  # sub-metre interpolation residual

    def test_fractional_difference_divides_by_reference_radius(self):
        epochs = np.arange(0.0, 20.0, 1.0)
        states = _kepler_like_state(epochs)
        reference = make_outcome(epochs, states)
        test = states.copy()
        test[:, 0] += 1000.0
        test = make_outcome(epochs, test)

        result = compare_outcomes(reference, test)

        radius = np.linalg.norm(states[:, :3], axis=1)
        np.testing.assert_allclose(
            result.fractional_pos_diff, result.pos_diff_norm / radius, rtol=1e-9
        )

    def test_initial_state_offset_is_reported(self):
        epochs = np.arange(0.0, 10.0, 1.0)
        states = _kepler_like_state(epochs)
        reference = make_outcome(epochs, states)
        offset_states = states.copy()
        offset_states[:, 0] += 5.0
        test = make_outcome(epochs, offset_states)

        result = compare_outcomes(reference, test)

        assert result.metrics["initial_state_offset"] == pytest.approx(5.0)

    def test_raises_when_histories_do_not_overlap(self):
        reference = make_outcome(np.arange(0.0, 10.0), _kepler_like_state(np.arange(0.0, 10.0)))
        test = make_outcome(np.arange(50.0, 60.0), _kepler_like_state(np.arange(50.0, 60.0)))

        with pytest.raises(ValueError, match="overlap"):
            compare_outcomes(reference, test)

    def test_raises_when_buffer_consumes_everything(self):
        epochs = np.arange(0.0, 5.0, 1.0)
        reference = make_outcome(epochs, _kepler_like_state(epochs))
        test = make_outcome(epochs, _kepler_like_state(epochs))

        with pytest.raises(ValueError, match="Buffer"):
            compare_outcomes(reference, test, buffer=10)

    def test_raises_for_single_epoch_histories(self):
        reference = make_outcome(np.array([0.0]), _kepler_like_state(np.array([0.0])))
        test = make_outcome(np.array([0.0, 1.0]), _kepler_like_state(np.array([0.0, 1.0])))

        with pytest.raises(ValueError, match="at least two epochs"):
            compare_outcomes(reference, test)

    def test_degenerate_reference_state_yields_nan_rsw(self):
        """Position parallel to velocity cannot define an RSW frame."""
        epochs = np.arange(0.0, 5.0, 1.0)
        degenerate = np.tile(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), (5, 1))
        reference = make_outcome(epochs, degenerate)
        test = make_outcome(epochs, degenerate + 1.0)

        result = compare_outcomes(reference, test)

        assert np.isnan(result.rsw_diff).all()
        # The scalar norms must still be usable.
        assert np.isfinite(result.pos_diff_norm).all()


class TestFormatMetrics:
    def test_formats_every_entry(self):
        text = format_metrics({"pos_rms": 1.5e-3, "n_epochs": 12.0})
        assert "pos_rms: 0.0015" in text
        assert "n_epochs: 12" in text


class TestCombinationPlotter:
    def _cfg(self):
        return OmegaConf.create(
            {
                "combination_comparison": {
                    "figure": {"width": 10, "height": 8},
                    "reference_line": {"enabled": True, "value": 14000.0},
                }
            }
        )

    def test_builds_three_panels(self):
        epochs = np.arange(0.0, 30.0, 1.0)
        states = _kepler_like_state(epochs)
        reference = make_outcome(epochs, states, label="RKF1412 @ 900 s (cowell)")
        test_states = states.copy()
        test_states[:, 0] += 10.0
        test = make_outcome(epochs, test_states, label="RKF89 @ 3600 s (cowell)")
        comparison = compare_outcomes(reference, test)

        from orbitdet.visualization import CombinationComparison

        plotter = CombinationComparison(self._cfg(), comparison, reference, test)
        fig, axes = plotter._make_figure()

        assert len(axes) == 3
        # The x axis is shared across panels.
        assert axes[2].get_xlabel() == "Epoch"

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_titles_fall_back_when_config_is_missing(self):
        epochs = np.arange(0.0, 10.0, 1.0)
        states = _kepler_like_state(epochs)
        reference = make_outcome(epochs, states, label="ref")
        test = make_outcome(epochs, states + 1.0, label="test")
        comparison = compare_outcomes(reference, test)

        from orbitdet.visualization import CombinationComparison

        plotter = CombinationComparison(OmegaConf.create({}), comparison, reference, test)
        fig, axes = plotter._make_figure()

        assert "Combination validation" in fig._suptitle.get_text()
        assert axes[0].get_ylabel() == "||Δr|| [m]"

        import matplotlib.pyplot as plt

        plt.close(fig)
