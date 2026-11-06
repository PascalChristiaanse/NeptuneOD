"""Tests for outlier rejection strategies.

Both ``ResidualThresholdOutlier`` and ``EpochFilterOutlier`` rely on Tudat's
``create_filtered_observation_set`` and ``observation_filter``, which are
mocked here to simulate filtering behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from orbitdet.observations.outlier_rejection.strategies import (
    EpochFilterOutlier,
    ResidualThresholdOutlier,
)

# ===========================================================================
# Mock helpers
# ===========================================================================


def _make_mock_time(epoch: float):
    return SimpleNamespace(to_float=lambda: epoch)


def _make_mock_obs_set(
    residuals: np.ndarray | None = None,
    times: list[float] | None = None,
) -> SimpleNamespace:
    """Create a mock SingleObservationSet."""
    if times is None:
        times = [0.0, 100.0, 200.0]
    if residuals is None:
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]])
    mock_times = [_make_mock_time(t) for t in times]
    return SimpleNamespace(
        residuals=residuals,
        observation_times=mock_times,
    )


def _filter_by_residual_threshold(
    obs_set: SimpleNamespace,
    threshold_rad: float,
    opposite: bool = False,
) -> SimpleNamespace:
    """Simulate Tudat's residual filtering on our mock."""
    residuals = np.asarray(obs_set.residuals)
    times = [t.to_float() for t in obs_set.observation_times]
    max_residual = np.max(np.abs(residuals), axis=1)
    if opposite:
        keep_mask = max_residual > threshold_rad
    else:
        keep_mask = max_residual <= threshold_rad
    filtered_residuals = residuals[keep_mask]
    filtered_times = [t for t, keep in zip(times, keep_mask) if keep]
    filtered_mock_times = [_make_mock_time(t) for t in filtered_times]
    return SimpleNamespace(
        residuals=filtered_residuals,
        observation_times=filtered_mock_times,
    )


def _filter_by_epochs(
    obs_set: SimpleNamespace,
    epochs_to_remove: list[float],
    opposite: bool = False,
) -> SimpleNamespace:
    """Simulate Tudat's epoch filtering on our mock."""
    times = [t.to_float() for t in obs_set.observation_times]
    residuals = np.asarray(obs_set.residuals)
    if opposite:
        keep_mask = [t in epochs_to_remove for t in times]
    else:
        keep_mask = [t not in epochs_to_remove for t in times]
    filtered_residuals = residuals[keep_mask]
    filtered_times = [t for t, keep in zip(times, keep_mask) if keep]
    filtered_mock_times = [_make_mock_time(t) for t in filtered_times]
    return SimpleNamespace(
        residuals=filtered_residuals,
        observation_times=filtered_mock_times,
    )


@pytest.fixture(autouse=True)
def _patch_tudat_filtering(monkeypatch):
    """Mock Tudat's observation_filter and create_filtered_observation_set.

    Since tudatpy's functions are C extensions (can't be monkeypatched
    directly), we patch the module-level references in the strategies module.
    """
    import orbitdet.observations.outlier_rejection.strategies as s

    # Track filter calls for inspection
    filter_calls = []

    def mock_observation_filter(filter_type, value, filter_out=True, use_opposite_condition=False):
        filter_calls.append(
            {
                "type": filter_type,
                "value": value,
                "filter_out": filter_out,
                "use_opposite_condition": use_opposite_condition,
            }
        )
        return SimpleNamespace(
            filter_type=filter_type,
            value=value,
            filter_out=filter_out,
            use_opposite_condition=use_opposite_condition,
        )

    def mock_create_filtered_observation_set(obs_set, filter_obj):
        if filter_obj.filter_type.name == "residual_filtering":
            return _filter_by_residual_threshold(
                obs_set,
                filter_obj.value,
                opposite=filter_obj.use_opposite_condition,
            )
        elif filter_obj.filter_type.name == "epochs_filtering":
            return _filter_by_epochs(
                obs_set,
                filter_obj.value,
                opposite=filter_obj.use_opposite_condition,
            )
        return obs_set

    # Mock the filter type enum
    mock_filter_type = SimpleNamespace(
        residual_filtering=SimpleNamespace(name="residual_filtering"),
        epochs_filtering=SimpleNamespace(name="epochs_filtering"),
    )

    # Patch at the module level where strategies.py imports them
    monkeypatch.setattr(
        s,
        "obs_proc",
        SimpleNamespace(
            observation_filter=mock_observation_filter,
            ObservationFilterType=mock_filter_type,
        ),
    )
    monkeypatch.setattr(
        s,
        "obs",
        SimpleNamespace(
            create_filtered_observation_set=mock_create_filtered_observation_set,
        ),
    )


# ===========================================================================
# ResidualThresholdOutlier
# ===========================================================================


class TestResidualThresholdOutlier:
    def test_all_below_threshold(self):
        """All residuals below threshold → all accepted."""
        residuals = np.array([[0.5e-6, 0.3e-6], [0.4e-6, 0.6e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = ResidualThresholdOutlier(threshold_arcsec=1.0)
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 0
        assert metadata["n_total"] == 2
        assert metadata["rejected_epochs"] == []

    def test_some_above_threshold(self):
        """Observations with residual > threshold are rejected."""
        residuals = np.array(
            [
                [0.1e-6, 0.2e-6],  # below threshold
                [5.0e-5, 6.0e-5],  # above threshold (~10 arcsec)
                [0.3e-6, 0.4e-6],  # below threshold
            ]
        )
        times = [0.0, 100.0, 200.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = ResidualThresholdOutlier(threshold_arcsec=1.0)
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 1
        assert metadata["n_total"] == 3
        assert metadata["rejected_epochs"] == [100.0]

    def test_all_above_threshold(self):
        """All residuals above threshold → all rejected."""
        residuals = np.array(
            [
                [5.0e-5, 6.0e-5],
                [7.0e-5, 8.0e-5],
            ]
        )
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = ResidualThresholdOutlier(threshold_arcsec=0.1)
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 0
        assert metadata["n_rejected"] == 2
        assert metadata["n_total"] == 2

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold_arcsec must be positive"):
            ResidualThresholdOutlier(threshold_arcsec=-1.0)

    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold_arcsec must be positive"):
            ResidualThresholdOutlier(threshold_arcsec=0.0)

    def test_default_threshold(self):
        strategy = ResidualThresholdOutlier()
        assert strategy is not None

    def test_metadata_strategy_name(self):
        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = ResidualThresholdOutlier(threshold_arcsec=1.5)
        _, metadata = strategy.apply(obs_set, bodies)

        assert metadata["strategy"] == "residual_threshold"
        assert metadata["threshold_arcsec"] == 1.5

    def test_rejected_epochs_are_correct(self):
        """Only the epochs of rejected observations should be listed."""
        residuals = np.array(
            [
                [1e-7, 2e-7],  # OK
                [5e-5, 6e-5],  # bad (~10 arcsec)
                [1e-7, 2e-7],  # OK
                [7e-5, 8e-5],  # bad (~14 arcsec)
            ]
        )
        times = [0.0, 100.0, 200.0, 300.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = ResidualThresholdOutlier(threshold_arcsec=1.0)
        _, metadata = strategy.apply(obs_set, bodies)

        assert metadata["rejected_epochs"] == [100.0, 300.0]


# ===========================================================================
# EpochFilterOutlier
# ===========================================================================


class TestEpochFilterOutlier:
    def test_empty_epoch_list_no_removal(self):
        """No epochs to remove → all accepted."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier(epochs_to_remove=[])
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 0
        assert metadata["n_total"] == 2

    def test_specific_epochs_removed(self):
        """Matching epochs are removed."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]])
        times = [0.0, 100.0, 200.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier(epochs_to_remove=[100.0])
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 1
        assert metadata["n_total"] == 3
        assert metadata["rejected_epochs"] == [100.0]

    def test_non_matching_epochs_not_removed(self):
        """Epochs not in the removal list are kept."""
        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier(epochs_to_remove=[999.0])
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 1
        assert metadata["n_rejected"] == 0

    def test_multiple_epochs_removed(self):
        """Multiple epochs can be removed at once."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6], [7e-6, 8e-6]])
        times = [0.0, 100.0, 200.0, 300.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier(epochs_to_remove=[100.0, 300.0])
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 2
        assert sorted(metadata["rejected_epochs"]) == [100.0, 300.0]

    def test_all_epochs_removed(self):
        """All epochs match → all rejected."""
        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier(epochs_to_remove=[0.0])
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 0
        assert metadata["n_rejected"] == 1

    def test_default_constructor_no_removal(self):
        """Default constructor (no epochs) → no removal."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier()
        filtered_set, metadata = strategy.apply(obs_set, bodies)

        assert metadata["n_accepted"] == 2
        assert metadata["n_rejected"] == 0

    def test_metadata_strategy_name(self):
        residuals = np.array([[1e-6, 2e-6]])
        times = [0.0]
        obs_set = _make_mock_obs_set(residuals, times)
        bodies = MagicMock()

        strategy = EpochFilterOutlier(epochs_to_remove=[0.0])
        _, metadata = strategy.apply(obs_set, bodies)

        assert metadata["strategy"] == "epoch_filter"


class TestEpochFilterTolerance:
    """A configured epoch is a decimal literal; the set's epoch is a float.

    The two are rarely bit-identical, so matching must be by tolerance.  Without
    it the filter silently removes nothing and the observation survives into the
    propagation, where it can fall outside the ephemeris span.
    """

    # The exact pair from the failure that motivated the tolerance: a config
    # value and the epoch Tudat actually held, 0.95 microseconds apart.
    CONFIGURED = -1162202738.851083
    ACTUAL = -1162202738.851084

    def test_sub_microsecond_difference_still_matches(self):
        obs_set = _make_mock_obs_set(np.zeros((1, 2)), [self.ACTUAL])
        strategy = EpochFilterOutlier(epochs_to_remove=[self.CONFIGURED])

        _, metadata = strategy.apply(obs_set, MagicMock())

        assert metadata["n_rejected"] == 1
        assert metadata["n_accepted"] == 0

    def test_matched_epoch_is_the_observations_own_value(self):
        # The downstream Tudat filter matches exactly, so the resolved epoch must
        # be the observation's value, not the configured one.
        strategy = EpochFilterOutlier(epochs_to_remove=[self.CONFIGURED])
        matched, unmatched = strategy._resolve_epochs([self.ACTUAL, 0.0])

        assert matched == [self.ACTUAL]
        assert unmatched == []

    def test_difference_is_warned_about(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        strategy = EpochFilterOutlier(epochs_to_remove=[self.CONFIGURED])
        strategy._resolve_epochs([self.ACTUAL])

        assert "difference of" in caplog.text
        assert "tolerance" in caplog.text

    def test_exact_match_does_not_warn(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        strategy = EpochFilterOutlier(epochs_to_remove=[100.0])
        strategy._resolve_epochs([100.0])

        assert caplog.text == ""

    def test_epoch_beyond_tolerance_is_not_matched(self):
        # 1 ms is the default; 10 ms away must not match.
        obs_set = _make_mock_obs_set(np.zeros((1, 2)), [100.0])
        strategy = EpochFilterOutlier(epochs_to_remove=[100.01])

        _, metadata = strategy.apply(obs_set, MagicMock())

        assert metadata["n_rejected"] == 0

    def test_unmatched_epoch_warns_with_the_nearest_gap(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        obs_set = _make_mock_obs_set(np.zeros((1, 2)), [100.0])
        strategy = EpochFilterOutlier(epochs_to_remove=[100.5])

        _, metadata = strategy.apply(obs_set, MagicMock())

        assert metadata["n_rejected"] == 0
        assert "matched no observation" in caplog.text
        assert "0.5" in caplog.text

    def test_custom_tolerance_is_honoured(self):
        obs_set = _make_mock_obs_set(np.zeros((1, 2)), [100.0])
        strategy = EpochFilterOutlier(epochs_to_remove=[100.01], tolerance_seconds=0.1)

        _, metadata = strategy.apply(obs_set, MagicMock())

        assert metadata["n_rejected"] == 1

    def test_zero_tolerance_requires_an_exact_match(self):
        obs_set = _make_mock_obs_set(np.zeros((1, 2)), [self.ACTUAL])
        strategy = EpochFilterOutlier(epochs_to_remove=[self.CONFIGURED], tolerance_seconds=0.0)

        _, metadata = strategy.apply(obs_set, MagicMock())

        assert metadata["n_rejected"] == 0

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            EpochFilterOutlier(tolerance_seconds=-1.0)

    def test_nearest_of_several_observations_wins(self):
        strategy = EpochFilterOutlier(epochs_to_remove=[100.0004])
        matched, _ = strategy._resolve_epochs([99.0, 100.0, 101.0])

        assert matched == [100.0]

    def test_empty_set_reports_every_epoch_unmatched(self):
        strategy = EpochFilterOutlier(epochs_to_remove=[1.0, 2.0])
        matched, unmatched = strategy._resolve_epochs([])

        assert matched == []
        assert unmatched == [1.0, 2.0]
