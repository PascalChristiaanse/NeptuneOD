"""Tests for weighting strategy helpers and concrete strategies.

Shared helpers (``_rms``, ``_interleave_weights``, ``_residuals_array``,
``_build_weights_df``) are pure NumPy and tested directly.

Concrete strategies (``IDv2Weight``, ``HybridGeometricWeight``) require
mocked ``SingleObservationSet`` objects.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from orbitdet.observations.weighting.grouping import Group, GroupList
from orbitdet.observations.weighting.strategies import (
    IDv2Weight,
    _build_weights_df,
    _interleave_weights,
    _residuals_array,
    _rms,
)

# Re-import for the hybrid strategy
from orbitdet.observations.weighting.strategies import HybridGeometricWeight

# ===========================================================================
# Shared helpers
# ===========================================================================


class TestRMS:
    def test_basic_rms(self):
        values = np.array([1.0, 2.0, 3.0])
        expected = np.sqrt(np.mean(np.square(values)))
        assert _rms(values) == pytest.approx(expected)

    def test_all_nan_returns_zero(self):
        assert _rms(np.array([np.nan, np.nan])) == 0.0

    def test_empty_array_returns_zero(self):
        assert _rms(np.array([])) == 0.0

    def test_mixed_finite_and_nan(self):
        values = np.array([3.0, np.nan, 4.0])
        expected = np.sqrt(np.mean(np.square([3.0, 4.0])))
        assert _rms(values) == pytest.approx(expected)

    def test_single_value(self):
        assert _rms(np.array([5.0])) == pytest.approx(5.0)

    def test_negative_values(self):
        values = np.array([-2.0, -4.0])
        expected = np.sqrt(np.mean(np.square(values)))
        assert _rms(values) == pytest.approx(expected)


class TestInterleaveWeights:
    def test_basic_interleave(self):
        ra = np.array([1.0, 2.0, 3.0])
        dec = np.array([4.0, 5.0, 6.0])
        result = _interleave_weights(ra, dec)
        expected = np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_single_element(self):
        ra = np.array([10.0])
        dec = np.array([20.0])
        result = _interleave_weights(ra, dec)
        np.testing.assert_array_equal(result, [10.0, 20.0])

    def test_empty_arrays(self):
        ra = np.array([])
        dec = np.array([])
        result = _interleave_weights(ra, dec)
        assert len(result) == 0

    def test_ra_at_even_indices(self):
        ra = np.array([1.0, 2.0])
        dec = np.array([3.0, 4.0])
        result = _interleave_weights(ra, dec)
        assert result[0] == 1.0  # RA at even index
        assert result[2] == 2.0  # RA at even index
        assert result[1] == 3.0  # DEC at odd index
        assert result[3] == 4.0  # DEC at odd index


class TestResidualsArray:
    def test_2d_residuals(self):
        obs_set = SimpleNamespace()
        obs_set.residuals = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = _residuals_array(obs_set)
        np.testing.assert_array_equal(result, [[0.1, 0.2], [0.3, 0.4]])

    def test_1d_residuals_reshaped(self):
        obs_set = SimpleNamespace()
        obs_set.residuals = np.array([0.1, 0.2])  # single obs, 1D
        result = _residuals_array(obs_set)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[0.1, 0.2]])

    def test_empty_1d_residuals(self):
        obs_set = SimpleNamespace()
        obs_set.residuals = np.array([])  # empty, 1D
        result = _residuals_array(obs_set)
        assert result.shape == (0, 2)


class TestBuildWeightsDf:
    def test_basic_dataframe(self):
        n_obs = 3
        times = np.array([0.0, 100.0, 200.0])
        residuals_rad = np.array([[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]])
        groups = GroupList([
            Group(indices=np.array([0, 1, 2], dtype=int), group_id="tf_0000", level="timeframe"),
        ])
        weights_ra = np.array([1e12, 1e12, 1e12])
        weights_dec = np.array([2e11, 2e11, 2e11])

        df = _build_weights_df(
            n_obs, times, residuals_rad, groups,
            weights_ra, weights_dec,
            "id_v2", "obs1", "timeframe",
        )

        assert len(df) == 3
        assert list(df.columns) == [
            "set_id", "strategy", "group_level", "group_id",
            "parent_id", "parent_level", "time",
            "ra_residual_rad", "dec_residual_rad",
            "ra_residual_arcsec", "dec_residual_arcsec",
            "group_ra_rms_arcsec", "group_dec_rms_arcsec",
            "weight_ra", "weight_dec",
        ]
        assert df["set_id"].tolist() == ["obs1"] * 3
        assert df["strategy"].tolist() == ["id_v2"] * 3
        assert df["group_id"].tolist() == ["tf_0000"] * 3

    def test_multiple_groups(self):
        n_obs = 4
        times = np.array([0.0, 100.0, 200.0, 300.0])
        residuals_rad = np.array([
            [1e-6, 2e-6], [2e-6, 3e-6], [3e-6, 4e-6], [4e-6, 5e-6],
        ])
        groups = GroupList([
            Group(indices=np.array([0, 1], dtype=int), group_id="tf_0000", level="timeframe"),
            Group(indices=np.array([2, 3], dtype=int), group_id="tf_0001", level="timeframe"),
        ])
        weights_ra = np.array([1e12, 1e12, 2e12, 2e12])
        weights_dec = np.array([1e11, 1e11, 2e11, 2e11])

        df = _build_weights_df(
            n_obs, times, residuals_rad, groups,
            weights_ra, weights_dec,
            "hybrid_geometric", "obs1", "timeframe",
        )

        assert df.loc[0, "group_id"] == "tf_0000"
        assert df.loc[3, "group_id"] == "tf_0001"
        assert df.loc[0, "weight_ra"] == 1e12
        assert df.loc[3, "weight_ra"] == 2e12

    def test_with_parent_references(self):
        n_obs = 2
        times = np.array([0.0, 100.0])
        residuals_rad = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        groups = GroupList([
            Group(
                indices=np.array([0, 1], dtype=int),
                group_id="tf_0000", level="timeframe",
                parent_level="section", parent_id="early",
            ),
        ])
        weights_ra = np.array([1e12, 1e12])
        weights_dec = np.array([1e11, 1e11])

        df = _build_weights_df(
            n_obs, times, residuals_rad, groups,
            weights_ra, weights_dec,
            "id_v2", "obs1", "timeframe",
        )

        assert df["parent_id"].tolist() == ["early", "early"]
        assert df["parent_level"].tolist() == ["section", "section"]


# ===========================================================================
# Mock helpers for strategy tests
# ===========================================================================


def _make_mock_obs_set(
    residuals: np.ndarray,
    times: list[float],
) -> SimpleNamespace:
    """Create a mock SingleObservationSet with residuals and times."""
    mock_times = [SimpleNamespace(to_float=lambda t=t: t) for t in times]
    return SimpleNamespace(
        residuals=residuals,
        observation_times=mock_times,
    )


def _make_timeframe_groups(
    n_obs: int,
    split_indices: list[int] | None = None,
    set_id: str = "obs1",
) -> GroupList:
    """Build a GroupList with timeframe groups.

    If split_indices is given (e.g. [2]), the first group has indices [0,1]
    and the second has [2,3,...]. Otherwise a single group is returned.
    """
    if split_indices:
        groups = []
        start = 0
        for si in split_indices:
            groups.append(
                Group(
                    indices=np.arange(start, si, dtype=int),
                    group_id=f"{set_id}_tf_{len(groups):04d}",
                    level="timeframe",
                    parent_level="set",
                    parent_id=set_id,
                )
            )
            start = si
        groups.append(
            Group(
                indices=np.arange(start, n_obs, dtype=int),
                group_id=f"{set_id}_tf_{len(groups):04d}",
                level="timeframe",
                parent_level="set",
                parent_id=set_id,
            )
        )
        return GroupList(groups)
    return GroupList([
        Group(
            indices=np.arange(n_obs, dtype=int),
            group_id=f"{set_id}_tf_0000",
            level="timeframe",
            parent_level="set",
            parent_id=set_id,
        ),
    ])


# ===========================================================================
# IDv2Weight strategy
# ===========================================================================


class TestIDv2Weight:
    def test_single_timeframe_basic(self):
        """With one timeframe, weight = 1 / RMS(residuals)^2."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(2)

        strategy = IDv2Weight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        # Expected: RA RMS = sqrt(mean([1e-6^2, 3e-6^2])) = sqrt(5e-12) ≈ 2.236e-6
        # DEC RMS = sqrt(mean([2e-6^2, 4e-6^2])) = sqrt(10e-12) ≈ 3.162e-6
        # Descaled: RA = 2.236e-6 * sqrt(2), DEC = 3.162e-6 * sqrt(2)
        # IDv2 sigma = RMS of descaled (only one value, so = descaled)
        # w_ra = 1 / (2.236e-6 * sqrt(2))^2 = 1 / (5e-12 * 2) = 1e11
        # w_dec = 1 / (3.162e-6 * sqrt(2))^2 = 1 / (10e-12 * 2) = 5e10
        ra_rms = np.sqrt(np.mean(np.square(residuals[:, 0])))
        dec_rms = np.sqrt(np.mean(np.square(residuals[:, 1])))
        scale = np.sqrt(2)
        expected_w_ra = 1.0 / (ra_rms * scale) ** 2
        expected_w_dec = 1.0 / (dec_rms * scale) ** 2

        assert weights_array.shape == (4,)
        np.testing.assert_allclose(weights_array[0], expected_w_ra)
        np.testing.assert_allclose(weights_array[1], expected_w_dec)
        assert len(df) == 2

    def test_multiple_timeframes(self):
        """Two timeframes: descaled RMS of each, then RMS of those."""
        residuals = np.array([
            [1e-6, 2e-6], [3e-6, 4e-6],  # timeframe 0
            [5e-6, 6e-6], [7e-6, 8e-6],  # timeframe 1
        ])
        times = [0.0, 100.0, 100000.0, 100100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(4, split_indices=[2])

        strategy = IDv2Weight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        # TF0: RA RMS = sqrt(mean([1,9])) * 1e-6 = sqrt(5)e-6
        #      DEC RMS = sqrt(mean([4,16])) * 1e-6 = sqrt(10)e-6
        #      descaled RA = sqrt(5)e-6 * sqrt(2), DEC = sqrt(10)e-6 * sqrt(2)
        # TF1: RA RMS = sqrt(mean([25,49])) * 1e-6 = sqrt(37)e-6
        #      DEC RMS = sqrt(mean([36,64])) * 1e-6 = sqrt(50)e-6
        #      descaled RA = sqrt(37)e-6 * sqrt(2), DEC = sqrt(50)e-6 * sqrt(2)
        # IDv2 RA sigma = RMS([sqrt(5)*sqrt(2), sqrt(37)*sqrt(2)]) * 1e-6
        #               = sqrt(2) * RMS([sqrt(5), sqrt(37)]) * 1e-6
        # IDv2 DEC sigma = sqrt(2) * RMS([sqrt(10), sqrt(50)]) * 1e-6

        descaled_ra = np.array([np.sqrt(5.0), np.sqrt(37.0)]) * np.sqrt(2) * 1e-6
        descaled_dec = np.array([np.sqrt(10.0), np.sqrt(50.0)]) * np.sqrt(2) * 1e-6
        id_v2_ra_sigma = np.sqrt(np.mean(np.square(descaled_ra)))
        id_v2_dec_sigma = np.sqrt(np.mean(np.square(descaled_dec)))

        expected_w_ra = 1.0 / id_v2_ra_sigma**2
        expected_w_dec = 1.0 / id_v2_dec_sigma**2

        np.testing.assert_allclose(weights_array[0], expected_w_ra)
        np.testing.assert_allclose(weights_array[1], expected_w_dec)
        # All observations get the same weight (constant per set)
        np.testing.assert_allclose(weights_array[0::2], expected_w_ra)
        np.testing.assert_allclose(weights_array[1::2], expected_w_dec)

    def test_no_timeframe_groups_fallback(self):
        """When no timeframe groups exist, fall back to set-level RMS."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        # Empty GroupList — no timeframe groups
        groups = GroupList()

        strategy = IDv2Weight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        ra_sigma = max(np.sqrt(np.mean(np.square(residuals[:, 0]))), 0.01 * np.pi / (180 * 3600))
        dec_sigma = max(np.sqrt(np.mean(np.square(residuals[:, 1]))), 0.01 * np.pi / (180 * 3600))
        expected_w_ra = 1.0 / ra_sigma**2
        expected_w_dec = 1.0 / dec_sigma**2

        np.testing.assert_allclose(weights_array[0], expected_w_ra)
        np.testing.assert_allclose(weights_array[1], expected_w_dec)
        assert df["group_level"].iloc[0] == "set"

    def test_min_sigma_floor_applied(self):
        """Very small residuals should be floored by min_sigma_arcsec."""
        residuals = np.array([[1e-12, 1e-12]])  # tiny residuals
        times = [0.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(1)

        strategy = IDv2Weight()
        weights_array, df = strategy.compute_weights(
            obs_set, groups, "obs1", min_sigma_arcsec=0.1,
        )

        min_sigma_rad = 0.1 * np.pi / (180 * 3600)
        # The descaled sigma should be floored to min_sigma_rad
        expected_w = 1.0 / min_sigma_rad**2
        np.testing.assert_allclose(weights_array[0], expected_w)

    def test_empty_observation_set(self):
        obs_set = _make_mock_obs_set(np.empty((0, 2)), [])
        groups = GroupList()

        strategy = IDv2Weight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        assert len(weights_array) == 0
        assert df.empty

    def test_dataframe_columns(self):
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(2)

        strategy = IDv2Weight()
        _, df = strategy.compute_weights(obs_set, groups, "obs1")

        expected_columns = {
            "set_id", "strategy", "group_level", "group_id",
            "parent_id", "parent_level", "time",
            "ra_residual_rad", "dec_residual_rad",
            "ra_residual_arcsec", "dec_residual_arcsec",
            "group_ra_rms_arcsec", "group_dec_rms_arcsec",
            "weight_ra", "weight_dec",
        }
        assert set(df.columns) == expected_columns


# ===========================================================================
# HybridGeometricWeight strategy
# ===========================================================================


class TestHybridGeometricWeight:
    def test_single_timeframe(self):
        """With one timeframe, geometric mean of set-level and tf-level weights."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(2)

        strategy = HybridGeometricWeight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        # Set-level RMS
        ra_set_sigma = np.sqrt(np.mean(np.square(residuals[:, 0])))
        dec_set_sigma = np.sqrt(np.mean(np.square(residuals[:, 1])))
        w_ra_set = 1.0 / ra_set_sigma**2
        w_dec_set = 1.0 / dec_set_sigma**2

        # TF-level RMS (same as set-level since only one TF)
        ra_tf_sigma = ra_set_sigma
        dec_tf_sigma = dec_set_sigma
        w_ra_tf = 1.0 / ra_tf_sigma**2
        w_dec_tf = 1.0 / dec_tf_sigma**2

        expected_w_ra = np.sqrt(w_ra_set * w_ra_tf)
        expected_w_dec = np.sqrt(w_dec_set * w_dec_tf)

        np.testing.assert_allclose(weights_array[0], expected_w_ra)
        np.testing.assert_allclose(weights_array[1], expected_w_dec)

    def test_multiple_timeframes_different_weights(self):
        """Each timeframe gets its own geometric mean weight."""
        residuals = np.array([
            [1e-6, 2e-6], [3e-6, 4e-6],  # TF0: larger residuals
            [1e-7, 2e-7], [3e-7, 4e-7],  # TF1: smaller residuals
        ])
        times = [0.0, 100.0, 100000.0, 100100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(4, split_indices=[2])

        strategy = HybridGeometricWeight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        # Set-level RMS (all residuals)
        ra_set_sigma = np.sqrt(np.mean(np.square(residuals[:, 0])))
        dec_set_sigma = np.sqrt(np.mean(np.square(residuals[:, 1])))
        w_ra_set = 1.0 / ra_set_sigma**2
        w_dec_set = 1.0 / dec_set_sigma**2

        # TF0 RMS
        ra_tf0_sigma = np.sqrt(np.mean(np.square(residuals[0:2, 0])))
        dec_tf0_sigma = np.sqrt(np.mean(np.square(residuals[0:2, 1])))
        w_ra_tf0 = 1.0 / ra_tf0_sigma**2
        w_dec_tf0 = 1.0 / dec_tf0_sigma**2

        # TF1 RMS
        ra_tf1_sigma = np.sqrt(np.mean(np.square(residuals[2:4, 0])))
        dec_tf1_sigma = np.sqrt(np.mean(np.square(residuals[2:4, 1])))
        w_ra_tf1 = 1.0 / ra_tf1_sigma**2
        w_dec_tf1 = 1.0 / dec_tf1_sigma**2

        expected_w_ra_tf0 = np.sqrt(w_ra_set * w_ra_tf0)
        expected_w_dec_tf0 = np.sqrt(w_dec_set * w_dec_tf0)
        expected_w_ra_tf1 = np.sqrt(w_ra_set * w_ra_tf1)
        expected_w_dec_tf1 = np.sqrt(w_dec_set * w_dec_tf1)

        # TF0 observations
        np.testing.assert_allclose(weights_array[0], expected_w_ra_tf0)
        np.testing.assert_allclose(weights_array[1], expected_w_dec_tf0)
        np.testing.assert_allclose(weights_array[2], expected_w_ra_tf0)
        np.testing.assert_allclose(weights_array[3], expected_w_dec_tf0)

        # TF1 observations
        np.testing.assert_allclose(weights_array[4], expected_w_ra_tf1)
        np.testing.assert_allclose(weights_array[5], expected_w_dec_tf1)
        np.testing.assert_allclose(weights_array[6], expected_w_ra_tf1)
        np.testing.assert_allclose(weights_array[7], expected_w_dec_tf1)

    def test_no_timeframe_groups_fallback(self):
        """When no timeframe groups exist, use set-level only."""
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = GroupList()

        strategy = HybridGeometricWeight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        ra_sigma = max(np.sqrt(np.mean(np.square(residuals[:, 0]))), 0.01 * np.pi / (180 * 3600))
        dec_sigma = max(np.sqrt(np.mean(np.square(residuals[:, 1]))), 0.01 * np.pi / (180 * 3600))
        expected_w_ra = 1.0 / ra_sigma**2
        expected_w_dec = 1.0 / dec_sigma**2

        np.testing.assert_allclose(weights_array[0], expected_w_ra)
        np.testing.assert_allclose(weights_array[1], expected_w_dec)
        assert df["group_level"].iloc[0] == "set"

    def test_min_sigma_floor_applied(self):
        residuals = np.array([[1e-12, 1e-12]])
        times = [0.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(1)

        strategy = HybridGeometricWeight()
        weights_array, df = strategy.compute_weights(
            obs_set, groups, "obs1", min_sigma_arcsec=0.1,
        )

        min_sigma_rad = 0.1 * np.pi / (180 * 3600)
        expected_w = 1.0 / min_sigma_rad**2
        # Geometric mean of set-level and tf-level (both floored)
        expected_hybrid = np.sqrt(expected_w * expected_w)
        np.testing.assert_allclose(weights_array[0], expected_hybrid)

    def test_empty_observation_set(self):
        obs_set = _make_mock_obs_set(np.empty((0, 2)), [])
        groups = GroupList()

        strategy = HybridGeometricWeight()
        weights_array, df = strategy.compute_weights(obs_set, groups, "obs1")

        assert len(weights_array) == 0
        assert df.empty

    def test_dataframe_columns(self):
        residuals = np.array([[1e-6, 2e-6], [3e-6, 4e-6]])
        times = [0.0, 100.0]
        obs_set = _make_mock_obs_set(residuals, times)
        groups = _make_timeframe_groups(2)

        strategy = HybridGeometricWeight()
        _, df = strategy.compute_weights(obs_set, groups, "obs1")

        expected_columns = {
            "set_id", "strategy", "group_level", "group_id",
            "parent_id", "parent_level", "time",
            "ra_residual_rad", "dec_residual_rad",
            "ra_residual_arcsec", "dec_residual_arcsec",
            "group_ra_rms_arcsec", "group_dec_rms_arcsec",
            "weight_ra", "weight_dec",
        }
        assert set(df.columns) == expected_columns