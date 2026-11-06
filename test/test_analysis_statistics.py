"""Tests for the analysis statistics primitives added for the plot refactor."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdet.analysis import (
    maximum,
    mean,
    median,
    minimum,
    pooled_within_group_rms,
    quadrature,
    rms,
    sample_std,
)


class TestReductionPrimitives:
    def test_mean_ignores_non_finite(self):
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert mean([1.0, np.nan, 3.0]) == pytest.approx(2.0)
        assert mean([]) != mean([])  # NaN

    def test_sample_std_is_ddof_one_by_default(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert sample_std(values) == pytest.approx(np.std(values, ddof=1))
        assert sample_std(values, ddof=0) == pytest.approx(np.std(values, ddof=0))

    def test_sample_std_nan_when_too_few(self):
        assert sample_std([1.0]) != sample_std([1.0])  # NaN (needs 2 for ddof=1)
        assert sample_std([1.0], ddof=0) == pytest.approx(0.0)

    def test_median_min_max(self):
        assert median([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert minimum([3.0, -1.0, 2.0]) == pytest.approx(-1.0)
        assert maximum([3.0, -1.0, 2.0]) == pytest.approx(3.0)
        assert minimum([np.nan]) != minimum([np.nan])  # NaN

    def test_quadrature(self):
        assert quadrature([3.0, 4.0]) == pytest.approx(5.0)
        assert quadrature([1.0, 2.0, 2.0]) == pytest.approx(3.0)
        # Equals rms * sqrt(n).
        values = [1.0, 2.0, 3.0]
        assert quadrature(values) == pytest.approx(rms(values) * np.sqrt(3))


class TestPooledWithinGroupRms:
    def test_removes_group_means_and_uses_n_minus_m_dof(self):
        # Two clusters with different means but identical within-cluster spread.
        values = [1.0, 3.0, 11.0, 13.0]
        labels = [0, 0, 1, 1]
        # Within deviations: [-1, 1, -1, 1] -> sum sq = 4, dof = 4 - 2 = 2.
        assert pooled_within_group_rms(values, labels) == pytest.approx(np.sqrt(4.0 / 2.0))

    def test_matches_manual_for_single_group(self):
        values = [1.0, 2.0, 3.0]
        labels = [0, 0, 0]
        # dof = 3 - 1 = 2, same as sample_std.
        assert pooled_within_group_rms(values, labels) == pytest.approx(sample_std(values))

    def test_nan_when_no_residual_dof(self):
        # One observation per cluster -> N == M -> no dof.
        assert pooled_within_group_rms([1.0, 2.0], [0, 1]) != pooled_within_group_rms(
            [1.0, 2.0], [0, 1]
        )  # NaN

    def test_ignores_non_finite(self):
        values = [1.0, 3.0, np.nan, 11.0, 13.0]
        labels = [0, 0, 0, 1, 1]
        assert pooled_within_group_rms(values, labels) == pytest.approx(np.sqrt(4.0 / 2.0))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="matching lengths"):
            pooled_within_group_rms([1.0, 2.0], [0])
