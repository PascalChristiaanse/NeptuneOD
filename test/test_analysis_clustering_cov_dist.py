"""Tests for the analysis clustering, covariance and distribution modules."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdet.analysis import (
    block_sigmas,
    cluster_by_gap,
    condition_index,
    density_histogram,
    find_psd_peaks,
    lombscargle_psd,
    marginal_sigmas,
    normality_test,
    principal_axes,
    qq_points,
    split_by_gap,
    standardize,
    welch_psd,
)


class TestClustering:
    def test_split_by_gap_on_two_nights(self):
        epochs = np.array([0.0, 10.0, 20.0, 1000.0, 1010.0])
        clusters = split_by_gap(epochs, gap_seconds=100.0)
        assert [g.tolist() for g in clusters] == [[0, 1, 2], [3, 4]]

    def test_min_size_absorbs_small_internal_gap(self):
        # A gap splits only if the current cluster already has >= min_size members.
        # First gap sees a 1-member cluster (no split); second gap sees 2 (splits).
        epochs = np.array([0.0, 1000.0, 2000.0])
        clusters = split_by_gap(epochs, gap_seconds=100.0, min_size=2)
        assert [g.tolist() for g in clusters] == [[0, 1], [2]]

    def test_empty(self):
        assert split_by_gap([], 10.0) == []

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            split_by_gap([1.0, 2.0], -1.0)

    def test_cluster_by_gap_labels_unsorted_input(self):
        epochs = np.array([1000.0, 0.0, 1010.0, 10.0])
        labels = cluster_by_gap(epochs, gap_seconds=100.0)
        # Positions: 1000->group1, 0->group0, 1010->group1, 10->group0.
        assert labels.tolist() == [1, 0, 1, 0]

    def test_cluster_single_group(self):
        labels = cluster_by_gap([0.0, 1.0, 2.0], gap_seconds=100.0)
        assert labels.tolist() == [0, 0, 0]

    def test_cluster_empty_and_nonfinite(self):
        assert cluster_by_gap([], 10.0).tolist() == []
        with pytest.raises(ValueError, match="non-finite"):
            cluster_by_gap([0.0, np.nan], 10.0)


class TestCovariance:
    def test_marginal_sigmas_diagonal(self):
        W = np.diag([4.0, 9.0])
        np.testing.assert_allclose(marginal_sigmas(W), [0.5, 1.0 / 3.0])

    def test_marginal_sigmas_full_is_diag_of_inverse(self):
        W = np.array([[2.0, 0.5], [0.5, 1.0]])
        C = np.linalg.inv(W)
        np.testing.assert_allclose(marginal_sigmas(W), np.sqrt(np.diag(C)))

    def test_block_sigmas_matches_marginal_for_block_diagonal(self):
        W = np.zeros((4, 4))
        W[0:2, 0:2] = np.array([[2.0, 0.5], [0.5, 1.0]])
        W[2:4, 2:4] = np.array([[3.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(block_sigmas(W), marginal_sigmas(W))

    def test_block_sigmas_bad_size_raises(self):
        with pytest.raises(ValueError, match="multiple"):
            block_sigmas(np.eye(3), block_size=2)

    def test_principal_axes_aligned_ellipse(self):
        covariance = np.diag([9.0, 1.0])
        semi_axes, angle = principal_axes(covariance)
        np.testing.assert_allclose(semi_axes, [3.0, 1.0])
        assert abs(angle) == pytest.approx(0.0)

    def test_principal_axes_rotated(self):
        # 45-degree ellipse: major axis along y=x.
        covariance = np.array([[2.0, 1.0], [1.0, 2.0]])
        semi_axes, angle = principal_axes(covariance)
        np.testing.assert_allclose(semi_axes, [np.sqrt(3.0), 1.0])
        assert angle == pytest.approx(np.pi / 4.0)

    def test_principal_axes_scales_with_n_sigma(self):
        covariance = np.diag([4.0, 1.0])
        semi_1, _ = principal_axes(covariance, n_sigma=1.0)
        semi_3, _ = principal_axes(covariance, n_sigma=3.0)
        np.testing.assert_allclose(semi_3, 3.0 * semi_1)

    def test_principal_axes_requires_2x2(self):
        with pytest.raises(ValueError, match="2x2"):
            principal_axes(np.eye(3))

    def test_condition_index(self):
        covariance = np.diag([4.0, 1.0])
        np.testing.assert_allclose(condition_index(covariance), [1.0, 2.0])

    def test_condition_index_zero_eigenvalue_is_inf(self):
        index = condition_index(np.diag([4.0, 0.0]))
        assert index[0] == pytest.approx(1.0)
        assert np.isinf(index[1])


class TestDistribution:
    def test_standardize(self):
        z = standardize([1.0, 2.0, 3.0])
        assert np.mean(z) == pytest.approx(0.0)
        assert np.std(z, ddof=1) == pytest.approx(1.0)

    def test_qq_points_normal_sample_lies_on_line(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(size=500)
        theoretical, ordered = qq_points(sample)
        assert theoretical.size == ordered.size == 500
        # Slope of ordered vs theoretical is ~1 for a normal sample.
        slope = np.polyfit(theoretical, ordered, 1)[0]
        assert slope == pytest.approx(1.0, abs=0.15)

    def test_qq_points_too_few(self):
        assert qq_points([1.0])[0].size == 0

    def test_density_histogram_integrates_to_one(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=1000)
        density, edges, error, centers = density_histogram(values, bins=20)
        assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)
        assert density.size == error.size == centers.size == 20

    def test_density_histogram_poisson_error(self):
        density, edges, error, _ = density_histogram(np.ones(10), bins=1)
        # Single bin: count 10, error = sqrt(10) / (10 * width).
        width = np.diff(edges)[0]
        assert error[0] == pytest.approx(np.sqrt(10.0) / (10.0 * width))

    def test_density_histogram_empty(self):
        density, _, _, _ = density_histogram([])
        assert density.size == 0

    def test_normality_test_normal_vs_uniform(self):
        rng = np.random.default_rng(2)
        _, p_normal = normality_test(rng.normal(size=200))
        _, p_uniform = normality_test(rng.uniform(size=200))
        assert p_normal > 0.05
        assert p_uniform < 0.05

    def test_normality_test_too_few(self):
        w, p = normality_test([1.0, 2.0])
        assert np.isnan(w) and np.isnan(p)

    def test_welch_psd_recovers_sinusoid_frequency(self):
        # 8 days of hourly samples at a 2 cycles/day sinusoid.
        days = np.arange(0.0, 8.0, 1.0 / 24.0)
        values = np.sin(2 * np.pi * 2.0 * days)
        frequencies, power = welch_psd(days, values, window_length=4.0)
        peak = frequencies[np.argmax(power)]
        assert peak == pytest.approx(2.0, abs=0.5)

    def test_lombscargle_psd_recovers_frequency(self):
        rng = np.random.default_rng(3)
        days = np.sort(rng.uniform(0.0, 100.0, size=300))
        values = np.sin(2 * np.pi * 0.2 * days)
        frequencies, power = lombscargle_psd(days, values)
        peak = frequencies[np.argmax(power)]
        assert peak == pytest.approx(0.2, abs=0.05)

    def test_find_psd_peaks_sorted_by_power(self):
        frequencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        power = np.array([0.0, 0.5, 0.0, 1.0, 0.0])
        peaks = find_psd_peaks(frequencies, power, prominence=0.01)
        assert peaks[0][0] == pytest.approx(0.4)
        assert peaks[0][1] == pytest.approx(1.0)
        # A higher-power peak would sort first (kept interior: find_peaks skips edges).
        power2 = np.array([0.0, 0.5, 2.0, 1.0, 0.0])
        peaks2 = find_psd_peaks(frequencies, power2, prominence=0.01)
        assert peaks2[0][0] == pytest.approx(0.3)
        assert peaks2[0][1] == pytest.approx(2.0)

    def test_psd_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            welch_psd([1.0], [1.0], window_length=1.0)
        with pytest.raises(ValueError, match="at least two"):
            lombscargle_psd([1.0], [1.0])
