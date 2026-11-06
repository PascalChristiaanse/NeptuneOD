"""Tests for fixed scan-frame weighting with random (block-diag) + systematic (rank-1) parts.

``FixedScanComponentsWeight`` builds a full per-transit ``(RA, Dec)`` weight
matrix: the random component is block diagonal (one 2x2 per observation) and the
systematic component is a Kronecker product ``ones(n, n) ⊗ C_sys`` that couples
every observation in a transit.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from orbitdet.observations.clustering import Cluster, ClusterList
from orbitdet.observations.weighting.strategies import FixedScanComponentsWeight

ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)
MAS_TO_RAD = ARCSEC_TO_RAD / 1e3


def _make_obs_set(residuals, times, dec=0.0):
    mock_times = [SimpleNamespace(to_float=lambda t=t: t) for t in times]
    observed = np.tile([0.0, dec], (len(times), 1))
    return SimpleNamespace(
        residuals=np.asarray(residuals, dtype=float),
        observation_times=mock_times,
        observable_type=1,
        concatenated_observations=observed.ravel(),
    )


def _view(scan_angles):
    """A stand-in for SetView exposing only what the strategy reads."""
    return SimpleNamespace(scan_angles=np.asarray(scan_angles, dtype=float))


def _clusters(set_id, n):
    return ClusterList(
        [Cluster(indices=np.arange(n, dtype=int), cluster_id=set_id, level="set", layer=-1)]
    )


def _rotation(psi, dec):
    return np.array(
        [
            [np.sin(psi) * np.cos(dec), np.cos(psi)],
            [np.cos(psi) * np.cos(dec), -np.sin(psi)],
        ]
    )


class TestConstruction:
    def test_defaults_and_raw_components(self):
        s = FixedScanComponentsWeight(
            random_along_scan_mas=3.0,
            random_across_scan_mas=4.0,
            systematic_along_scan_mas=5.0,
            systematic_across_scan_mas=6.0,
        )
        assert s.random_along_scan_mas == 3.0
        assert s.systematic_across_scan_mas == 6.0
        # Reporting sigma is the quadrature sum (parent's attribute).
        assert s.along_scan_sigma_mas == pytest.approx(np.hypot(3.0, 5.0))
        assert s.across_scan_sigma_mas == pytest.approx(np.hypot(4.0, 6.0))

    def test_measured_values(self):
        s = FixedScanComponentsWeight(
            random_along_scan_mas=0.382,
            random_across_scan_mas=114.0,
            systematic_along_scan_mas=8.37,
            systematic_across_scan_mas=103.0,
        )
        assert s.transit_gap_seconds == pytest.approx(1800.0)
        assert s.preserve_full_weight_matrix is True

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            FixedScanComponentsWeight(random_along_scan_mas=-1.0)

    def test_rejects_zero_random(self):
        with pytest.raises(ValueError):
            FixedScanComponentsWeight(random_along_scan_mas=0.0, random_across_scan_mas=1.0)
        with pytest.raises(ValueError):
            FixedScanComponentsWeight(random_along_scan_mas=1.0, random_across_scan_mas=0.0)

    def test_allows_zero_systematic(self):
        s = FixedScanComponentsWeight(
            random_along_scan_mas=5.0,
            random_across_scan_mas=200.0,
            systematic_along_scan_mas=0.0,
            systematic_across_scan_mas=0.0,
        )
        assert s.systematic_along_scan_mas == 0.0


class TestWeightMatrix:
    def test_single_observation_uses_quadrature(self):
        # One observation per transit -> systematic Kronecker block is the 2x2
        # itself, so total covariance = quadrature-combined scan covariance.
        psi, dec = 0.6, 0.2
        epochs = [0.0]
        view = _view(np.array([psi]))
        obs_set = _make_obs_set(np.zeros((1, 2)), epochs, dec=dec)

        s = FixedScanComponentsWeight(
            random_along_scan_mas=1.0,
            random_across_scan_mas=2.0,
            systematic_along_scan_mas=3.0,
            systematic_across_scan_mas=4.0,
        )
        w = s._full_weight_matrix(obs_set, 1, view=view)
        t = _rotation(psi, dec)
        c_scan = np.diag(
            [
                (np.hypot(1.0, 3.0) * MAS_TO_RAD) ** 2,
                (np.hypot(2.0, 4.0) * MAS_TO_RAD) ** 2,
            ]
        )
        expected = np.linalg.inv(t.T @ c_scan @ t)
        np.testing.assert_allclose(w, expected, rtol=1e-10)

    def test_systematic_is_rank_1_kronecker_within_transit(self):
        # Two observations in one transit, psi=0, dec=0.  Pure systematic (tiny
        # random for invertibility) should reproduce ones(2,2) ⊗ C_sys.
        psi, dec = 0.0, 0.0
        epochs = [0.0, 10.0]
        view = _view(np.array([psi, psi]))
        obs_set = _make_obs_set(np.zeros((2, 2)), epochs, dec=dec)

        s = FixedScanComponentsWeight(
            random_along_scan_mas=1e-3,
            random_across_scan_mas=1e-3,
            systematic_along_scan_mas=10.0,
            systematic_across_scan_mas=20.0,
        )
        w = s._full_weight_matrix(obs_set, 2, view=view)
        cov = np.linalg.inv(w)

        t = _rotation(psi, dec)
        c_sys_scan = np.diag([(10.0 * MAS_TO_RAD) ** 2, (20.0 * MAS_TO_RAD) ** 2])
        c_sys_radec = t.T @ c_sys_scan @ t
        expected_sys = np.kron(np.ones((2, 2)), c_sys_radec)
        # The systematic part dominates the tiny random part.
        np.testing.assert_allclose(cov, expected_sys, rtol=1e-3)

    def test_offdiagonal_couples_observations(self):
        # A systematic component must create non-zero off-diagonal (RA,RA) terms
        # between the two observations of a transit.
        epochs = [0.0, 10.0]
        view = _view(np.array([0.5, 0.5]))
        obs_set = _make_obs_set(np.zeros((2, 2)), epochs, dec=0.1)

        s = FixedScanComponentsWeight(
            random_along_scan_mas=1e-3,
            random_across_scan_mas=1e-3,
            systematic_along_scan_mas=10.0,
            systematic_across_scan_mas=20.0,
        )
        w = s._full_weight_matrix(obs_set, 2, view=view)
        cov = np.linalg.inv(w)
        # Observation 0 RA vs observation 1 RA (indices 0 and 2).
        assert abs(cov[0, 2]) > 0.0

    def test_two_transits_have_no_cross_coupling(self):
        # Observations in different transits must not be correlated.
        epochs = [0.0, 10.0, 4000.0, 4010.0]
        view = _view(np.zeros(4))
        obs_set = _make_obs_set(np.zeros((4, 2)), epochs, dec=0.0)

        s = FixedScanComponentsWeight(
            random_along_scan_mas=1.0,
            random_across_scan_mas=3.0,
            systematic_along_scan_mas=10.0,
            systematic_across_scan_mas=20.0,
        )
        w = s._full_weight_matrix(obs_set, 4, view=view)
        cov = np.linalg.inv(w)
        # Cross-transit blocks (obs 0,1 vs obs 2,3) are zero.
        np.testing.assert_allclose(cov[0:4, 4:8], 0.0, atol=1e-30)
        np.testing.assert_allclose(cov[4:8, 0:4], 0.0, atol=1e-30)

    def test_matrix_is_symmetric(self):
        epochs = [0.0, 10.0, 4000.0]
        view = _view(np.array([0.1, 0.9, 1.7]))
        obs_set = _make_obs_set(np.zeros((3, 2)), epochs, dec=0.3)
        s = FixedScanComponentsWeight(
            random_along_scan_mas=1.0,
            random_across_scan_mas=2.0,
            systematic_along_scan_mas=3.0,
            systematic_across_scan_mas=4.0,
        )
        w = s._full_weight_matrix(obs_set, 3, view=view)
        np.testing.assert_allclose(w, w.T, rtol=1e-12)

    def test_none_without_scan_angles(self):
        epochs = [777000.0, 777010.0]
        # No view -> no scan angles available.
        obs_set = _make_obs_set(np.zeros((2, 2)), epochs, dec=0.0)
        s = FixedScanComponentsWeight(
            random_along_scan_mas=1.0,
            random_across_scan_mas=2.0,
            systematic_along_scan_mas=3.0,
            systematic_across_scan_mas=4.0,
        )
        assert s._full_weight_matrix(obs_set, 2) is None

    def test_fallback_without_scan_angles_uses_diagonal(self):
        epochs = [888000.0, 888010.0]
        obs_set = _make_obs_set(np.zeros((2, 2)), epochs, dec=0.0)
        s = FixedScanComponentsWeight(
            random_along_scan_mas=1.0,
            random_across_scan_mas=2.0,
            systematic_along_scan_mas=3.0,
            systematic_across_scan_mas=4.0,
        )
        weights, df = s.compute_weights(obs_set, _clusters("Gaia", 2), "Gaia")
        assert weights.shape == (4,)
        # Falls back to the parent's diagonal (RA, Dec) sigma = quadrature sum.
        np.testing.assert_allclose(weights[0::2], 1.0 / (np.hypot(1.0, 3.0) * MAS_TO_RAD) ** 2)
