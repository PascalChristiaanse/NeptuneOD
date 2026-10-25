"""Tests for the fixed along/across-scan weighting strategy.

``FixedScanWeight`` assigns a fixed 1σ uncertainty in the Gaia scan frame.  Since
the scan frame is rotated relative to ``(RA, Dec)``, the resulting weight matrix
is a non-diagonal 2x2 block per observation, applied via
``set_block_diagonal_weights``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from orbitdet.data.gaia import _register_scan_angles
from orbitdet.observations.grouping import Group, GroupList
from orbitdet.observations.weighting.strategies import FixedScanWeight

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


def _groups(set_id, n):
    return GroupList(
        [Group(indices=np.arange(n, dtype=int), group_id=set_id, level="set", layer=-1)]
    )


# ===========================================================================
# Construction / validation
# ===========================================================================


class TestFixedScanWeightConstruction:
    def test_defaults(self):
        s = FixedScanWeight()
        assert s.along_scan_sigma_mas == 5.0
        assert s.across_scan_sigma_mas == 200.0

    def test_custom_values(self):
        s = FixedScanWeight(along_scan_sigma_mas=1.0, across_scan_sigma_mas=10.0)
        assert s.along_scan_sigma_mas == 1.0
        assert s.across_scan_sigma_mas == 10.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_bad_along(self, bad):
        with pytest.raises(ValueError):
            FixedScanWeight(along_scan_sigma_mas=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_bad_across(self, bad):
        with pytest.raises(ValueError):
            FixedScanWeight(across_scan_sigma_mas=bad)


# ===========================================================================
# Block weights
# ===========================================================================


class TestFixedScanBlockWeights:
    def test_block_is_inverse_of_scan_covariance(self):
        # psi = 0, dec = 0 -> T = [[0, 1], [1, 0]], so AL = dDec, AC = dRA.
        # W_radec = T^T diag(w_AL, w_AC) T = diag(w_AC, w_AL).
        epochs = [0.0, 1.0]
        _register_scan_angles(epochs, np.zeros(2))
        obs_set = _make_obs_set(np.zeros((2, 2)), epochs, dec=0.0)

        s = FixedScanWeight(along_scan_sigma_mas=5.0, across_scan_sigma_mas=200.0)
        blocks = s.compute_block_weights(obs_set, _groups("Gaia", 2), "Gaia")

        assert blocks is not None and len(blocks) == 2
        w_al = 1.0 / (5.0 * MAS_TO_RAD) ** 2
        w_ac = 1.0 / (200.0 * MAS_TO_RAD) ** 2
        # RA block entry = w_AC, Dec block entry = w_AL.
        np.testing.assert_allclose(blocks[0], np.diag([w_ac, w_al]), rtol=1e-12)

    def test_block_recovers_scan_covariance(self):
        # W^{-1} must equal T^{-1} diag(sig_AL^2, sig_AC^2) T^{-T}, i.e. the
        # (RA, Dec) covariance whose scan-frame projection is diagonal.
        psi = 0.7
        dec = 0.3
        epochs = [0.0]
        _register_scan_angles(epochs, np.array([psi]))
        obs_set = _make_obs_set(np.zeros((1, 2)), epochs, dec=dec)

        s = FixedScanWeight(along_scan_sigma_mas=5.0, across_scan_sigma_mas=200.0)
        blocks = s.compute_block_weights(obs_set, _groups("Gaia", 1), "Gaia")
        w = blocks[0]

        sigma_al = 5.0 * MAS_TO_RAD
        sigma_ac = 200.0 * MAS_TO_RAD
        t = np.array(
            [
                [np.sin(psi) * np.cos(dec), np.cos(psi)],
                [np.cos(psi) * np.cos(dec), -np.sin(psi)],
            ]
        )
        c_scan = np.diag([sigma_al**2, sigma_ac**2])
        c_radec = np.linalg.inv(t) @ c_scan @ np.linalg.inv(t).T
        np.testing.assert_allclose(np.linalg.inv(w), c_radec, rtol=1e-10)

        # And the scan-frame projection of W^{-1} must be diagonal.
        c_scan_recovered = t @ np.linalg.inv(w) @ t.T
        np.testing.assert_allclose(c_scan_recovered, c_scan, rtol=1e-10, atol=1e-27)

    def test_block_is_symmetric_positive_definite(self):
        psi = 1.2
        dec = -0.4
        epochs = [0.0, 1.0, 2.0]
        _register_scan_angles(epochs, np.full(3, psi))
        obs_set = _make_obs_set(np.zeros((3, 2)), epochs, dec=dec)

        s = FixedScanWeight()
        blocks = s.compute_block_weights(obs_set, _groups("Gaia", 3), "Gaia")
        for w in blocks:
            np.testing.assert_allclose(w, w.T, rtol=1e-12)
            assert np.all(np.linalg.eigvalsh(w) > 0.0)

    def test_returns_none_without_scan_angles(self):
        # Epochs with no registered scan angles -> None (engine falls back to diagonal).
        obs_set = _make_obs_set(np.zeros((2, 2)), [12345.0, 12346.0])
        s = FixedScanWeight()
        assert s.compute_block_weights(obs_set, _groups("Gaia", 2), "Gaia") is None


# ===========================================================================
# Diagonal reporting weights
# ===========================================================================


class TestFixedScanDiagonalWeights:
    def test_diagonal_matches_block_diagonal(self):
        psi = 0.5
        dec = 0.1
        epochs = [0.0, 1.0]
        _register_scan_angles(epochs, np.full(2, psi))
        obs_set = _make_obs_set(np.zeros((2, 2)), epochs, dec=dec)

        s = FixedScanWeight()
        weights_array, df = s.compute_weights(obs_set, _groups("Gaia", 2), "Gaia")
        blocks = s.compute_block_weights(obs_set, _groups("Gaia", 2), "Gaia")

        # Interleaved [RA, Dec] diagonal must match the block diagonals.
        assert weights_array[0::2] == pytest.approx([b[0, 0] for b in blocks])
        assert weights_array[1::2] == pytest.approx([b[1, 1] for b in blocks])
        assert len(df) == 2
        assert (df["strategy"] == "fixed_scan").all()

    def test_diagonal_fallback_without_scan_angles(self):
        obs_set = _make_obs_set(np.zeros((2, 2)), [999.0, 1000.0])
        s = FixedScanWeight(along_scan_sigma_mas=5.0, across_scan_sigma_mas=200.0)
        weights_array, _ = s.compute_weights(obs_set, _groups("Gaia", 2), "Gaia")
        w_al = 1.0 / (5.0 * MAS_TO_RAD) ** 2
        w_ac = 1.0 / (200.0 * MAS_TO_RAD) ** 2
        assert weights_array[0::2] == pytest.approx([w_al, w_al])
        assert weights_array[1::2] == pytest.approx([w_ac, w_ac])

    def test_empty_set(self):
        obs_set = _make_obs_set(np.zeros((0, 2)), [])
        s = FixedScanWeight()
        weights_array, df = s.compute_weights(obs_set, _groups("Gaia", 0), "Gaia")
        assert len(weights_array) == 0
        assert df.empty
