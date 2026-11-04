"""Tests for the normalized scan-residual figure and its shared helpers.

Covers the scan-frame error propagation helper, the Gaia error registry, the
scan-frame residual rotation, and the ``NormalizedResidualScan`` figure.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from orbitdet.data.gaia import (  # noqa: E402
    GAIA_ERROR_COLUMNS,
    _register_gaia_errors,
    get_gaia_errors_for_epochs,
)
from orbitdet.visualization.residuals import (  # noqa: E402
    _scan_frame_sigmas,
)

ARCSEC = np.deg2rad(1.0 / 3600.0)


# ===========================================================================
# _scan_frame_sigmas
# ===========================================================================


class TestScanFrameSigmas:
    def test_matches_direct_rotation(self):
        # Compare against an explicit T C T^T computation.
        rng = np.random.default_rng(0)
        n = 20
        ra_err = rng.uniform(0.1, 1.0, n) * ARCSEC
        dec_err = rng.uniform(0.1, 1.0, n) * ARCSEC
        corr = rng.uniform(-0.9, 0.9, n)
        psi = rng.uniform(0, 2 * np.pi, n)
        dec = rng.uniform(-np.pi / 2, np.pi / 2, n)

        sigma_al, sigma_ac = _scan_frame_sigmas(ra_err, dec_err, corr, psi, dec)

        for i in range(n):
            c = np.array(
                [
                    [ra_err[i] ** 2, corr[i] * ra_err[i] * dec_err[i]],
                    [corr[i] * ra_err[i] * dec_err[i], dec_err[i] ** 2],
                ]
            )
            t = np.array(
                [
                    [np.sin(psi[i]) * np.cos(dec[i]), np.cos(psi[i])],
                    [np.cos(psi[i]) * np.cos(dec[i]), -np.sin(psi[i])],
                ]
            )
            c_scan = t @ c @ t.T
            assert sigma_al[i] == pytest.approx(np.sqrt(c_scan[0, 0]), rel=1e-12)
            assert sigma_ac[i] == pytest.approx(np.sqrt(c_scan[1, 1]), rel=1e-12)

    def test_isotropic_uncorrelated_is_rotation_invariant(self):
        # Equal RA/Dec sigma with zero correlation -> sigma is the same in any frame.
        sigma = 0.5 * ARCSEC
        ra_err = np.full(5, sigma)
        dec_err = np.full(5, sigma)
        corr = np.zeros(5)
        psi = np.linspace(0, np.pi, 5)
        dec = np.zeros(5)
        sigma_al, sigma_ac = _scan_frame_sigmas(ra_err, dec_err, corr, psi, dec)
        np.testing.assert_allclose(sigma_al, sigma, rtol=1e-12)
        np.testing.assert_allclose(sigma_ac, sigma, rtol=1e-12)

    def test_correlation_survives_rotation(self):
        # A perfectly correlated (rank-1) covariance must NOT collapse to the
        # naive variance-only propagation.
        ra_err = np.array([1.0]) * ARCSEC
        dec_err = np.array([1.0]) * ARCSEC
        corr = np.array([1.0])
        psi = np.array([np.pi / 4])
        dec = np.array([0.0])
        sigma_al, sigma_ac = _scan_frame_sigmas(ra_err, dec_err, corr, psi, dec)
        # AL is along the correlation direction -> sigma = sqrt(2)*sigma.
        assert sigma_al[0] == pytest.approx(np.sqrt(2.0) * ARCSEC, rel=1e-12)
        # AC is perpendicular -> zero.
        assert sigma_ac[0] == pytest.approx(0.0, abs=1e-18)


# ===========================================================================
# Gaia error registry
# ===========================================================================


class TestGaiaErrorRegistry:
    def test_round_trip_exact_match(self):
        epochs = [1.0, 2.0, 3.0]
        errors = np.arange(18, dtype=float).reshape(3, 6)
        _register_gaia_errors(epochs, errors)
        result = get_gaia_errors_for_epochs(epochs)
        np.testing.assert_allclose(result, errors)

    def test_per_epoch_fallback(self):
        _register_gaia_errors([10.0, 20.0], np.ones((2, 6)))
        # Request a subset in a different order -> per-epoch fallback.
        result = get_gaia_errors_for_epochs([20.0, 10.0])
        assert result is not None
        np.testing.assert_allclose(result[0], np.ones(6))
        np.testing.assert_allclose(result[1], np.ones(6))

    def test_missing_epoch_returns_none(self):
        _register_gaia_errors([1.0], np.ones((1, 6)))
        assert get_gaia_errors_for_epochs([999.0]) is None

    def test_column_order(self):
        assert GAIA_ERROR_COLUMNS == (
            "ra_error_random",
            "dec_error_random",
            "ra_dec_correlation_random",
            "ra_error_systematic",
            "dec_error_systematic",
            "ra_dec_correlation_systematic",
        )


# ===========================================================================
# _scan_frame_components
# ===========================================================================


class TestScanFrameComponents:
    def test_matches_manual_rotation(self):
        from orbitdet.visualization.residuals import _scan_frame_components

        rng = np.random.default_rng(1)
        n = 10
        residuals = rng.normal(0, 1e-6, (n, 2))
        dec = rng.uniform(-1.0, 1.0, n)
        psi = rng.uniform(0, 2 * np.pi, n)

        al, ac = _scan_frame_components(residuals, dec, psi)

        east = residuals[:, 0] * np.cos(dec)
        north = residuals[:, 1]
        np.testing.assert_allclose(al, east * np.sin(psi) + north * np.cos(psi))
        np.testing.assert_allclose(ac, east * np.cos(psi) - north * np.sin(psi))

    def test_round_trip_with_sigmas(self):
        # A residual equal to its own sigma must normalize to 1.
        from orbitdet.visualization.residuals import (
            _scan_frame_components,
            _scan_frame_sigmas,
        )

        psi = np.array([0.3])
        dec = np.array([0.2])
        ra_err = np.array([1.0e-6])
        dec_err = np.array([2.0e-6])
        corr = np.array([0.4])

        sigma_al, sigma_ac = _scan_frame_sigmas(ra_err, dec_err, corr, psi, dec)
        # Build a residual that is exactly (sigma_al, sigma_ac) in the scan frame,
        # then rotate back to (RA, Dec) and confirm the components recover it.
        east = sigma_al * np.sin(psi) + sigma_ac * np.cos(psi)
        north = sigma_al * np.cos(psi) - sigma_ac * np.sin(psi)
        residuals = np.array([[east[0] / np.cos(dec[0]), north[0]]])
        al, ac = _scan_frame_components(residuals, dec, psi)
        np.testing.assert_allclose(al, sigma_al, rtol=1e-12)
        np.testing.assert_allclose(ac, sigma_ac, rtol=1e-12)


# ===========================================================================
# NormalizedResidualScan
# ===========================================================================


def _make_scan_set(n, epochs, residuals, dec=0.0):
    from tudatpy.estimation.observable_models_setup import links

    times = [SimpleNamespace(to_float=lambda t=float(e): t) for e in epochs]
    observed = np.tile([0.0, dec], (n, 1))
    link_ends = {
        links.receiver: SimpleNamespace(reference_point="", body_name="Gaia"),
        links.transmitter: SimpleNamespace(body_name="Triton"),
    }
    return SimpleNamespace(
        observation_times=times,
        residuals=np.asarray(residuals, dtype=float),
        concatenated_observations=observed.ravel(),
        link_definition=SimpleNamespace(link_ends=link_ends),
    )


# ===========================================================================


class TestAppliedScanSigmas:
    def test_diagonal_weights_recover_sigmas(self):
        from orbitdet.visualization.residuals import _applied_scan_sigmas

        # psi=0, dec=0 -> AL = dDec, AC = dRA.  Diagonal weights -> sigmas are
        # 1/sqrt(w) in the corresponding direction.
        sigma_ra = 3.0 * ARCSEC
        sigma_dec = 7.0 * ARCSEC
        w = np.diag([1.0 / sigma_ra**2, 1.0 / sigma_dec**2])
        obs_set = SimpleNamespace(weight_matrix=w)
        psi = np.array([0.0])
        dec = np.array([0.0])
        sigma_al, sigma_ac = _applied_scan_sigmas(obs_set, psi, dec)
        # AL = dDec -> sigma_dec; AC = dRA -> sigma_ra.
        assert sigma_al[0] == pytest.approx(sigma_dec, rel=1e-9)
        assert sigma_ac[0] == pytest.approx(sigma_ra, rel=1e-9)

    def test_block_weights_recover_scan_sigmas(self):
        from orbitdet.visualization.residuals import _applied_scan_sigmas

        # Build a (RA, Dec) weight block from fixed scan sigmas, then confirm the
        # helper recovers those scan sigmas.
        psi = 0.6
        dec = 0.25
        sigma_al = 5.0 * ARCSEC / 1e3
        sigma_ac = 200.0 * ARCSEC / 1e3
        t = np.array(
            [
                [np.sin(psi) * np.cos(dec), np.cos(psi)],
                [np.cos(psi) * np.cos(dec), -np.sin(psi)],
            ]
        )
        w = t.T @ np.diag([1.0 / sigma_al**2, 1.0 / sigma_ac**2]) @ t
        obs_set = SimpleNamespace(weight_matrix=w)
        recovered_al, recovered_ac = _applied_scan_sigmas(obs_set, np.array([psi]), np.array([dec]))
        assert recovered_al[0] == pytest.approx(sigma_al, rel=1e-9)
        assert recovered_ac[0] == pytest.approx(sigma_ac, rel=1e-9)


# ===========================================================================
# _applied_radec_sigmas / _archive_radec_sigmas
# ===========================================================================


class TestAppliedRaDecSigmas:
    def test_diagonal_weights_recover_sigmas(self):
        from orbitdet.visualization.residuals import _applied_radec_sigmas

        sigma_ra = 3.0 * ARCSEC
        sigma_dec = 7.0 * ARCSEC
        w = np.diag([1.0 / sigma_ra**2, 1.0 / sigma_dec**2])
        obs_set = SimpleNamespace(weight_matrix=w)
        recovered_ra, recovered_dec = _applied_radec_sigmas(obs_set, 1)
        assert recovered_ra[0] == pytest.approx(sigma_ra, rel=1e-9)
        assert recovered_dec[0] == pytest.approx(sigma_dec, rel=1e-9)

    def test_wrong_shape_raises(self):
        from orbitdet.visualization.residuals import _applied_radec_sigmas

        obs_set = SimpleNamespace(weight_matrix=np.zeros((0, 0)))
        with pytest.raises(ValueError):
            _applied_radec_sigmas(obs_set, 1)


class TestArchiveRaDecSigmas:
    def test_combines_random_and_systematic_in_quadrature(self):
        from orbitdet.visualization.residuals import _archive_radec_sigmas

        errors = np.zeros((1, 6))
        errors[0, 0] = 3.0 * ARCSEC  # ra random
        errors[0, 1] = 4.0 * ARCSEC  # dec random
        errors[0, 3] = 4.0 * ARCSEC  # ra systematic
        errors[0, 4] = 3.0 * ARCSEC  # dec systematic
        sigma_ra, sigma_dec = _archive_radec_sigmas(errors)
        assert sigma_ra[0] == pytest.approx(5.0 * ARCSEC, rel=1e-9)
        assert sigma_dec[0] == pytest.approx(5.0 * ARCSEC, rel=1e-9)


# ===========================================================================
# NormalizedResiduals
