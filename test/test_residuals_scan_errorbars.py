"""Tests for the Gaia archive error bars on the scan-residual figure.

Covers the scan-frame error propagation helper, the Gaia error registry, and the
per-transit systematic band drawing.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from orbitdet.data.gaia import (  # noqa: E402
    GAIA_ERROR_COLUMNS,
    _register_gaia_errors,
    get_gaia_errors_for_epochs,
)
from orbitdet.visualization.residuals import (  # noqa: E402
    _draw_systematic_bands,
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
# _draw_systematic_bands
# ===========================================================================


class TestSystematicBands:
    def _times(self, n):
        import pandas as pd

        return pd.to_datetime(np.arange(n), unit="s", origin=pd.Timestamp("2000-01-01"))

    def test_one_band_per_constant_run(self):
        fig, ax = plt.subplots()
        times = self._times(6)
        # Two transits: first 3 share sigma=1, last 3 share sigma=2.
        sigma = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        _draw_systematic_bands(ax, times, sigma, "0.5", 0.1)
        # fill_between adds a PolyCollection per run.
        assert len(ax.collections) == 2
        plt.close(fig)

    def test_zero_sigma_skipped(self):
        fig, ax = plt.subplots()
        times = self._times(3)
        _draw_systematic_bands(ax, times, np.zeros(3), "0.5", 0.1)
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_empty_input(self):
        fig, ax = plt.subplots()
        _draw_systematic_bands(ax, self._times(0), np.array([]), "0.5", 0.1)
        assert len(ax.collections) == 0
        plt.close(fig)


# ===========================================================================
# ResidualsScan integration (error bars drawn)
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


class TestResidualsScanErrorBars:
    def test_error_bars_and_bands_drawn(self):
        from omegaconf import OmegaConf

        from orbitdet.data.gaia import _register_scan_angles
        from orbitdet.visualization.residuals import ResidualsScan

        n = 4
        epochs = [0.0, 1.0, 2.0, 3.0]
        residuals = np.zeros((n, 2))
        obs_set = _make_scan_set(n, epochs, residuals)

        _register_scan_angles(epochs, np.zeros(n))
        errors = np.zeros((n, 6))
        errors[:, 0] = 1.0 * ARCSEC  # ra random
        errors[:, 1] = 1.0 * ARCSEC  # dec random
        errors[:, 3] = 0.5 * ARCSEC  # ra systematic
        errors[:, 4] = 0.5 * ARCSEC  # dec systematic
        _register_gaia_errors(epochs, errors)

        collection = SimpleNamespace(get_single_observation_sets=lambda *a, **k: [obs_set])
        cfg = OmegaConf.create(
            {
                "figures": {
                    "scan_test": {
                        "error_bars": {"enabled": True},
                    }
                }
            }
        )
        plot = ResidualsScan(cfg, collection, use_figure_name="scan_test")
        fig, axs = plot._make_figure()

        # Error bars are LineCollections; systematic bands are PolyCollections.
        assert len(axs[0].collections) >= 1
        assert len(axs[1].collections) >= 1
        plt.close(fig)

    def test_disabled_by_default(self):
        from omegaconf import OmegaConf

        from orbitdet.data.gaia import _register_scan_angles
        from orbitdet.visualization.residuals import ResidualsScan

        n = 3
        epochs = [0.0, 1.0, 2.0]
        obs_set = _make_scan_set(n, epochs, np.zeros((n, 2)))
        _register_scan_angles(epochs, np.zeros(n))

        collection = SimpleNamespace(get_single_observation_sets=lambda *a, **k: [obs_set])
        cfg = OmegaConf.create({"figures": {"scan_test": {}}})
        plot = ResidualsScan(cfg, collection, use_figure_name="scan_test")
        fig, axs = plot._make_figure()
        # Only the scatter collections should be present (one per panel).
        assert len(axs[0].collections) == 1
        plt.close(fig)
