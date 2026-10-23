"""Tests for the per-transit uncertainty-ellipse residual scatter overlay.

The overlay reduces each Gaia transit's ``2N x 2N`` archive covariance to the
``2x2`` covariance of the transit *mean* and draws the corresponding ``n_sigma``
ellipse.  These tests exercise the pure-numpy helpers directly and the overlay
against a synthetic block-diagonal weight matrix with a known covariance.
"""

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from matplotlib.patches import Ellipse

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from orbitdet.visualization.residual_scatter import (  # noqa: E402
    ResidualScatterWithEllipses,
    _ellipse_parameters,
    _invert_weight_block,
    _transit_mean_covariance,
    _transit_slices,
)

ARCSEC = np.deg2rad(1.0 / 3600.0)


# ===========================================================================
# _transit_slices
# ===========================================================================


class TestTransitSlices:
    def test_single_transit(self):
        # 3 observations, all coupled -> one transit spanning [0, 3).
        w = np.ones((6, 6))
        assert _transit_slices(w) == [(0, 3)]

    def test_two_transits(self):
        # Two 2-observation transits: coupling only within each 4x4 block.
        w = np.zeros((8, 8))
        w[0:4, 0:4] = 1.0
        w[4:8, 4:8] = 1.0
        assert _transit_slices(w) == [(0, 2), (2, 4)]

    def test_fully_diagonal_is_all_singletons(self):
        w = np.eye(6)
        assert _transit_slices(w) == [(0, 1), (1, 2), (2, 3)]

    def test_uneven_transits(self):
        w = np.zeros((10, 10))
        w[0:2, 0:2] = 1.0  # 1 obs
        w[2:8, 2:8] = 1.0  # 3 obs
        w[8:10, 8:10] = 1.0  # 1 obs
        assert _transit_slices(w) == [(0, 1), (1, 4), (4, 5)]


# ===========================================================================
# _transit_mean_covariance
# ===========================================================================


class TestTransitMeanCovariance:
    def test_single_observation_is_identity_reduction(self):
        cov = np.array([[4.0, 1.0], [1.0, 9.0]])
        np.testing.assert_allclose(_transit_mean_covariance(cov), cov)

    def test_independent_observations_average_down(self):
        # N independent observations each with covariance C -> mean cov = C / N.
        c = np.array([[4.0, 0.0], [0.0, 9.0]])
        n = 4
        cov = np.kron(np.eye(n), c)
        np.testing.assert_allclose(_transit_mean_covariance(cov), c / n)

    def test_fully_correlated_does_not_average_down(self):
        # Perfectly correlated samples: mean covariance equals the common block.
        c = np.array([[4.0, 1.0], [1.0, 9.0]])
        n = 3
        cov = np.kron(np.ones((n, n)), c)
        np.testing.assert_allclose(_transit_mean_covariance(cov), c)

    def test_empty(self):
        np.testing.assert_allclose(_transit_mean_covariance(np.zeros((0, 0))), np.zeros((2, 2)))


# ===========================================================================
# _invert_weight_block
# ===========================================================================


class TestInvertWeightBlock:
    def test_inverts_well_conditioned(self):
        w = np.array([[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(_invert_weight_block(w), np.diag([0.5, 0.25]))

    def test_singular_falls_back_to_pinv(self):
        w = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        result = _invert_weight_block(w)
        assert np.all(np.isfinite(result))
        # Moore-Penrose inverse of a rank-1 all-ones 2x2 is [[.25,.25],[.25,.25]].
        np.testing.assert_allclose(result, np.full((2, 2), 0.25))


# ===========================================================================
# _ellipse_parameters
# ===========================================================================


class TestEllipseParameters:
    def test_axis_aligned(self):
        cov = np.diag([9.0, 4.0])
        major, minor, angle = _ellipse_parameters(cov, n_sigma=1.0)
        assert major == pytest.approx(3.0)
        assert minor == pytest.approx(2.0)
        assert angle == pytest.approx(0.0)

    def test_n_sigma_scales_axes(self):
        cov = np.diag([9.0, 4.0])
        major, minor, _ = _ellipse_parameters(cov, n_sigma=2.0)
        assert major == pytest.approx(6.0)
        assert minor == pytest.approx(4.0)

    def test_rotated(self):
        # 45-degree rotation of a diagonal covariance.
        theta = np.deg2rad(45.0)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        cov = rot @ np.diag([9.0, 1.0]) @ rot.T
        major, minor, angle = _ellipse_parameters(cov, n_sigma=1.0)
        assert major == pytest.approx(3.0)
        assert minor == pytest.approx(1.0)
        assert abs(angle) == pytest.approx(45.0)


# ===========================================================================
# ResidualScatterWithEllipses overlay
# ===========================================================================


def _make_gaia_set(residuals, weight_matrix, has_full=True):
    """Build a minimal observation-set stand-in for the overlay."""
    n = len(residuals)
    times = [SimpleNamespace(to_float=lambda t=float(i): t) for i in range(n)]
    return SimpleNamespace(
        residuals=np.asarray(residuals, dtype=float),
        observation_times=times,
        weight_matrix=np.asarray(weight_matrix, dtype=float),
        has_full_weight_matrix_contribution=has_full,
    )


def _transit_weight(c, n, eps_rel=1e-6):
    """Weight matrix for one transit of ``n`` correlated observations.

    The transit covariance is ``C = 1_n 1_n^T (x) c + eps * I`` (a common
    ``2x2`` block ``c`` shared by every crossing plus a small diagonal to keep it
    positive definite), so the mean covariance is ``c + eps / n * I``.
    """
    c = np.asarray(c, dtype=float)
    eps = eps_rel * float(np.mean(np.diag(c)))
    cov = np.kron(np.ones((n, n)), c) + eps * np.eye(2 * n)
    return np.linalg.inv(cov)


def _block_diag(*blocks):
    from scipy.linalg import block_diag

    return block_diag(*blocks)


def _make_plot(obs_sets, cfg=None):
    from omegaconf import OmegaConf

    collection = SimpleNamespace(get_single_observation_sets=lambda *a, **k: obs_sets)
    plot = ResidualScatterWithEllipses(OmegaConf.create(cfg or {}), collection)
    return plot


class TestConfigResolution:
    """The figure config composes under ``cfg.figures.<name>``; the plot must find it."""

    def test_reads_config_from_figures_namespace(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"figures": {"residual_scatter_ellipses": {"ellipses": {"max_ellipses": 5}}}}
        )
        plot = _make_plot([], cfg)
        resolved = plot._resolve_plot_cfg(cfg)
        assert resolved is not None
        assert resolved.ellipses.max_ellipses == 5

    def test_falls_back_to_top_level(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"residual_scatter_ellipses": {"ellipses": {"max_ellipses": 7}}})
        plot = _make_plot([], cfg)
        resolved = plot._resolve_plot_cfg(cfg)
        assert resolved is not None
        assert resolved.ellipses.max_ellipses == 7

    def test_max_ellipses_from_figures_namespace_is_honoured(self):
        # Regression: the cap was ignored because the config was read from the
        # wrong namespace, so every transit got an ellipse.
        from omegaconf import OmegaConf

        c = np.diag([(1.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        n_transits = 6
        w = _block_diag(*[_transit_weight(c, 2) for _ in range(n_transits)])
        residuals = np.repeat(np.arange(n_transits)[:, None], 2, axis=1) * ARCSEC
        residuals = np.repeat(residuals, 2, axis=0)
        obs_set = _make_gaia_set(residuals, w)

        cfg = OmegaConf.create(
            {"figures": {"residual_scatter_ellipses": {"ellipses": {"max_ellipses": 5}}}}
        )
        plot = _make_plot([obs_set], cfg)
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, plot._resolve_plot_cfg(cfg))
        assert len([p for p in ax.patches if isinstance(p, Ellipse)]) == 5
        plt.close(fig)


class TestOverlayEllipses:
    def test_draws_one_ellipse_per_transit(self):
        # Two transits of 2 correlated observations each, at distinct centers.
        c = np.diag([(2.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        w = _block_diag(_transit_weight(c, 2), _transit_weight(c, 2))
        residuals = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]]) * ARCSEC
        obs_set = _make_gaia_set(residuals, w)

        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"ellipses": {"n_sigma": 1.0}})
        plot = _make_plot([obs_set], cfg)
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, cfg)

        ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
        assert len(ellipses) == 2
        # Semi-major axis ~ 2 arcsec -> width ~ 4 arcsec at 1 sigma.
        assert ellipses[0].width == pytest.approx(4.0, rel=1e-2)
        assert ellipses[0].height == pytest.approx(2.0, rel=1e-2)
        plt.close(fig)

    def test_default_is_three_sigma(self):
        c = np.diag([(2.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        w = _transit_weight(c, 2)
        obs_set = _make_gaia_set(np.zeros((2, 2)), w)
        plot = _make_plot([obs_set])
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, None)
        ellipse = [p for p in ax.patches if isinstance(p, Ellipse)][0]
        # 3 sigma * 2 arcsec semi-major -> width ~ 12 arcsec.
        assert ellipse.width == pytest.approx(12.0, rel=1e-2)
        plt.close(fig)

    def test_ellipses_have_transparent_fill(self):
        c = np.diag([(1.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        w = _transit_weight(c, 2)
        obs_set = _make_gaia_set(np.zeros((2, 2)), w)
        plot = _make_plot([obs_set])
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, None)
        ellipse = [p for p in ax.patches if isinstance(p, Ellipse)][0]
        # Fill must be present but semi-transparent.
        assert ellipse.get_facecolor()[3] > 0.0
        assert ellipse.get_facecolor()[3] < 1.0
        plt.close(fig)

    def test_duplicate_ellipses_are_suppressed(self):
        # The same set listed twice must not double the number of ellipses.
        c = np.diag([(1.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        w = _transit_weight(c, 2)
        obs_set = _make_gaia_set(np.zeros((2, 2)), w)
        plot = _make_plot([obs_set, obs_set])
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set, obs_set]
        plot._overlay_ellipses(ax, None)
        assert len([p for p in ax.patches if isinstance(p, Ellipse)]) == 1
        plt.close(fig)

    def test_skips_sets_without_full_matrix(self):
        obs_set = _make_gaia_set(np.zeros((2, 2)), np.eye(4), has_full=False)
        plot = _make_plot([obs_set])
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, None)
        assert not [p for p in ax.patches if isinstance(p, Ellipse)]
        plt.close(fig)

    def test_min_obs_per_transit_filters_singletons(self):
        # A diagonal weight matrix -> every transit is a singleton -> filtered out.
        obs_set = _make_gaia_set(np.zeros((3, 2)), np.eye(6))
        plot = _make_plot([obs_set])
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, None)
        assert not [p for p in ax.patches if isinstance(p, Ellipse)]
        plt.close(fig)

    def test_max_ellipses_cap(self):
        # 5 transits of 2 obs at distinct centers, cap at 2.
        c = np.diag([(1.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        n_transits = 5
        w = _block_diag(*[_transit_weight(c, 2) for _ in range(n_transits)])
        residuals = np.repeat(np.arange(n_transits)[:, None], 2, axis=1) * ARCSEC
        residuals = np.repeat(residuals, 2, axis=0)
        obs_set = _make_gaia_set(residuals, w)

        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"ellipses": {"max_ellipses": 2}})
        plot = _make_plot([obs_set], cfg)
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, cfg)
        assert len([p for p in ax.patches if isinstance(p, Ellipse)]) == 2
        plt.close(fig)

    def test_ellipse_center_is_transit_mean(self):
        c = np.diag([(1.0 * ARCSEC) ** 2, (1.0 * ARCSEC) ** 2])
        w = _transit_weight(c, 2)
        # Mean RA residual = 3 arcsec, mean Dec = -2 arcsec.
        residuals = np.array([[2.0, -1.0], [4.0, -3.0]]) * ARCSEC
        obs_set = _make_gaia_set(residuals, w)

        plot = _make_plot([obs_set])
        fig, ax = plt.subplots()
        plot._observation_sets = [obs_set]
        plot._overlay_ellipses(ax, None)
        ellipse = [p for p in ax.patches if isinstance(p, Ellipse)][0]
        assert ellipse.center[0] == pytest.approx(3.0, rel=1e-6)
        assert ellipse.center[1] == pytest.approx(-2.0, rel=1e-6)
        plt.close(fig)
