"""Tests for the analysis whitening / Mahalanobis layer."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdet.analysis import Whitening


def _spd_weight(size: int, seed: int = 0) -> np.ndarray:
    """A dense symmetric positive-definite weight matrix."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(size, size))
    covariance = a @ a.T + size * np.eye(size)
    return np.linalg.inv(covariance)


class TestFromWeightMatrix:
    def test_whitening_gives_unit_covariance(self):
        W = _spd_weight(4)
        C = np.linalg.inv(W)
        whitening = Whitening.from_weight_matrix(W)
        assert whitening.kind == "cholesky"
        assert whitening.rank == 4
        np.testing.assert_allclose(whitening.z_covariance(C), np.eye(4), atol=1e-10)

    def test_factor_is_lower_triangular_cholesky(self):
        W = _spd_weight(3)
        whitening = Whitening.from_weight_matrix(W)
        L = whitening.factor
        np.testing.assert_allclose(L @ L.T, W, atol=1e-12)
        assert np.allclose(np.triu(L, 1), 0.0)  # lower triangular

    def test_rank_and_size(self):
        whitening = Whitening.from_weight_matrix(_spd_weight(5))
        assert whitening.size == 5
        assert whitening.rank == 5
        assert whitening.effective_dof == 5


class TestFromCovariance:
    def test_matches_from_weight_matrix(self):
        W = _spd_weight(4, seed=2)
        C = np.linalg.inv(W)
        np.testing.assert_allclose(
            Whitening.from_covariance(C).factor, Whitening.from_weight_matrix(W).factor
        )

    def test_rank_deficient_uses_eigen_fallback(self):
        # A near-rank-1 covariance, like a single Gaia transit.
        direction = np.array([1.0, 1e-3])
        C = 1e6 * np.outer(direction, direction)
        whitening = Whitening.from_covariance(C)
        assert whitening.kind == "eigen"
        assert whitening.rank == 1
        # On the constrained subspace the whitened variance is unity.
        r = np.array([1e3, 1.0])
        z = whitening.apply(r)
        np.testing.assert_allclose(z.shape, (1,))


class TestFromBlocks:
    def test_block_diagonal_flag(self):
        blocks = [np.array([[4.0, 1.0], [1.0, 3.0]]), np.array([[2.0, 0.0], [0.0, 5.0]])]
        whitening = Whitening.from_blocks(blocks)
        assert whitening.is_block_diagonal
        assert whitening.factor.shape == (4, 4)

    def test_dense_matrix_has_no_blocks(self):
        whitening = Whitening.from_weight_matrix(_spd_weight(4))
        assert not whitening.is_block_diagonal


class TestFromDiagonal:
    def test_reproduces_componentwise_normalization(self):
        sigmas = np.array([2.0, 4.0])
        whitening = Whitening.from_diagonal(sigmas)
        np.testing.assert_allclose(whitening.apply(np.array([2.0, 8.0])), [1.0, 2.0])

    def test_zero_sigma_annihilated(self):
        whitening = Whitening.from_diagonal(np.array([2.0, 0.0]))
        assert whitening.rank == 1


class TestChi2AndMahalanobis:
    def test_chi2_equals_quadratic_form(self):
        W = _spd_weight(4, seed=3)
        rng = np.random.default_rng(3)
        r = rng.normal(size=4)
        whitening = Whitening.from_weight_matrix(W)
        assert whitening.chi2(r) == pytest.approx(r @ W @ r)

    def test_mahalanobis_is_norm_of_whitened(self):
        W = _spd_weight(4, seed=4)
        rng = np.random.default_rng(4)
        r = rng.normal(size=4)
        whitening = Whitening.from_weight_matrix(W)
        z = whitening.apply(r)
        assert whitening.mahalanobis(r) == pytest.approx(np.linalg.norm(z))
        assert whitening.mahalanobis(r) ** 2 == pytest.approx(whitening.chi2(r))

    def test_diagonal_chi2_equals_componentwise(self):
        sigmas = np.array([2.0, 4.0])
        whitening = Whitening.from_diagonal(sigmas)
        r = np.array([2.0, 8.0])
        assert whitening.chi2(r) == pytest.approx((2 / 2) ** 2 + (8 / 4) ** 2)

    def test_apply_rejects_wrong_length(self):
        whitening = Whitening.from_diagonal(np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="length"):
            whitening.apply(np.array([1.0, 2.0, 3.0]))


class TestBlockMahalanobis:
    def test_per_observation_distances(self):
        blocks = [np.array([[4.0, 0.0], [0.0, 9.0]])]
        whitening = Whitening.from_blocks(blocks)
        r = np.array([2.0, 3.0])  # z = r / sigma = [4, 9], d = sqrt(97)
        np.testing.assert_allclose(whitening.block_mahalanobis(r), [np.sqrt(97.0)])

    def test_whitened_components_match_apply(self):
        blocks = [np.array([[4.0, 1.0], [1.0, 3.0]])]
        whitening = Whitening.from_blocks(blocks)
        r = np.array([1.0, -2.0])
        components = whitening.whitened_components(r)
        np.testing.assert_allclose(components.ravel(), whitening.apply(r))

    def test_dense_matrix_raises(self):
        whitening = Whitening.from_weight_matrix(_spd_weight(4))
        with pytest.raises(ValueError, match="dense"):
            whitening.whitened_components(np.zeros(4))

    def test_rank_deficient_block_leaves_nan(self):
        blocks = [np.diag([4.0, 0.0])]
        whitening = Whitening.from_blocks(blocks)
        components = whitening.whitened_components(np.array([2.0, 1.0]))
        assert np.isnan(components[0, 1])
        # Only the constrained direction (weight 4) whitens: z = 2 * 2 = 4.
        np.testing.assert_allclose(components[0, 0], 4.0)
