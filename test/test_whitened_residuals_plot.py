"""Tests for symmetric whitening."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdet.analysis import Whitening


def _correlated_weight(sigma_al=1e-6, sigma_ac=2e-6, rho=0.8):
    """A block-diagonal weight matrix with a correlated 2x2 block per observation."""
    cov = np.array(
        [
            [sigma_al**2, rho * sigma_al * sigma_ac],
            [rho * sigma_al * sigma_ac, sigma_ac**2],
        ]
    )
    block = np.linalg.inv(cov)
    weight = np.zeros((4, 4))
    weight[0:2, 0:2] = block
    weight[2:4, 2:4] = block
    return weight


# ===========================================================================
# Symmetric whitening
# ===========================================================================


class TestSymmetricWhitening:
    def test_factor_is_symmetric(self):
        weight = _correlated_weight()
        whitening = Whitening.from_weight_matrix(weight, method="symmetric")
        assert whitening.kind == "symmetric"
        np.testing.assert_allclose(whitening.factor, whitening.factor.T, atol=1e-12)

    def test_cholesky_factor_is_triangular(self):
        weight = _correlated_weight()
        whitening = Whitening.from_weight_matrix(weight, method="cholesky")
        assert whitening.kind == "cholesky"
        np.testing.assert_allclose(whitening.factor, np.tril(whitening.factor), atol=1e-12)

    def test_both_methods_give_unit_covariance(self):
        weight = _correlated_weight()
        covariance = np.linalg.inv(weight)
        for method in ("cholesky", "symmetric"):
            whitening = Whitening.from_weight_matrix(weight, method=method)
            cov_z = whitening.factor.T @ covariance @ whitening.factor
            np.testing.assert_allclose(cov_z, np.eye(4), atol=1e-9)

    def test_both_methods_give_same_chi2(self):
        weight = _correlated_weight()
        residual = np.array([1e-6, 2e-6, 3e-6, -4e-6])
        cholesky = Whitening.from_weight_matrix(weight, method="cholesky")
        symmetric = Whitening.from_weight_matrix(weight, method="symmetric")
        assert cholesky.chi2(residual) == pytest.approx(symmetric.chi2(residual), rel=1e-9)
        assert cholesky.chi2(residual) == pytest.approx(residual @ weight @ residual, rel=1e-9)

    def test_symmetric_spreads_correlation(self):
        # Cholesky dumps the correlation onto component 1 (lower-triangular);
        # the symmetric root mixes both components.
        weight = _correlated_weight()
        cholesky = Whitening.from_weight_matrix(weight, method="cholesky")
        symmetric = Whitening.from_weight_matrix(weight, method="symmetric")
        # Cholesky's first row has no dependence on the second component.
        assert cholesky.factor[0, 1] == pytest.approx(0.0, abs=1e-15)
        # The symmetric root's off-diagonal entries are non-zero.
        assert abs(symmetric.factor[0, 1]) > 1e-3

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown whitening method"):
            Whitening.from_weight_matrix(_correlated_weight(), method="bogus")

    def test_symmetric_block_diagonal_detected(self):
        weight = _correlated_weight()
        whitening = Whitening.from_weight_matrix(weight, method="symmetric")
        assert whitening.is_block_diagonal
