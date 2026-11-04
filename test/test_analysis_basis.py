"""Tests for the analysis basis transforms."""

from __future__ import annotations

import numpy as np

from orbitdet.analysis import BasisTransform, block_diagonal_basis, identity_basis, scan_basis


class TestScanBasis:
    def test_matches_weighting_strategy_convention(self):
        """The 2x2 scan rotation equals the T used by the weighting strategies."""
        psi, dec = 0.7, -0.4
        t = scan_basis(psi, dec=dec).transform
        cos_dec = np.cos(dec)
        expected = np.array(
            [
                [np.sin(psi) * cos_dec, np.cos(psi)],
                [np.cos(psi) * cos_dec, -np.sin(psi)],
            ]
        )
        np.testing.assert_allclose(t, expected)

    def test_de_projected_rotation_is_orthogonal(self):
        t = scan_basis(0.7)  # dec=None -> de-projected, 2x2
        assert t.is_orthogonal
        # Orthogonal => covariance and weight use the same congruence.
        C = np.array([[4.0, 1.2], [1.2, 2.5]])
        np.testing.assert_allclose(t.weight(C), t.covariance(C))

    def test_block_diagonal_of_de_projected_is_orthogonal(self):
        psi = np.array([0.3, 1.1, -2.0])
        t = scan_basis(psi)  # dec=None -> de-projected
        assert t.is_orthogonal
        assert t.transform.shape == (6, 6)

    def test_raw_rotation_is_not_orthogonal_and_dual_differs(self):
        psi, dec = 0.7, -0.4
        t = scan_basis(psi, dec=dec)  # raw -> cos(dec) applied
        assert not t.is_orthogonal
        W = np.array([[0.3, 0.05], [0.05, 0.6]])
        assert not np.allclose(t.weight(W), t.covariance(W))

    def test_block_diagonal_for_many_observations(self):
        psi = np.array([0.1, 0.9])
        t = scan_basis(psi, dec=np.array([0.2, -0.3]))
        assert t.transform.shape == (4, 4)
        # Block-diagonal: cross blocks are zero.
        np.testing.assert_allclose(t.transform[:2, 2:], 0.0)

    def test_vector_matches_matrix_rotation(self):
        """scan_basis(...).residuals equals views.rotate for raw (non-de-projected)."""
        from orbitdet.analysis.views import rotate

        psi = np.array([0.5])
        dec = np.array([0.3])
        first = np.array([1.0])
        second = np.array([2.0])

        t = scan_basis(psi, dec=dec)
        got = t.residuals(np.array([first[0], second[0]]))

        east = first * np.cos(dec)
        al, ac = rotate(first, second, dec, psi)
        np.testing.assert_allclose(got, [al[0], ac[0]])
        # And the de-projected variant equals rotate(..., de_projected=True).
        t2 = scan_basis(psi)
        got2 = t2.residuals(np.array([first[0], second[0]]))
        al_dp, ac_dp = rotate(first, second, dec, psi, de_projected=True)
        np.testing.assert_allclose(got2, [al_dp[0], ac_dp[0]])
        assert east[0] != first[0]  # sanity: the factor actually did something


class TestBasisTransforms:
    def test_identity_leaves_everything_unchanged(self):
        basis = identity_basis(4)
        C = np.diag([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(basis.covariance(C), C)
        np.testing.assert_allclose(basis.weight(C), C)

    def test_covariance_weight_round_trip(self):
        basis = scan_basis(np.array([0.4, 1.3]), dec=np.array([0.1, -0.2]))
        # A positive-definite covariance, and its weight matrix W = C^-1.
        rng = np.random.default_rng(0)
        a = rng.normal(size=(4, 4))
        C = a @ a.T + 4 * np.eye(4)
        W = np.linalg.inv(C)

        # The weight transform is the inverse of the covariance transform.
        np.testing.assert_allclose(basis.weight(W), np.linalg.inv(basis.covariance(C)))
        # Round-trip back to canonical reproduces C.
        back = BasisTransform(np.linalg.inv(basis.transform))
        np.testing.assert_allclose(back.covariance(basis.covariance(C)), C)

    def test_covariance_is_congruence(self):
        basis = scan_basis(np.array([0.4]), dec=np.array([0.1]))
        C = np.array([[4.0, 1.2], [1.2, 2.5]])
        t = basis.transform
        np.testing.assert_allclose(basis.covariance(C), t @ C @ t.T)

    def test_named_constructors(self):
        assert scan_basis(0.1).name == "scan"
        assert identity_basis(2).name == "canonical"
        blocks = [np.eye(2), np.eye(2)]
        assert block_diagonal_basis(blocks).transform.shape == (4, 4)
