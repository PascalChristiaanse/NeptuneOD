"""Whitening and Mahalanobis distances for correlated observations.

A set of observations whose residuals are correlated cannot be reduced to
independent unit-variance variables by dividing each component by its own
marginal sigma.  The cross-observation correlation survives, the variance is
inflated (for a near-rank-deficient Gaia transit covariance, by orders of
magnitude), and any statistic built from ``r / sigma`` is wrong.  The correct
reduction is **whitening** through the Cholesky factor of the weight matrix.

Convention
----------
The applied weight matrix is the inverse covariance ``W = C^{-1}``.  Its
Cholesky factor ``L`` (lower triangular) satisfies ``W = L L^T``, and the
**whitened** residual is::

    z = L^T r,

so that ``Cov(z) = L^T C L = I`` exactly (verified to machine precision).  The
factor also yields the **Mahalanobis distance** for free::

    d = ||z|| = sqrt(r^T W r),

i.e. the chi-squared of the set is the squared Mahalanobis distance.  The
quadratic form is invariant under a change of basis (a rotation ``T`` applied to
both ``r`` and ``W`` leaves ``r^T W r`` unchanged), so ``d`` does not depend on
the basis; the whitened *vector* does.

Rank-deficient sets
-------------------
A near-rank-deficient weight matrix - e.g. a single Gaia transit covariance,
whose archived matrix is essentially rank-1 - has no lower-triangular Cholesky
factor.  For such a matrix the module falls back to a symmetric positive
semi-definite factor from the eigendecomposition, keeping only the directions
that carry information and recording :attr:`Whitening.rank` (the number of
independent constraints, which is the correct count of degrees of freedom).  On
the constrained subspace ``Cov(z) = I``; the unconstrained directions are
annihilated rather than whitened, because a direction with no information cannot
be whitened.

Block structure
---------------
When the weight matrix is block-diagonal (diagonal or per-observation 2x2), the
whitened residuals are again per-observation pairs and
:meth:`Whitening.whitened_components` returns them as an ``(n, 2)`` array, so
they can be attached to the frame.  A dense (cross-observation) matrix mixes
observations and has no per-observation pair representation; use
:meth:`Whitening.apply` for its whitened vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Whitening"]

#: Relative tolerance used when deciding a bag of eigenvalues is zero.
_DEFAULT_RTOL = 1e-10


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return ``(matrix + matrix^T) / 2`` as float64 (a defensible covariance)."""
    matrix = np.asarray(matrix, dtype="float64")
    return 0.5 * (matrix + matrix.T)


def _factor_weight(weight_matrix: np.ndarray, tol: float | None) -> tuple[np.ndarray, int, str]:
    """Factor a weight matrix as ``L L^T``, lower-triangular when possible.

    Parameters
    ----------
    weight_matrix : np.ndarray
        A symmetric positive (semi-)definite weight matrix.
    tol : float | None
        Absolute eigenvalue threshold below which a direction is treated as
        unconstrained.  ``None`` derives it from the largest eigenvalue.

    Returns
    -------
    tuple[np.ndarray, int, str]
        ``(L, rank, kind)`` where ``L L^T == W`` (on the constrained subspace for
        the rank-deficient case), ``rank`` is the number of retained directions,
        and ``kind`` is ``"cholesky"`` or ``"eigen"``.
    """
    weight_matrix = _symmetrize(weight_matrix)
    size = weight_matrix.shape[0]
    # Prefer a true lower-triangular Cholesky factor: it matches z = L^T r and
    # exists whenever W is positive definite.
    try:
        return np.linalg.cholesky(weight_matrix), size, "cholesky"
    except np.linalg.LinAlgError:
        pass

    # Positive semi-definite fallback: keep the informative directions only.
    eigenvalues, eigenvectors = np.linalg.eigh(weight_matrix)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    threshold = (tol if tol is not None else _DEFAULT_RTOL * scale) if scale else 0.0
    keep = eigenvalues > threshold
    rank = int(np.count_nonzero(keep))
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    return factor, rank, "eigen"


def _factor_weight_symmetric(
    weight_matrix: np.ndarray, tol: float | None
) -> tuple[np.ndarray, int, str]:
    """Factor a weight matrix as the symmetric square root ``W^{1/2}``.

    Unlike the Cholesky factor, the symmetric square root ``S = V sqrt(Lambda) V^T``
    privileges no component: it spreads the correlation evenly over the pair
    instead of dumping it onto the first component.  It still satisfies
    ``W = S S`` (so ``S^T = S`` and ``z = S r`` has unit covariance), but the
    whitened components are symmetric mixtures rather than a triangular one.

    This is the ZCA whitening transform: of all whitening matrices it is the one
    closest to the identity, so the whitened residuals stay as similar as
    possible to the originals while still being decorrelated.

    Parameters
    ----------
    weight_matrix : np.ndarray
        A symmetric positive (semi-)definite weight matrix.
    tol : float | None
        Relative threshold below which an eigenvalue counts as zero.

    Returns
    -------
    tuple[np.ndarray, int, str]
        The ``(n, n)`` symmetric factor (rank ``r`` when rank-deficient, with the
        null directions annihilated), its rank, and ``"symmetric"``.
    """
    weight_matrix = _symmetrize(weight_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(weight_matrix)
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    threshold = (tol if tol is not None else _DEFAULT_RTOL * scale) if scale else 0.0
    keep = eigenvalues > threshold
    rank = int(np.count_nonzero(keep))
    # S = V sqrt(Lambda) V^T, restricted to the informative subspace.  Keeping the
    # full (n, n) shape means z = S r has the same length as r, so the whitened
    # vector still lines up with the observations; the null directions map to 0.
    scaled = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    factor = scaled @ eigenvectors[:, keep].T
    return factor, rank, "symmetric"


def _block_factors(
    weight_matrix: np.ndarray,
    block_size: int,
    tol: float | None,
    method: str = "cholesky",
) -> tuple[np.ndarray, ...] | None:
    """Return per-block factors if ``weight_matrix`` is block-diagonal.

    Parameters
    ----------
    weight_matrix : np.ndarray
        A symmetric matrix.
    block_size : int
        Size of each diagonal block (``2`` for the per-observation pairs of the
        derived frame).
    tol : float | None
        Absolute threshold below which an off-block entry counts as zero.
    method : str
        ``"cholesky"`` or ``"symmetric"``; selects the per-block factorisation.

    Returns
    -------
    tuple[np.ndarray, ...] | None
        One factor per block, or ``None`` when the matrix is not block-diagonal
        (in which case the observations are correlated across blocks).
    """
    size = weight_matrix.shape[0]
    if block_size <= 0 or size % block_size:
        return None
    n_blocks = size // block_size
    scale = float(np.max(np.abs(weight_matrix))) if weight_matrix.size else 0.0
    if scale == 0.0:
        return None
    threshold = tol if tol is not None else _DEFAULT_RTOL * scale

    mask = np.zeros((size, size), dtype=bool)
    for i in range(n_blocks):
        s = slice(i * block_size, (i + 1) * block_size)
        mask[s, s] = True
    if np.any(np.abs(weight_matrix[~mask]) > threshold):
        return None

    factor_fn = _factor_weight_symmetric if method == "symmetric" else _factor_weight
    factors = []
    for i in range(n_blocks):
        s = slice(i * block_size, (i + 1) * block_size)
        factor, _, _ = factor_fn(weight_matrix[s, s], tol)
        factors.append(factor)
    return tuple(factors)


@dataclass(frozen=True)
class Whitening:
    """A whitening transform ``z = L^T r`` for one correlated observation set.

    Build it from the applied weight matrix, the covariance, or the per-observation
    blocks.  The factor satisfies ``W = L L^T``, so ``z = L^T r`` has unit
    covariance and ``||z||^2 = r^T W r`` is the set's chi-squared.

    Parameters
    ----------
    factor : np.ndarray
        The ``(n, r)`` factor ``L`` (``r == n`` for a full-rank Cholesky factor,
        ``r < n`` for a rank-deficient matrix).
    rank : int
        Number of independent directions (constraints).
    kind : str
        ``"cholesky"``, ``"symmetric"``, or ``"eigen"`` (the rank-deficient
        Cholesky fallback).
    blocks : tuple[np.ndarray, ...] | None
        Per-block factors when the matrix is block-diagonal, else ``None``.
    block_size : int
        Size of the per-block structure (``2`` for observation pairs).
    """

    factor: np.ndarray
    rank: int
    kind: str
    blocks: tuple[np.ndarray, ...] | None = field(default=None, repr=False, compare=False)
    block_size: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor", np.asarray(self.factor, dtype="float64"))

    # -- Construction ---------------------------------------------------

    @classmethod
    def from_weight_matrix(
        cls,
        weight_matrix: np.ndarray,
        *,
        tol: float | None = None,
        block_size: int = 2,
        method: str = "cholesky",
    ) -> Whitening:
        """Build the whitening from a weight matrix ``W`` (``= C^{-1}``).

        Parameters
        ----------
        weight_matrix : np.ndarray
            The applied ``(n, n)`` weight matrix, symmetric positive
            (semi-)definite.
        tol : float | None
            Absolute eigenvalue threshold for the rank-deficient fallback.
        block_size : int
            Per-block size used to detect block-diagonal structure.
        method : str
            ``"cholesky"`` (default) for the lower-triangular factor, which
            concentrates the correlation onto the first component of each block,
            or ``"symmetric"`` for the symmetric square root ``W^{1/2}``, which
            spreads the correlation evenly across the components.  Both give
            ``Cov(z) = I``; they differ only in how the mixing is distributed.

        Returns
        -------
        Whitening
            The factor with ``z = L^T r``.

        Raises
        ------
        ValueError
            If ``method`` is not ``"cholesky"`` or ``"symmetric"``.
        """
        weight_matrix = _symmetrize(weight_matrix)
        if method == "cholesky":
            factor, rank, kind = _factor_weight(weight_matrix, tol)
        elif method == "symmetric":
            factor, rank, kind = _factor_weight_symmetric(weight_matrix, tol)
        else:
            raise ValueError(
                f"Unknown whitening method '{method}'; expected 'cholesky' or 'symmetric'."
            )
        blocks = _block_factors(weight_matrix, block_size, tol, method=method)
        return cls(factor, rank, kind, blocks=blocks, block_size=block_size)

    @classmethod
    def from_covariance(
        cls,
        covariance: np.ndarray,
        *,
        tol: float | None = None,
        block_size: int = 2,
    ) -> Whitening:
        """Build the whitening from a covariance ``C`` (``= W^{-1}``).

        The covariance is inverted (with the pseudo-inverse for a singular
        matrix) and the result factored, so a rank-deficient covariance still
        yields the constrained subspace.

        Parameters
        ----------
        covariance : np.ndarray
            The ``(n, n)`` covariance matrix.
        tol : float | None
            Absolute eigenvalue threshold for the rank-deficient fallback.
        block_size : int
            Per-block size used to detect block-diagonal structure.

        Returns
        -------
        Whitening
            The factor with ``z = L^T r``.
        """
        covariance = _symmetrize(covariance)
        try:
            weight_matrix = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            weight_matrix = np.linalg.pinv(covariance)
        return cls.from_weight_matrix(weight_matrix, tol=tol, block_size=block_size)

    @classmethod
    def from_blocks(
        cls,
        blocks: list[np.ndarray],
        *,
        tol: float | None = None,
    ) -> Whitening:
        """Build the whitening from explicit per-observation weight blocks.

        Parameters
        ----------
        blocks : list[np.ndarray]
            One ``(k, k)`` weight block per observation (``k = 2`` for the
            derived frame's pairs).
        tol : float | None
            Absolute eigenvalue threshold for the rank-deficient fallback.

        Returns
        -------
        Whitening
            The whitening of the assembled block-diagonal weight matrix.
        """
        blocks = [np.asarray(block, dtype="float64") for block in blocks]
        if blocks:
            size = blocks[0].shape[0]
            weight_matrix = np.zeros((size * len(blocks), size * len(blocks)))
            for i, block in enumerate(blocks):
                weight_matrix[i * size : (i + 1) * size, i * size : (i + 1) * size] = block
        else:
            weight_matrix = np.zeros((0, 0))
        return cls.from_weight_matrix(weight_matrix, tol=tol, block_size=size if blocks else 2)

    @classmethod
    def from_diagonal(cls, sigmas: np.ndarray) -> Whitening:
        """Build the whitening from per-component 1σ uncertainties.

        For a diagonal weight matrix ``W = diag(1/σ²)`` the Cholesky factor is
        ``L = diag(1/σ)``, so ``z = r / σ`` reproduces the componentwise
        normalization.

        Parameters
        ----------
        sigmas : np.ndarray
            One 1σ per component.

        Returns
        -------
        Whitening
            The diagonal whitening.
        """
        sigmas = np.asarray(sigmas, dtype="float64").ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(sigmas > 0.0, 1.0 / sigmas**2, 0.0)
        return cls.from_weight_matrix(np.diag(weights))

    # -- Transformations ------------------------------------------------

    @property
    def size(self) -> int:
        """Dimension of the residual vectors this whitening acts on (``n``)."""
        return int(self.factor.shape[0])

    @property
    def effective_dof(self) -> int:
        """Number of independent constraints (the rank); the chi-squared DOF."""
        return self.rank

    @property
    def is_block_diagonal(self) -> bool:
        """Whether the whitened residuals are per-observation pairs."""
        return self.blocks is not None

    def apply(self, residuals: np.ndarray) -> np.ndarray:
        """Return the whitened residual vector ``z = L^T r``.

        Parameters
        ----------
        residuals : np.ndarray
            The ``(n,)`` residual vector (interleaved ``[first_1, second_1,
            ...]``).

        Returns
        -------
        np.ndarray
            The ``(rank,)`` whitened vector, with unit covariance on the
            constrained subspace.

        Raises
        ------
        ValueError
            If ``residuals`` does not have length ``n``.
        """
        residuals = np.asarray(residuals, dtype="float64").ravel()
        if residuals.size != self.size:
            raise ValueError(f"Residual vector has length {residuals.size}, expected {self.size}.")
        return self.factor.T @ residuals

    def z_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Return ``Cov(z) = L^T C L`` for the given residual covariance ``C``.

        With ``C = W^{-1}`` this is the identity (on the constrained subspace),
        which is the defining property of a whitening transform.

        Parameters
        ----------
        covariance : np.ndarray
            The residual covariance ``(n, n)``.

        Returns
        -------
        np.ndarray
            The ``(rank, rank)`` covariance of the whitened vector.
        """
        covariance = np.asarray(covariance, dtype="float64")
        return self.factor.T @ covariance @ self.factor

    def chi2(self, residuals: np.ndarray) -> float:
        """Return ``r^T W r = ||z||^2``, the set's chi-squared.

        Parameters
        ----------
        residuals : np.ndarray
            The ``(n,)`` residual vector.

        Returns
        -------
        float
            The chi-squared sum.
        """
        z = self.apply(residuals)
        return float(z @ z)

    def mahalanobis(self, residuals: np.ndarray) -> float:
        """Return the Mahalanobis distance ``||z|| = sqrt(r^T W r)``.

        Parameters
        ----------
        residuals : np.ndarray
            The ``(n,)`` residual vector.

        Returns
        -------
        float
            The Mahalanobis distance of the whole vector.
        """
        return float(np.sqrt(self.chi2(residuals)))

    def block_mahalanobis(self, residuals: np.ndarray) -> np.ndarray:
        """Return one Mahalanobis distance per observation block.

        Only defined for a block-diagonal (or diagonal) weight matrix, where each
        observation is independent.  For a dense matrix the observations are
        correlated and only the whole-vector :meth:`mahalanobis` is meaningful.

        Parameters
        ----------
        residuals : np.ndarray
            The ``(n,)`` residual vector.

        Returns
        -------
        np.ndarray
            ``(n_blocks,)`` array of per-observation Mahalanobis distances.

        Raises
        ------
        ValueError
            If the weight matrix is not block-diagonal.
        """
        components = self.whitened_components(residuals)
        return np.sqrt(np.nansum(components**2, axis=1))

    def whitened_components(self, residuals: np.ndarray) -> np.ndarray:
        """Return the whitened residuals as per-observation ``(n_blocks, 2)`` pairs.

        The two columns are the whitened ``first`` / ``second`` components.  When
        a block is rank-deficient (fewer than ``block_size`` informative
        directions) the missing components are ``NaN``.

        Parameters
        ----------
        residuals : np.ndarray
            The ``(n,)`` residual vector.

        Returns
        -------
        np.ndarray
            ``(n_blocks, block_size)`` array of whitened components.

        Raises
        ------
        ValueError
            If the weight matrix is not block-diagonal, or the residual vector
            length does not match.
        """
        if self.blocks is None:
            raise ValueError(
                "This weight matrix is dense (off-diagonal across observations), so its "
                "whitened residuals have no per-observation pair representation; use "
                "apply() for the whitened vector instead."
            )
        residuals = np.asarray(residuals, dtype="float64").ravel()
        n_blocks = len(self.blocks)
        if residuals.size != n_blocks * self.block_size:
            raise ValueError(
                f"Residual vector has length {residuals.size}, expected "
                f"{n_blocks * self.block_size}."
            )
        pairs = residuals.reshape(n_blocks, self.block_size)
        out = np.full((n_blocks, self.block_size), np.nan)
        for i, block in enumerate(self.blocks):
            whitened = block.T @ pairs[i]
            out[i, : whitened.size] = whitened
        return out
