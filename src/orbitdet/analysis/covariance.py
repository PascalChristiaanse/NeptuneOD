"""Covariance diagnostics for the analysis layer.

Per-observation sets carry a weight matrix ``W = C^{-1}``; the figures and
tables read *covariance* quantities off it - marginal 1σ uncertainties, error
ellipses, conditioning.  These are pure linear-algebra helpers so the same
convention is used everywhere (the plots currently reimplement several of them,
some of them subtly wrong).

The one rule worth stating up front: for a full (off-diagonal) weight matrix the
marginal 1σ is ``sqrt(diag(C))`` with ``C = W^{-1}``, **not** ``1/sqrt(diag(W))``.
The two agree only for a diagonal matrix; for a near-rank-deficient block such as
a Gaia transit covariance they differ by orders of magnitude.

Nothing here reads config or touches the file system.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "marginal_sigmas",
    "block_sigmas",
    "principal_axes",
    "condition_index",
]


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return ``(matrix + matrixᵀ) / 2`` as float64."""
    matrix = np.asarray(matrix, dtype="float64")
    return 0.5 * (matrix + matrix.T)


def _invert(matrix: np.ndarray) -> np.ndarray:
    """Invert, falling back to the pseudo-inverse for a singular matrix."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def marginal_sigmas(weight_matrix) -> np.ndarray:
    """Per-component marginal 1σ from a weight matrix ``W``.

    ``sigma_i = sqrt(diag(W^{-1}))`` - the marginal standard deviation, which is
    the correct per-component uncertainty for **any** weight matrix (diagonal,
    block-diagonal or full).  For a diagonal matrix it reduces to ``1/sqrt(diag W)``.
    A singular matrix is inverted with the pseudo-inverse.

    Parameters
    ----------
    weight_matrix : array-like
        The ``(n, n)`` weight matrix ``W``.

    Returns
    -------
    np.ndarray
        ``(n,)`` array of marginal 1σ uncertainties (same unit as the inverse of
        the weight's unit, i.e. the residual's unit).
    """
    covariance = _invert(_symmetrize(weight_matrix))
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def block_sigmas(weight_matrix, block_size: int = 2) -> np.ndarray:
    """Per-component 1σ from the diagonal blocks of a weight matrix.

    Inverts each ``block_size × block_size`` diagonal block on its own and reads
    its marginal sigmas.  This is exact for a block-diagonal (or diagonal) weight
    matrix and a *local* approximation for a full one, where it ignores
    cross-block correlations - which is exactly what the scan-frame residual
    figures need (they rotate each observation's own 2×2 block).

    Parameters
    ----------
    weight_matrix : array-like
        The ``(n, n)`` weight matrix ``W``, with ``n`` a multiple of ``block_size``.
    block_size : int
        Size of each diagonal block (``2`` for the frame's per-observation pairs).

    Returns
    -------
    np.ndarray
        ``(n,)`` array of per-component 1σ uncertainties.

    Raises
    ------
    ValueError
        If ``block_size`` is not positive or does not divide the matrix size.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    weight_matrix = _symmetrize(weight_matrix)
    size = weight_matrix.shape[0]
    if size % block_size:
        raise ValueError(f"Matrix size {size} is not a multiple of block_size {block_size}.")

    sigmas = np.empty(size, dtype="float64")
    for start in range(0, size, block_size):
        stop = start + block_size
        block = weight_matrix[start:stop, start:stop]
        sigmas[start:stop] = np.sqrt(np.clip(np.diag(_invert(block)), 0.0, None))
    return sigmas


def principal_axes(covariance, *, n_sigma: float = 1.0) -> tuple[np.ndarray, float]:
    """Principal semi-axes and orientation of a 2×2 covariance.

    The error ellipse of a 2-D Gaussian with covariance ``C`` has semi-axes
    ``n_sigma * sqrt(eigenvalue)`` oriented along the eigenvectors.  The returned
    angle is the orientation of the **major** axis, measured counter-clockwise
    from the ``x`` axis (the convention :class:`matplotlib.patches.Ellipse`
    expects).

    Parameters
    ----------
    covariance : array-like
        The ``(2, 2)`` covariance matrix.
    n_sigma : float
        Confidence scale of the ellipse (``1`` for 1σ).  Defaults to ``1``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(semi_axes, angle)`` where ``semi_axes`` is ``[major, minor]`` and
        ``angle`` is the major-axis orientation in **radians**.

    Raises
    ------
    ValueError
        If ``covariance`` is not ``2 × 2``.
    """
    covariance = np.asarray(covariance, dtype="float64")
    if covariance.shape != (2, 2):
        raise ValueError(f"principal_axes expects a 2x2 covariance, got {covariance.shape}.")

    eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(covariance))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # eigh returns ascending eigenvalues; reorder to [major, minor].
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    major_vector = eigenvectors[:, order[0]]

    semi_axes = n_sigma * np.sqrt(eigenvalues)
    angle = float(np.arctan2(major_vector[1], major_vector[0]))
    return semi_axes, angle


def condition_index(covariance) -> np.ndarray:
    """Condition indices ``sqrt(λ_max / λ_i)`` of a covariance matrix.

    Each eigen-direction is scaled so that the best-constrained direction gives
    ``1``; a large index flags a poorly constrained mode.  Eigenvalues are
    clipped at zero (numerical noise) and sorted descending.

    Parameters
    ----------
    covariance : array-like
        The ``(n, n)`` covariance matrix.

    Returns
    -------
    np.ndarray
        ``(n,)`` array of condition indices, largest first (``inf`` where an
        eigenvalue is exactly zero).
    """
    covariance = _symmetrize(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues_safe = np.clip(eigenvalues, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.sqrt(eigenvalues_safe[0] / eigenvalues_safe)
    return np.where(np.isfinite(index), index, np.inf)
