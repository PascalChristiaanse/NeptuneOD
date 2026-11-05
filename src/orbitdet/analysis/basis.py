"""Basis transforms for the analysis layer.

A *basis transform* is a linear change of the two scalar components of an
observation - for example from the canonical ``(first, second)`` frame to the
along-scan / across-scan (AL/AC) frame a scanning instrument is diagonal in.
It is pure linear algebra with no knowledge of configurations, collections or
the file system, so it can be unit-tested on plain arrays.

Vectors, covariances and weight matrices transform differently
--------------------------------------------------------------
A transform ``T`` maps a residual vector as ``r' = T r``.  The covariance and
the weight matrix do **not** both map as ``T (.) T^T``:

* covariance (a congruence):  ``C' = T C T^T``
* weight matrix (the dual):   ``W' = T^{-T} W T^{-1}``

These coincide only when ``T`` is orthogonal (``T^{-1} = T^T``).  The scan
rotation carries a ``cos(δ)`` factor because the pipeline's first component is
**raw right ascension** ``α`` (the Gaia archive stores true ``α``; its
``ra_error`` is the de-projected ``σ_{α cosδ}``, which the bindings divide back
out, and the observable is built from ``α``).  That factor makes ``T``
non-orthogonal, so the two transforms differ and using ``T W T^T`` would be
silently wrong.  If a caller has already de-projected the components (so the
first holds ``α·cosδ``) the ``cos(δ)`` factor is dropped and ``T`` becomes
orthogonal; this module computes the correct dual either way.

:meth:`BasisTransform.weight` is therefore the method worth centralising: it is
the one a caller would otherwise get wrong by writing ``T @ W @ T.T``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["BasisTransform", "scan_basis", "identity_basis", "block_diagonal_basis"]


def _scan_rotation_2x2(psi: float, cos_dec: float | None) -> np.ndarray:
    """Return the 2x2 rotation taking ``(first, second)`` residuals to AL/AC.

    The convention matches :func:`orbitdet.observations.weighting.strategies`
    and :func:`orbitdet.analysis.views.rotate`::

        AL = first·cos(δ)·sin(ψ) + second·cos(ψ)
        AC = first·cos(δ)·cos(ψ) - second·sin(ψ)

    so the matrix is::

        T = [[sin(ψ)·cos(δ),  cos(ψ)],
             [cos(ψ)·cos(δ), -sin(ψ)]]

    with ``psi`` measured from North towards East.  When the first component is
    already de-projected by ``cos(δ)`` pass ``cos_dec=None`` and the factor is
    dropped, giving the orthogonal ``[[sin(ψ), cos(ψ)], [cos(ψ), -sin(ψ)]]``.
    """
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    factor = 1.0 if cos_dec is None else cos_dec
    return np.array(
        [
            [sin_psi * factor, cos_psi],
            [cos_psi * factor, -sin_psi],
        ]
    )


@dataclass(frozen=True)
class BasisTransform:
    """A linear basis change on observation-component vectors.

    Parameters
    ----------
    transform : np.ndarray
        The ``(m, m)`` matrix ``T`` with ``r' = T @ r``.  ``m`` is ``2`` for a
        single pair of components, or ``2n`` for the block-diagonal transform of
        all observations in a set.
    name : str
        Label for the target frame (``"canonical"``, ``"scan"``, ...).
    de_projected : bool
        Whether the transform was built from already de-projected components
        (``True``) or from raw components with a ``cos(δ)`` factor applied
        (``False``).  Informational only; it records the convention so a
        consumer can tell how the transform was derived.
    """

    transform: np.ndarray
    name: str = "canonical"
    de_projected: bool = True
    _inverse: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", np.asarray(self.transform, dtype="float64"))

    # -- Construction ---------------------------------------------------

    @classmethod
    def identity(cls, size: int, name: str = "canonical") -> BasisTransform:
        """Return the identity transform of the given size.

        Parameters
        ----------
        size : int
            Dimension of the vector the transform acts on.
        name : str
            Label for the frame.

        Returns
        -------
        BasisTransform
            The identity basis.
        """
        return cls(np.eye(size), name=name)

    @classmethod
    def scan(
        cls,
        psi: float | np.ndarray,
        dec: float | np.ndarray | None = None,
    ) -> BasisTransform:
        """Build the scan-frame rotation (AL/AC) for one or many observations.

        Parameters
        ----------
        psi : float | np.ndarray
            Position angle of scan, measured north through east (radians).  A
            scalar builds a ``2x2`` transform; an array of ``n`` angles builds
            the ``2n x 2n`` block-diagonal transform for ``n`` observations.
        dec : float | np.ndarray | None
            Declination (radians) applied as the tangent-plane ``cos(δ)`` factor.
            ``None`` means the components are already de-projected, so no factor
            is applied and the transform is orthogonal.

        Returns
        -------
        BasisTransform
            The scan basis.  ``de_projected`` is ``True`` when ``dec`` is
            ``None``.
        """
        psi = np.asarray(psi, dtype="float64")
        if psi.ndim == 0:
            cos_dec = None if dec is None else float(np.cos(dec))
            return cls(
                _scan_rotation_2x2(float(psi), cos_dec), name="scan", de_projected=dec is None
            )

        cos_dec_array = None if dec is None else np.cos(np.asarray(dec, dtype="float64"))
        blocks = [
            _scan_rotation_2x2(float(psi[i]), None if cos_dec_array is None else cos_dec_array[i])
            for i in range(psi.size)
        ]
        return cls(
            block_diagonal_matrix(blocks),
            name="scan",
            de_projected=dec is None,
        )

    # -- Transformations ------------------------------------------------

    def _inverse_matrix(self) -> np.ndarray:
        cached = self._inverse
        if cached is None:
            cached = np.linalg.inv(self.transform)
            object.__setattr__(self, "_inverse", cached)
        return cached

    @property
    def is_orthogonal(self) -> bool:
        """Whether ``T`` is orthogonal (``T T^T = I``), so dual transforms coincide."""
        t = self.transform
        return bool(np.allclose(t @ t.T, np.eye(t.shape[0])))

    def residuals(self, values: np.ndarray) -> np.ndarray:
        """Map component values (or residuals) into the target basis: ``T @ r``."""
        return self.transform @ np.asarray(values, dtype="float64")

    def covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Map a covariance into the target basis: ``T C T^T``."""
        t = self.transform
        return t @ np.asarray(covariance, dtype="float64") @ t.T

    def weight(self, weight_matrix: np.ndarray) -> np.ndarray:
        """Map a weight matrix into the target basis: ``T^{-T} W T^{-1}``.

        This is the transform a caller is most likely to get wrong by writing
        ``T W T^T``.  The two agree only when :attr:`is_orthogonal` is true.
        """
        t_inv = self._inverse_matrix()
        return t_inv.T @ np.asarray(weight_matrix, dtype="float64") @ t_inv


def block_diagonal_matrix(blocks: list[np.ndarray]) -> np.ndarray:
    """Assemble 2x2 blocks into one block-diagonal ``(2n, 2n)`` matrix."""
    n = len(blocks)
    out = np.zeros((2 * n, 2 * n))
    for i, block in enumerate(blocks):
        out[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
    return out


def identity_basis(size: int, name: str = "canonical") -> BasisTransform:
    """Convenience wrapper for :meth:`BasisTransform.identity`."""
    return BasisTransform.identity(size, name=name)


def scan_basis(
    psi: float | np.ndarray,
    dec: float | np.ndarray | None = None,
) -> BasisTransform:
    """Convenience wrapper for :meth:`BasisTransform.scan`."""
    return BasisTransform.scan(psi, dec=dec)


def block_diagonal_basis(blocks: list[np.ndarray], name: str = "block_diagonal") -> BasisTransform:
    """Build a :class:`BasisTransform` from explicit 2x2 blocks.

    Parameters
    ----------
    blocks : list[np.ndarray]
        One ``2x2`` matrix per observation.
    name : str
        Label for the target frame.

    Returns
    -------
    BasisTransform
        The block-diagonal transform over all observations.
    """
    return BasisTransform(block_diagonal_matrix(blocks), name=name)
