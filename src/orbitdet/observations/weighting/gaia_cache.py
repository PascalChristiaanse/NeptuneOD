"""Cache for Gaia archive weights and covariance components.

Two related caches live here:

* :func:`store_gaia_weights` stores the **diagonal of the full Gaia archive
  weight matrix** computed by the tudatpy ``GaiaAstrometry`` interface (via
  :class:`~orbitdet.data.gaia.GaiaQuery`), so that weighting strategies (e.g.
  ``gaia_archive``) can retrieve it later.  The full (off-diagonal) weight
  matrix itself is carried by the observation sets and is therefore *not*
  duplicated here; this cache only holds the per-observation diagonal for
  reporting and fallback.

* :func:`store_gaia_covariance_components` stores the **random and systematic
  covariance blocks separately** (per transit).  The full archive matrix that
  the observation sets carry is the sum of the two, and a sum cannot be
  decomposed again, so a strategy that wants to re-weight the two components
  independently (``gaia_components``) needs them kept apart.  Like the diagonal
  cache this is keyed by set identifier.

The caches are populated by the Gaia observation factory and by
:meth:`orbitdet.data.gaia.GaiaQuery.to_tudat`.
"""

from __future__ import annotations

import numpy as np

# Maps set_id -> flat weight array of shape (2 * n_obs,)
# with RA at even indices and DEC at odd indices.
_GAIA_WEIGHT_CACHE: dict[str, np.ndarray] = {}

# Maps set_id -> (random_blocks, systematic_blocks), each a list of per-transit
# covariance matrices in observation order.
_GAIA_COMPONENT_CACHE: dict[str, tuple[list, list]] = {}


def store_gaia_weights(set_id: str, weights: np.ndarray) -> None:
    """Store Gaia archive weights for a given set identifier.

    Parameters
    ----------
    set_id : str
        The observation set identifier (e.g. ``"Gaia"``).
    weights : np.ndarray
        Flat weight array of shape ``(2 * n_obs,)`` with RA at even indices
        and DEC at odd indices.
    """
    _GAIA_WEIGHT_CACHE[set_id] = weights.copy()


def get_gaia_weights(set_id: str) -> np.ndarray | None:
    """Retrieve cached Gaia archive weights for a set identifier.

    Parameters
    ----------
    set_id : str
        The observation set identifier.

    Returns
    -------
    np.ndarray or None
        The cached weight array, or ``None`` if no weights are cached for
        this set.
    """
    return _GAIA_WEIGHT_CACHE.get(set_id)


def store_gaia_covariance_components(
    set_id: str, random_blocks: list, systematic_blocks: list
) -> None:
    """Store the random and systematic Gaia covariance blocks for a set.

    Parameters
    ----------
    set_id : str
        The observation set identifier (e.g. ``"Gaia"``).
    random_blocks : list
        Per-transit random covariance matrices, in observation order.
    systematic_blocks : list
        Per-transit systematic covariance matrices, in observation order.
    """
    _GAIA_COMPONENT_CACHE[set_id] = (
        [np.array(block, dtype=float, copy=True) for block in random_blocks],
        [np.array(block, dtype=float, copy=True) for block in systematic_blocks],
    )


def get_gaia_covariance_components(set_id: str) -> tuple[list, list] | None:
    """Retrieve cached Gaia random/systematic covariance blocks.

    Parameters
    ----------
    set_id : str
        The observation set identifier.

    Returns
    -------
    tuple[list, list] or None
        ``(random_blocks, systematic_blocks)`` if cached, else ``None``.
    """
    return _GAIA_COMPONENT_CACHE.get(set_id)


def clear_gaia_weights() -> None:
    """Clear all cached Gaia archive weights and covariance components."""
    _GAIA_WEIGHT_CACHE.clear()
    _GAIA_COMPONENT_CACHE.clear()
