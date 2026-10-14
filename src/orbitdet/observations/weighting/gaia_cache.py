"""Cache for Gaia archive weights.

Stores the diagonal of the full Gaia archive weight matrix computed by the
tudatpy ``GaiaAstrometry`` interface (via :class:`~orbitdet.data.gaia.GaiaQuery`)
so that weighting strategies (e.g. ``gaia_archive``) can retrieve it later.

The full (off-diagonal) weight matrix itself is carried by the observation sets
and is therefore *not* duplicated here; this cache only holds the per-observation
diagonal for reporting and fallback.

The cache is populated by :meth:`orbitdet.data.gaia.GaiaQuery.to_tudat` and
consumed by the :class:`GaiaArchiveWeight` strategy.
"""

from __future__ import annotations

import numpy as np

# Maps set_id -> flat weight array of shape (2 * n_obs,)
# with RA at even indices and DEC at odd indices.
_GAIA_WEIGHT_CACHE: dict[str, np.ndarray] = {}


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


def clear_gaia_weights() -> None:
    """Clear all cached Gaia archive weights."""
    _GAIA_WEIGHT_CACHE.clear()