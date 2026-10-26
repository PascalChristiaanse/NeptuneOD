"""Shared setup for pipeline stages that need SPICE.

Every Snakemake rule runs in its own process, so the SPICE kernel pool is empty
at the start of each one. ``KernelManager.furnish()`` must therefore be called
in every stage that touches ephemerides — not just in the kernels rule.

The kernels rule still exists: it owns the *download* (network I/O, slow,
failure-prone) and records a stamp. Furnishing is cheap and idempotent, so the
downstream stages just redo it.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from orbitdet.data import KernelManager

logger = logging.getLogger(__name__)


def furnish_kernels(cfg: DictConfig) -> KernelManager:
    """Load all kernels declared by *cfg* into the SPICE pool.

    Assumes the files are already on disk (the ``kernels`` rule downloads them).
    Returns the manager so callers can inspect the pool if needed.
    """
    km = KernelManager(cfg)
    km.furnish()
    logger.info("SPICE kernel pool furnished.")
    return km
