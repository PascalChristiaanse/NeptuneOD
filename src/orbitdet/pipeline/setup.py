"""Shared setup for pipeline stages that need SPICE.

Every Snakemake rule runs in its own process, so the SPICE kernel pool is empty
at the start of each one. ``KernelManager.furnish()`` must therefore be called
in every stage that touches ephemerides -- not just in the kernels rule.

The kernels rule still exists: it owns the *download* (network I/O, slow,
failure-prone) and records a stamp. Furnishing is cheap and idempotent, so the
downstream stages just redo it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig
from tudatpy.estimation import observations as obs

from orbitdet.data import KernelManager
from orbitdet.pipeline.context import build_context
from orbitdet.simulation import get_environment

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


def prepare_environment(
    cfg: DictConfig,
    output_dir: str | Path,
    observations: obs.ObservationCollection,
):
    """Build a ready-to-use system of bodies for a collection-based stage.

    Three steps are always needed together, which is why they live here rather
    than in each stage:

    1. build the runtime context (epochs and seed, needed by the environment);
    2. furnish the SPICE pool, which is empty in every fresh process;
    3. re-add the ground stations the collection references.

    Step 3 is the non-obvious one. Building the observation collection adds
    ground stations to the system of bodies as a *side effect*. A stage that
    rebuilds the environment from config alone therefore lacks them, and
    simulator construction fails with "point is not a ground station".

    Parameters
    ----------
    cfg:
        Resolved configuration.
    output_dir:
        Directory this stage writes into.
    observations:
        The collection whose link ends determine which stations are re-added.

    Returns
    -------
    tuple
        ``(ctx, bodies)``.
    """
    # Imported here to avoid a circular import: observation_models imports from
    # this module's siblings, and setup is imported by nearly every stage.
    from orbitdet.pipeline.observation_models import (
        add_observatories,
        collect_observatory_codes,
    )

    ctx = build_context(cfg, output_dir)
    furnish_kernels(cfg)
    bodies = get_environment(cfg, ctx)
    add_observatories(cfg, bodies, collect_observatory_codes(observations))
    return ctx, bodies
