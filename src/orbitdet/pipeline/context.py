"""Explicit runtime context for pipeline stages.

``orbitdet.reproducibility.initialize`` builds a ``RuntimeContext`` but sources
its output directory from ``HydraConfig``. Pipeline stages need neither: the
caller passes the output directory, so the DAG owns every path.

The returned object is the *same* ``RuntimeContext`` type, so every existing
``get_environment(cfg, ctx)`` / ``get_propagator_settings(cfg, ctx, ...)`` call
site keeps working unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch_time_object

from orbitdet.reproducibility import RuntimeContext
from orbitdet.reproducibility.runtime import get_git_commit, setup_logging

logger = logging.getLogger(__name__)


def build_context(cfg: DictConfig, output_dir: str | Path) -> RuntimeContext:
    """Construct a ``RuntimeContext`` without touching Hydra's global state.

    Parameters
    ----------
    cfg:
        Resolved configuration from :func:`orbitdet.pipeline.config.load_config`.
    output_dir:
        Directory this stage writes into. Created if missing.

    Returns
    -------
    RuntimeContext
        With ``start_epoch``, ``end_epoch`` and ``initial_epoch`` populated from
        the config, which is what the simulation builders read.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg)

    ctx = RuntimeContext(
        git_commit=get_git_commit(),
        output_dir=output_dir,
        seed=int(cfg.seed),
        test_mode=False,
    )

    # The simulation builders read these three attributes off the context.
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)

    logger.info(
        "Context ready: output_dir=%s, arc=%s -> %s",
        output_dir,
        cfg.start_date,
        cfg.end_date,
    )
    return ctx
