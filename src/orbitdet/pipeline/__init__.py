"""File-based pipeline stages for Snakemake-driven execution.

Each stage is a pure function of ``(cfg, ctx, input paths) -> output paths``.
Stages never read ``HydraConfig`` and never call ``plt.show()``; the caller
(Snakemake, or a test) owns every path.

This is deliberately separate from ``scripts/solve_least_squares.py``, which
remains the single-process interactive entry point.
"""

from orbitdet.pipeline.config import load_config
from orbitdet.pipeline.context import build_context

__all__ = ["load_config", "build_context"]
