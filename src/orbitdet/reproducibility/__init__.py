# src/orbitdet/reproducibility/__init__.py

from orbitdet.reproducibility.aim import (
    aim_log_artifact,
    aim_log_figure,
    aim_log_metrics,
    get_aim_run,
)
from orbitdet.reproducibility.hydra import register_resolvers
from orbitdet.reproducibility.runtime import (
    RuntimeContext,
    enforce_initialization,
    get_context,
    initialize,
    initialize_test_mode,
    require_initialized,
)

register_resolvers()

__all__ = [
    "initialize",
    "initialize_test_mode",
    "get_context",
    "require_initialized",
    "enforce_initialization",
    "RuntimeContext",
    "get_aim_run",
    "aim_log_metrics",
    "aim_log_figure",
    "aim_log_artifact",
]
