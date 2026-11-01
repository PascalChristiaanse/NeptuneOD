# src/orbitdet/reproducibility/logging/handlers.py
"""Custom logging handlers and filters.

Kept deliberately small: the pipeline reuses stdlib handlers wherever possible
and only adds what the stdout/stderr split requires.
"""

from __future__ import annotations

import logging


def coerce_level(value: object) -> int:
    """Return a numeric logging level from a name or number.

    Accepts ``"INFO"``, ``"info"``, ``20`` or anything ``logging`` understands,
    falling back to :data:`logging.INFO` for unknown input so a typo in a config
    file cannot silently disable logging.

    Args:
        value: A level name (case-insensitive) or numeric level.

    Returns:
        The numeric level.
    """
    if isinstance(value, bool):  # bool is an int subclass; not a valid level
        return logging.INFO
    if isinstance(value, int):
        return value

    name = str(value).strip().upper()
    return logging.getLevelNamesMapping().get(name, logging.INFO)


class MaxLevelFilter(logging.Filter):
    """Allow only records whose level is *below* a threshold.

    Used to make the console streams disjoint: the stdout handler accepts
    ``INFO`` and below, while the stderr handler owns ``WARNING`` and above. This
    is what makes a submitit/Slurm ``.err`` file contain errors rather than the
    whole run log.

    Args:
        level: Exclusive upper bound. Records with ``levelno >= level`` are
            rejected.
    """

    def __init__(self, level: str | int = logging.WARNING) -> None:
        super().__init__()
        self._level = coerce_level(level)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self._level
