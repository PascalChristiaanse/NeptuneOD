# src/orbitdet/reproducibility/logging/handlers.py
"""Custom logging handlers and filters.

Kept deliberately small: the pipeline reuses stdlib handlers wherever possible
and only adds what the stdout/stderr split and (later) Aim integration require.
"""

from __future__ import annotations

import collections
import logging
import sys
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aim.sdk import Run


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


class AimLogHandler(logging.Handler):
    """Forward log records to an AimStack run.

    Records are written as Aim ``LogRecord`` objects on the ``__log_records``
    sequence, which is the same sequence Aim's own ``Run.log_info`` /
    ``log_warning`` / ``log_error`` helpers use. Because it is a normal logging
    handler it runs on the :class:`~logging.handlers.QueueListener` thread, so a
    slow or unreachable Aim server can never block application code.

    Two deliberate choices:

    * Records are tracked explicitly with a ``LogRecord`` built from the stdlib
      record rather than via ``Run.log_warning()``. Aim's built-in helpers
      capture the *caller* frame for their ``logger_info``, which would record
      this handler's location for every message; passing the stdlib record's
      ``pathname``/``lineno`` preserves the real call site.
    * Errors are counted and logged to ``sys.stderr`` at most once per
      ``max_warnings`` failures instead of being raised. A logging handler must
      never break the application, and a repeated failure (such as a stopped Aim
      server) must not flood the console.

    The run may be attached after construction with :meth:`attach_run`; records
    arriving before that are dropped. This lets the pipeline be configured
    before the Aim run exists.

    Args:
        run: The Aim run to write to, or ``None`` to start detached.
        level: Minimum level to forward.
        max_warnings: How many handler failures to report before going quiet.
    """

    #: Aim sequence name; matches Aim's own built-in logging helpers.
    SEQUENCE_NAME = "__log_records"

    def __init__(
        self,
        run: Run | None = None,
        level: str | int = logging.INFO,
        max_warnings: int = 3,
    ) -> None:
        super().__init__(level=coerce_level(level))
        self._run: Run | None = run
        self._lock = threading.Lock()
        self._failure_count = 0
        self._max_warnings = max_warnings
        # Bounded so a long-repeated message cannot grow memory without limit.
        self._warned_messages: collections.deque[str] = collections.deque(maxlen=100)

    def attach_run(self, run: Run | None) -> None:
        """Point the handler at *run*, or detach when ``None``.

        Args:
            run: The Aim run to write to, or ``None`` to stop writing.
        """
        with self._lock:
            self._run = run

    @property
    def run(self) -> Run | None:
        """The currently attached Aim run, if any."""
        return self._run

    @property
    def failure_count(self) -> int:
        """Number of records that could not be written to Aim."""
        return self._failure_count

    def emit(self, record: logging.LogRecord) -> None:
        """Write *record* to the Aim run, tolerating any failure."""
        run = self._run
        if run is None:
            return

        # The logging module can pass a bare message as a string when a handler
        # is used directly (not via a logger). format() normalises that.
        try:
            message = record.getMessage()
        except Exception:
            message = str(getattr(record, "msg", record))

        try:
            from aim.sdk.logging import LogRecord as AimLogRecord

            aim_record = AimLogRecord(
                message,
                record.levelno,
                timestamp=record.created,
                logger_info=(record.pathname, record.lineno),
            )
            run.track(aim_record, name=self.SEQUENCE_NAME)
        except Exception as exc:
            self._record_failure(record, exc)

    def _record_failure(self, record: logging.LogRecord, exc: BaseException) -> None:
        """Count a failure and warn about it a bounded number of times."""
        with self._lock:
            self._failure_count += 1
            count = self._failure_count

        if count > self._max_warnings:
            return

        # Signature is level plus logger name, so one broken logger does not
        # hide failures from another.
        signature = f"{record.levelno}:{record.name}"
        with self._lock:
            if signature in self._warned_messages:
                return
            self._warned_messages.append(signature)

        print(
            f"[logging] AimLogHandler failed to log a record from "
            f"'{record.name}' ({type(exc).__name__}: {exc}); suppressing further "
            f"warnings after {self._max_warnings}.",
            file=sys.stderr,
        )
