# src/orbitdet/reproducibility/logging/__init__.py
"""The repository's single logging pipeline.

Call :func:`configure_logging` once, from the Hydra entrypoint, before doing any
work. It installs four sinks - stdout, stderr, a rotating run log, and an
ERROR-only log - behind one non-blocking queue, and it is idempotent so repeated
calls (e.g. across Hydra multirun jobs) are harmless.

Example:
    >>> from orbitdet.reproducibility.logging import configure_logging
    >>> configure_logging(cfg, run_dir=Path("results/..."))
"""

from __future__ import annotations

import atexit
import logging
import logging.config
import logging.handlers
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from orbitdet.reproducibility.logging.config import (
    AIM_HANDLER_NAME,
    STDERR_HANDLER_NAME,
    LoggingSettings,
    build_sink_payload,
)
from orbitdet.reproducibility.logging.handlers import AimLogHandler
from orbitdet.reproducibility.logging.listener import (
    DEFAULT_JOIN_TIMEOUT,
    BoundedQueueListener,
)

if TYPE_CHECKING:
    from aim.sdk import Run

__all__ = [
    "LoggingSettings",
    "BoundedQueueListener",
    "AimLogHandler",
    "configure_logging",
    "shutdown_logging",
    "attach_aim_run",
    "get_aim_handler",
    "get_logger",
    "get_listener",
    "is_configured",
]

_lock = threading.Lock()
_console_listener: BoundedQueueListener | None = None
_io_listener: BoundedQueueListener | None = None
_aim_handler: AimLogHandler | None = None
_configured = False
_configured_run_dir: Path | None = None


def is_configured() -> bool:
    """Whether the pipeline is currently installed."""
    return _configured


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Thin wrapper over :func:`logging.getLogger`; present so call sites have a
    single documented entry point that does not depend on the logging layout.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        The logger.
    """
    return logging.getLogger(name)


def get_listener() -> tuple[BoundedQueueListener | None, BoundedQueueListener | None]:
    """Return the active (console, io) listeners, or ``(None, None)``."""
    return _console_listener, _io_listener


def get_aim_handler() -> AimLogHandler | None:
    """Return the active Aim handler, or ``None`` when not enabled."""
    return _aim_handler


def attach_aim_run(run: Run | None) -> None:
    """Attach (or detach) the Aim run used by the Aim logging sink.

    Configuration happens before the Aim run exists, so the pipeline starts with
    a detached Aim handler. Call this once the run has been created.

    Args:
        run: The Aim run to forward records to, or ``None`` to detach.
    """
    if _aim_handler is not None:
        _aim_handler.attach_run(run)


def configure_logging(
    cfg: DictConfig | None = None,
    *,
    run_dir: str | Path | None = None,
    use_queue: bool = True,
    join_timeout: float = DEFAULT_JOIN_TIMEOUT,
) -> None:
    """Install the logging pipeline. Idempotent.

    Removes any pre-existing root handlers (including Hydra's ``job_logging``
    handlers, which route errors to stdout) and installs the configured sinks.

    Args:
        cfg: The Hydra config. When it has a ``logging`` section those settings
            are honoured; otherwise defaults are used.
        run_dir: Directory for the run log. When ``None`` it is taken from
            ``HydraConfig``, and logging falls back to console-only if Hydra is
            not initialised.
        use_queue: When ``True`` (the default), sinks run on a background
            listener thread so no application thread can block on I/O. Set
            ``False`` for deterministic tests.
        join_timeout: Seconds the listener may take to drain at shutdown before
            being abandoned.

    Raises:
        ValueError: If ``use_queue`` is true but the listener cannot be started.
    """
    global _console_listener, _io_listener, _aim_handler, _configured, _configured_run_dir

    settings = LoggingSettings.from_config(cfg)
    resolved_run_dir = _resolve_run_dir(run_dir)

    with _lock:
        already_configured = _configured
        same_run_dir = _configured_run_dir == resolved_run_dir
        _configured = True
        _configured_run_dir = resolved_run_dir

    if already_configured and same_run_dir:
        return

    if already_configured:
        shutdown_logging()

    log_file = resolved_run_dir / settings.filename if resolved_run_dir is not None else None

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    payload, sink_names = build_sink_payload(settings, log_file=log_file)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)

    payload["root"]["handlers"] = list(sink_names)

    logging.config.dictConfig(payload)

    sinks = _resolve_sinks(sink_names)

    for h in sinks:
        root.removeHandler(h)

    if settings.aim.enabled:
        aim_handler = AimLogHandler(
            run=None,
            level=settings.effective_aim_level,
            max_warnings=settings.aim.max_warnings,
        )
        aim_handler.set_name(AIM_HANDLER_NAME)
        sinks.append(aim_handler)
        _aim_handler = aim_handler

    root.setLevel(settings.root_level)

    # Split sinks into console (fast, attach directly even in queued mode)
    # and I/O (file + Aim, run on a background listener thread).
    # Console writes are ~µs and must appear *before* a long C++ call that
    # holds the GIL.  Queuing them would defer the write until the GIL is
    # released (3+ minutes later for tudatpy propagation), making the
    # terminal appear frozen.
    console_sink_names = {"console", STDERR_HANDLER_NAME}
    console_sinks: list[logging.Handler] = []
    io_sinks: list[logging.Handler] = []
    for sink in sinks:
        (console_sinks if sink.name in console_sink_names else io_sinks).append(sink)

    # Attach console sinks directly — they write on the main thread.
    for sink in console_sinks:
        root.addHandler(sink)

    if use_queue and io_sinks:
        io_queue: queue.Queue = queue.Queue()

        io_listener = BoundedQueueListener(
            io_queue, *io_sinks, join_timeout=join_timeout
        )
        io_listener.start()
        _io_listener = io_listener

        root.addHandler(logging.handlers.QueueHandler(io_queue))
    elif io_sinks:
        for sink in io_sinks:
            root.addHandler(sink)


def shutdown_logging() -> None:
    """Drain both queues, detach handlers, and flush.

    Safe to call multiple times and safe to call when never configured. Also
    registered with :mod:`atexit`, so normal interpreter exit flushes logs
    without an explicit call.
    """
    global _console_listener, _io_listener, _aim_handler, _configured, _configured_run_dir

    with _lock:
        io_listener = _io_listener
        aim_handler = _aim_handler
        _console_listener = None
        _io_listener = None
        _aim_handler = None
        _configured = False
        _configured_run_dir = None

    # Stop the I/O listener first (drains the queue so Aim records are not
    # lost). Console sinks run on the main thread; there is nothing to stop.
    if io_listener is not None:
        io_listener.stop()

    if aim_handler is not None:
        aim_handler.attach_run(None)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logging.shutdown()


def _resolve_sinks(sink_names: list[str]) -> list[logging.Handler]:
    """Fetch the sink handlers constructed by ``dictConfig``.

    ``logging.getHandlerByName`` returns handlers even when they are attached to
    no logger, which is exactly the case here.

    Args:
        sink_names: Handler names from ``build_sink_payload``.

    Returns:
        The handler instances in the requested order.

    Raises:
        RuntimeError: If a declared sink handler was not constructed.
    """
    sinks: list[logging.Handler] = []
    for name in sink_names:
        handler = logging.getHandlerByName(name)
        if handler is None:
            raise RuntimeError(f"Logging sink handler '{name}' was not constructed.")
        sinks.append(handler)
    return sinks


def _resolve_run_dir(run_dir: str | Path | None) -> Path | None:
    """Resolve the run directory from an argument or the active Hydra run."""
    if run_dir is not None:
        return Path(run_dir)

    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        # Hydra not initialised (tests, plain scripts): console-only logging.
        return None


atexit.register(shutdown_logging)
