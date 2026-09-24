# src/orbitdet/reproducibility/logging/listener.py
"""A non-blocking, bounded logging queue listener.

The stdlib :class:`logging.handlers.QueueListener` drains a queue on a daemon
thread, but its :meth:`~logging.handlers.QueueListener.stop` calls
``thread.join()`` with no timeout. If a sink is wedged (a network share that has
gone away, a full disk, a stopped Aim server) that join blocks forever and takes
interpreter shutdown with it.

:class:`BoundedQueueListener` keeps the stdlib behaviour but adds a timeout so
shutdown can never hang: on timeout the listener is abandoned rather than waited
on. A daemon thread still lets the interpreter exit.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import threading

logger = logging.getLogger(__name__)

#: Default seconds to wait for the listener thread to drain during shutdown.
DEFAULT_JOIN_TIMEOUT = 5.0


class BoundedQueueListener(logging.handlers.QueueListener):
    """A :class:`QueueListener` that cannot hang the process on shutdown.

    Args:
        log_queue: The queue to drain.
        handlers: Sink handlers, in order.
        join_timeout: Maximum seconds to wait when stopping. On timeout the
            thread is abandoned; it is a daemon and will not keep the
            interpreter alive.
        respect_handler_level: Forward to the stdlib base class. Set ``True`` so
            each sink's own level (e.g. WARNING on the stderr handler) is
            honoured.
    """

    def __init__(
        self,
        log_queue: queue.Queue[logging.LogRecord | None],
        *handlers: logging.Handler,
        join_timeout: float = DEFAULT_JOIN_TIMEOUT,
        respect_handler_level: bool = True,
    ) -> None:
        super().__init__(log_queue, *handlers, respect_handler_level=respect_handler_level)
        self._join_timeout = join_timeout
        self._stop_lock = threading.Lock()
        self._stopped = False
        self._drained = False

    def stop(self) -> None:
        """Flush the queue and stop the listener, bounded by ``join_timeout``.

        Safe to call more than once and safe to call when the listener was never
        started.
        """
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

            thread = getattr(self, "_thread", None)
            if thread is None:
                return

            # Ask the thread to finish: the sentinel is processed only after all
            # queued records, so this is also the flush signal.
            self.queue.put_nowait(None)

            thread.join(self._join_timeout)
            self._drained = not thread.is_alive()

            if not self._drained:
                logger.warning(
                    "Log listener did not drain within %.1fs; abandoning it. "
                    "Some log records may be lost.",
                    self._join_timeout,
                    extra={"orbitdet_logging_internal": True},
                )
            self._thread = None

    @property
    def drained(self) -> bool:
        """Whether the queue was fully flushed during the last :meth:`stop`."""
        return self._drained
