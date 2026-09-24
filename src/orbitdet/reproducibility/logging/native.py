# src/orbitdet/reproducibility/logging/native.py
"""Tee-only capture for native (C++) output.

tudatpy writes directly to file descriptors 1/2 from C++, bypassing Python's
``logging`` completely. Two problems follow:

* such output never appears in the run log;
* redirecting it via ``os.dup2`` and re-injecting lines into ``logging`` (the
  old ``FdCapture``) could deadlock the process and caused the HPC freezes
  documented in ``docs/LOGGING_ARCHITECTURE_REPORT.md``.

:class:`FdTee` is the safe replacement: it *tees* the original fd to a file for
the duration of a ``with`` block without ever writing back into the logging
system. It is the only code that touches file descriptors 1/2. Important rules:

* The pipe is non-blocking: a reader that cannot keep up never blocks writers.
* ``__exit__`` / :meth:`close` flush the trailing partial line and wait at most
  ``join_timeout`` seconds for the reader thread.
* Nothing is re-injected into ``logging`` - native output is written to the tee
  file only, and callers attach it to Aim as an artifact reference.
"""

from __future__ import annotations

import errno
import logging
import os
import select
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default seconds to wait for the reader thread to drain during ``close()``.
DEFAULT_JOIN_TIMEOUT = 5.0


class FdTee:
    """Tee file descriptor *fd* to *path* for the duration of a ``with`` block.

    The original file descriptor is duplicated before the block and restored
    afterwards, so both the caller's normal output and the tee file receive the
    bytes. Native (C++) writes that bypass Python's streams are captured too,
    which is the whole point.

    The reader thread uses a non-blocking pipe, so it can never stall the
    writer. If the disk is slower than the writer for a sustained period, some
    intermediate data is dropped (the reader skips ahead) rather than the
    process hanging - matching the "never block the application" policy of the
    rest of the logging pipeline.

    Args:
        fd: File descriptor to tee (1 for stdout, 2 for stderr).
        path: Destination file for the teed bytes.
        join_timeout: Seconds to wait for the reader thread in :meth:`close`
            before abandoning it.
    """

    def __init__(
        self,
        fd: int,
        path: str | Path,
        *,
        join_timeout: float = DEFAULT_JOIN_TIMEOUT,
    ) -> None:
        self._fd = fd
        self._path = Path(path)
        self._join_timeout = join_timeout

        self._original_fd: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._tee_handle = None
        self._lock = threading.Lock()
        self._active = False

    # -- context-manager protocol -------------------------------------------------

    def __enter__(self) -> FdTee:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> bool:
        self.close()
        return False

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> FdTee:
        """Begin teeing. Restoring the original fd is *not* automatic on errors."""
        with self._lock:
            if self._active:
                raise RuntimeError("FdTee is already active")
            self._active = True

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._tee_handle = open(self._path, "w", encoding="utf-8", errors="replace")

        self._original_fd = os.dup(self._fd)
        read_fd, write_fd = os.pipe()
        # Non-blocking so a slow reader can never stall a writer.
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)

        self._read_fd = read_fd
        os.dup2(write_fd, self._fd)
        os.close(write_fd)

        self._thread = threading.Thread(
            target=self._forward_output,
            name=f"FdTee-{self._fd}",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        """Restore the original fd and drain any remaining buffered output.

        Safe to call more than once. Waits at most ``join_timeout`` for the
        reader thread, then abandons it (it is a daemon and cannot outlive the
        process).
        """
        with self._lock:
            if not self._active:
                return
            self._active = False

            assert self._original_fd is not None
            # Stop teeing first, then restore the original descriptor.
            os.dup2(self._original_fd, self._fd)
            os.close(self._original_fd)
            self._original_fd = None

            thread = self._thread
            self._thread = None

        # Give the reader a final chance to drain the pipe (it sees EOF as
        # soon as write ends are closed), bounded by join_timeout.
        if thread is not None:
            thread.join(self._join_timeout)
            if thread.is_alive():
                logger.warning(
                    "FdTee reader thread did not finish within %.1fs; "
                    "abandoning it. Some native output may be missing from %s.",
                    self._join_timeout,
                    self._path,
                    extra={"orbitdet_logging_internal": True},
                )

        if self._tee_handle is not None:
            self._tee_handle.flush()
            self._tee_handle.close()
            self._tee_handle = None

        if self._read_fd is not None:
            if thread is None or not thread.is_alive():
                os.close(self._read_fd)
                self._read_fd = None

    # -- internals ------------------------------------------------------------------

    def _forward_output(self) -> None:
        """Read from the (non-blocking) pipe and append to the tee file."""
        assert self._read_fd is not None
        assert self._tee_handle is not None

        while True:
            try:
                chunk = os.read(self._read_fd, 65536)
                if not chunk:
                    break
                self._tee_handle.write(chunk.decode("utf-8", errors="replace"))
            except BlockingIOError:
                # Nothing available right now; wait briefly for more.
                try:
                    select.select([self._read_fd], [], [], 0.1)
                except (OSError, ValueError):
                    break
                continue
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    continue
                if exc.errno in (errno.EBADF, errno.EINVAL, errno.EIO):
                    break
                raise
            except ValueError:
                # The fd was closed under us during interpreter shutdown.
                break
