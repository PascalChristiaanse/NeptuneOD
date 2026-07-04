# src/orbitdet/reproducibility/runtime.py

from __future__ import annotations

import atexit
import errno
import logging
import os
import random
import re
import select
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from orbitdet.reproducibility.aim import aim_finalize, aim_start_run

if TYPE_CHECKING:
    from aim.sdk import Run


@dataclass
class RuntimeContext:
    git_commit: str
    output_dir: Path
    seed: int
    test_mode: bool
    aim_run: Run | None = field(default=None, repr=False)


_CONTEXT: RuntimeContext | None = None
_NATIVE_FD_CAPTURES: tuple[FdCapture, FdCapture] | None = None
_PYTHON_LOG_LINE_PATTERN = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]")


class FdCapture:
    """Captures file descriptor output and logs it via Python logging."""

    def __init__(self, fd: int, logger: logging.Logger, level: int):
        self._fd = fd
        self._logger = logger
        self._level = level
        self._original_fd: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()
        return False

    def _forward_output(self) -> None:
        """Read from the captured fd and log each line."""
        assert self._read_fd is not None
        assert self._original_fd is not None

        encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
        buffer = b""

        while True:
            try:
                ready, _, _ = select.select([self._read_fd], [], [], 0.5)
                if self._read_fd not in ready:
                    continue

                chunk = os.read(self._read_fd, 4096)
                if not chunk:
                    break

                # Write to original fd and buffer for logging
                os.write(self._original_fd, chunk)
                buffer += chunk

                # Log complete lines
                lines = buffer.split(b"\n")
                buffer = lines[-1]  # Keep incomplete line in buffer

                for line in lines[:-1]:
                    text = line.decode(encoding, errors="replace")
                    if text.strip() and not _PYTHON_LOG_LINE_PATTERN.match(text):
                        self._logger.log(self._level, text)

            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise

    def start(self):
        """Start capturing the file descriptor."""
        with self._lock:
            if self._active:
                return self

            self._original_fd = os.dup(self._fd)
            read_fd, write_fd = os.pipe()

            self._read_fd = read_fd
            os.dup2(write_fd, self._fd)
            os.close(write_fd)

            self._thread = threading.Thread(
                target=self._forward_output,
                name=f"FdCapture-{self._fd}",
                daemon=True,
            )
            self._thread.start()
            self._active = True

        return self

    def stop(self):
        """Stop capturing the file descriptor."""
        thread: threading.Thread | None = None

        with self._lock:
            if not self._active:
                return self

            assert self._original_fd is not None
            os.dup2(self._original_fd, self._fd)

            thread = self._thread
            self._thread = None
            self._active = False

        if thread is not None:
            thread.join()

        with self._lock:
            if self._original_fd is not None:
                os.close(self._original_fd)
                self._original_fd = None

        return self


def _start_native_fd_capture(logger: logging.Logger) -> None:
    """Start capturing stdout and stderr at the file descriptor level."""
    global _NATIVE_FD_CAPTURES

    if _NATIVE_FD_CAPTURES is not None:
        return

    stdout_capture = FdCapture(1, logger, logging.INFO).start()
    stderr_capture = FdCapture(2, logger, logging.WARNING).start()
    _NATIVE_FD_CAPTURES = (stdout_capture, stderr_capture)


def _stop_native_fd_capture() -> None:
    global _NATIVE_FD_CAPTURES

    if _NATIVE_FD_CAPTURES is None:
        return

    for capture in reversed(_NATIVE_FD_CAPTURES):
        capture.stop()

    _NATIVE_FD_CAPTURES = None


atexit.register(_stop_native_fd_capture)


def setup_logging(cfg: DictConfig):
    if OmegaConf.select(cfg, "logging") is None:
        return

    logging.basicConfig(
        level=cfg.logging.level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("tudatpy").setLevel(cfg.logging.tudatpy_logging_level)

    for name in cfg.logging.muted_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    # sys.stderr = _LoggerStderr(logger, sys.stderr)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception


def get_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def git_is_dirty() -> bool:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()

    return len(status) > 0


def assert_clean_repo() -> None:
    if git_is_dirty():
        raise RuntimeError(
            "Repository has uncommitted changes. "
            "Commit or stash changes before running experiments."
        )


def save_conda_environment(output_dir: Path) -> None:
    """Export conda environment to YAML file, with fallback if conda is not in PATH."""
    conda_path = shutil.which("conda")

    if conda_path:
        try:
            # Use the found conda executable directly.
            environment = subprocess.check_output(
                [conda_path, "env", "export"],
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            # If conda is available but export fails, create a minimal environment file.
            logging.warning(
                f"Failed to export conda environment: {e}. "
                "Creating minimal environment file from Python metadata."
            )
            environment = (
                f"# Conda environment export failed. Using Python metadata from: {sys.prefix}\n"
            )
            environment += f"# Python version: {sys.version}\n"
            environment += "# Install from environment.yml in the repository root.\n"
    else:
        # If conda is not available, create a minimal environment file quietly.
        environment = (
            "# Conda environment export skipped because conda was not found in PATH. "
            f"Using Python metadata from: {sys.prefix}\n"
        )
        environment += f"# Python version: {sys.version}\n"
        environment += "# Install from environment.yml in the repository root.\n"

    with open(output_dir / "conda_environment.yaml", "w") as f:
        f.write(environment)


def save_metadata(
    output_dir: Path,
    git_commit: str,
    seed: int,
    test_mode: bool,
) -> None:
    metadata = {
        "git_commit": git_commit,
        "seed": seed,
        "test_mode": test_mode,
    }

    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f)


def save_config(
    output_dir: Path,
    cfg: DictConfig,
) -> None:
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def initialize(cfg: DictConfig) -> RuntimeContext:

    global _CONTEXT

    if _CONTEXT is not None:
        return _CONTEXT

    # assert_clean_repo()

    git_commit = get_git_commit()

    output_dir = Path(HydraConfig.get().runtime.output_dir)

    seed = int(cfg.seed)

    set_random_seed(seed)

    save_config(output_dir, cfg)

    save_metadata(
        output_dir=output_dir,
        git_commit=git_commit,
        seed=seed,
        test_mode=False,
    )

    save_conda_environment(output_dir)

    # Start an Aim run for experiment tracking
    aim_run = aim_start_run(cfg, git_commit, output_dir, seed)

    _CONTEXT = RuntimeContext(
        git_commit=git_commit,
        output_dir=output_dir,
        seed=seed,
        test_mode=False,
        aim_run=aim_run,
    )

    setup_logging(cfg)
    logger = logging.getLogger("FDCapture")
    _start_native_fd_capture(logger)

    OmegaConf.set_readonly(cfg, True)

    return _CONTEXT


def initialize_test_mode(
    seed: int = 0,
) -> RuntimeContext:
    global _CONTEXT

    if _CONTEXT is not None:
        return _CONTEXT

    output_dir = Path("test_outputs")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(seed)

    _CONTEXT = RuntimeContext(
        git_commit="TEST",
        output_dir=output_dir,
        seed=seed,
        test_mode=True,
    )

    return _CONTEXT


def get_context() -> RuntimeContext:
    if _CONTEXT is None:
        raise RuntimeError(
            "Reproducibility system not initialized. "
            "Call initialize() or initialize_test_mode() first."
        )

    return _CONTEXT


def require_initialized() -> None:
    get_context()


def enforce_initialization(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception:
            logging.getLogger(func.__module__).exception("Uncaught exception")
            raise
        finally:
            # Finalize the Aim run when the experiment finishes (success or failure)
            ctx = _CONTEXT
            if ctx is not None and ctx.aim_run is not None:
                aim_finalize(ctx.aim_run)

        if _CONTEXT is None:
            raise RuntimeError(
                "Reproducibility system was not initialized in this run. "
                "Call initialize(cfg) or initialize_test_mode() in your Hydra main function."
            )

        return result

    return wrapper
