# src/orbitdet/reproducibility/runtime.py

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from orbitdet.reproducibility.aim import aim_add_tag, aim_finalize, aim_start_run

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


def setup_logging(cfg: DictConfig):
    """Install the repository logging pipeline and the exception hook.

    Delegates to :func:`orbitdet.reproducibility.logging.configure_logging`,
    which builds the stdout/stderr/file sinks behind a non-blocking queue. The
    previous ``logging.basicConfig`` call was silently a no-op because Hydra had
    already attached root handlers, so ``cfg.logging.level`` had no effect.

    Args:
        cfg: The resolved Hydra configuration. The ``logging`` section is
            optional; defaults are used when it is absent.
    """
    from orbitdet.reproducibility.logging import configure_logging

    configure_logging(cfg)

    logger = logging.getLogger(__name__)

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

    # Logging is (re)configured on every call, before the context cache check.
    # Under a Hydra launcher that reuses the interpreter for several jobs, each
    # job has its own run directory; configuring logging after the early return
    # meant every job after the first wrote its logs into the first job's files
    # (or produced no log file at all). configure_logging() is idempotent while
    # the run directory is unchanged and reconfigures when it changes.
    setup_logging(cfg)

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

    # Point the logging pipeline's Aim sink at the new run. Logging is
    # configured before the run exists, so the handler starts detached.
    from orbitdet.reproducibility.logging import attach_aim_run

    attach_aim_run(aim_run)

    _CONTEXT = RuntimeContext(
        git_commit=git_commit,
        output_dir=output_dir,
        seed=seed,
        test_mode=False,
        aim_run=aim_run,
    )

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

    # Install the same logging pipeline as production (console split, no file,
    # no Aim) so tests exercise the real handler wiring instead of a separate
    # basicConfig path. run_dir=None keeps it console-only.
    from orbitdet.reproducibility.logging import configure_logging

    configure_logging(None, run_dir=None, use_queue=True)

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
        outcome: str | None = None
        result = None
        try:
            result = func(*args, **kwargs)
            outcome = "completed"
        except KeyboardInterrupt:
            outcome = "cancelled"
            logging.getLogger(func.__module__).warning("Run cancelled by user")
            raise
        except Exception:
            outcome = "crashed"
            logging.getLogger(func.__module__).exception("Uncaught exception")
            raise
        finally:
            # NB: resolve the module through sys.modules rather than the
            # closure's globals. Under the submitit launcher the decorated
            # function is pickled into the child process, which can leave the
            # wrapper's __globals__ pointing at a deserialized *copy* of this
            # module; reading _CONTEXT through that copy would always see None
            # and the completed/crashed tag would be silently lost.
            ctx = sys.modules[__name__]._CONTEXT  # type: ignore[attr-defined]
            if ctx is not None and ctx.aim_run is not None and outcome is not None:
                aim_add_tag(ctx.aim_run, outcome)
                aim_finalize(ctx.aim_run)

        if sys.modules[__name__]._CONTEXT is None:  # noqa: SLF001
            raise RuntimeError(
                "Reproducibility system was not initialized in this run. "
                "Call initialize(cfg) or initialize_test_mode() in your Hydra main function."
            )

        return result

    return wrapper
