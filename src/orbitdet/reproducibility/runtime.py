# src/orbitdet/reproducibility/runtime.py

import logging
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@dataclass
class RuntimeContext:
    git_commit: str
    output_dir: Path
    seed: int
    test_mode: bool


_CONTEXT: RuntimeContext | None = None


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

    try:
        if conda_path:
            # Use the found conda executable directly
            environment = subprocess.check_output(
                [conda_path, "env", "export"],
                text=True,
            )
        else:
            # Try using shell=True to access conda through bash initialization
            environment = subprocess.check_output(
                "conda env export",
                shell=True,
                text=True,
            )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        # If conda is still not available, create a minimal environment file from sys.prefix
        logging.warning(
            f"Failed to export conda environment: {e}. "
            "Creating minimal environment file from Python metadata."
        )
        environment = (
            f"# Conda environment export failed. Using Python metadata from: {sys.prefix}\n"
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

    assert_clean_repo()

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

    _CONTEXT = RuntimeContext(
        git_commit=git_commit,
        output_dir=output_dir,
        seed=seed,
        test_mode=False,
    )

    setup_logging(cfg)

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
        result = func(*args, **kwargs)

        if _CONTEXT is None:
            raise RuntimeError(
                "Reproducibility system was not initialized in this run. "
                "Call initialize(cfg) or initialize_test_mode() in your Hydra main function."
            )

        return result

    return wrapper
