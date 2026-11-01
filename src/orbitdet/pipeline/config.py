"""Hydra-as-a-library config loading for pipeline stages.

``@hydra.main`` takes over the process: it creates a timestamped output
directory, changes the working directory, and installs global state that
``HydraConfig.get()`` reads back. That is incompatible with a file-based DAG,
where the caller owns the output paths.

This module uses Hydra's *compose* API instead, which resolves the same
``conf/`` tree into a plain ``DictConfig`` and nothing else.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# ``src/orbitdet/pipeline/config.py`` -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "conf"


def load_config(
    experiment: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Resolve ``conf/experiment/<experiment>.yaml`` into a ``DictConfig``.

    Parameters
    ----------
    experiment:
        Experiment config name, e.g. ``"classic_triton_state"``. Resolved as
        ``conf/experiment/<experiment>.yaml``.
    overrides:
        Hydra-style overrides, e.g. ``["weighting=id_v2", "seed=7"]``.

    Returns
    -------
    DictConfig
        The fully composed configuration. Read-only, matching the behaviour of
        ``orbitdet.reproducibility.initialize``.
    """
    # Importing the package registers the custom resolvers (``git_commit:``,
    # ``now:``) that the conf tree relies on. Without this, compose() fails on
    # any config that interpolates them.
    import orbitdet.reproducibility  # noqa: F401

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name=f"experiment/{experiment}",
            overrides=list(overrides or []),
        )

    OmegaConf.set_readonly(cfg, True)
    return cfg


def load_resolved_config(path: str | Path) -> DictConfig:
    """Load a config previously written by :func:`orbitdet.pipeline.resolve.run_resolve_config`.

    The resolved file has every interpolation already evaluated, so this needs
    no Hydra import and no resolver registration. Stages use this rather than
    :func:`load_config` so that the config they read is the exact artifact
    Snakemake tracked as an input.
    """
    cfg = OmegaConf.load(str(path))
    OmegaConf.set_readonly(cfg, True)
    return cfg
