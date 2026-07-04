# src/orbitdet/reproducibility/aim.py
#
# AimStack experiment-tracking integration.
# All Aim-specific functions live here, keeping the rest of the
# reproducibility module free of a hard dependency on ``aim``.

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from aim import Run
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from matplotlib.figure import Figure


logger = logging.getLogger(__name__)

# Absolute path to the Aim repository (``results/.aim/``).
AIM_REPO_DIR = str((Path(__file__).resolve().parents[3] / "results").absolute())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_omegaconf(
    cfg: DictConfig,
    prefix: str = "",
    sep: str = ".",
) -> dict:
    """Recursively flatten an OmegaConf DictConfig into a flat key-value dict.

    Nested keys are joined with *sep*.  Sequences are converted to comma-separated
    strings so they display nicely in the Aim UI.
    """
    result: dict = {}
    for key, value in cfg.items():
        flat_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, DictConfig):
            result.update(_flatten_omegaconf(value, prefix=flat_key, sep=sep))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result[flat_key] = ",".join(str(v) for v in value)
        else:
            result[flat_key] = value
    return result


# ---------------------------------------------------------------------------
# Public API – called from runtime.py / user scripts
# ---------------------------------------------------------------------------


def aim_start_run(
    cfg: DictConfig,
    git_commit: str,
    output_dir: Path,
    seed: int,
) -> Run:
    """Start a new Aim run and log the full Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        The resolved Hydra configuration.
    git_commit : str
        Short git commit hash (may include ``_dirty`` suffix).
    output_dir : Path
        Absolute path to the Hydra output directory for this run.
    seed : int
        Random seed used for the experiment.

    Returns
    -------
    aim.sdk.Run
        The active Aim run instance.
    """
    from aim.sdk import Run
    from hydra.core.hydra_config import HydraConfig

    # Use the actual Hydra job name (script name) as the experiment name
    experiment_name = HydraConfig.get().job.name
    run = Run(
        repo=AIM_REPO_DIR,
        experiment=experiment_name,
        capture_terminal_logs=True,
    )

    # Name the run after the script + git info for easy identification
    run.name = f"{experiment_name} ({git_commit})"

    # Tag the run
    tags = []
    if "dirty" in git_commit:
        tags.append("dirty")
    if OmegaConf.select(cfg, "test_mode", default=False):
        tags.append("test")
    for tag in tags:
        run.add_tag(tag)

    # Set local artifacts_uri so log_artifact works
    run.set_artifacts_uri("file://" + AIM_REPO_DIR)

    # Log the full Hydra config as a flat dict of parameters
    flat_cfg = _flatten_omegaconf(cfg)
    run["hparams"] = flat_cfg

    # Also log important top-level context separately
    run["git_commit"] = git_commit
    run["seed"] = seed
    run["output_dir"] = str(output_dir)

    # Store the run hash inside the Hydra output dir so results on disk can
    # be linked back to the Aim run.
    _hash_path = Path(output_dir)
    _hash_path.mkdir(parents=True, exist_ok=True)
    (_hash_path / ".aim_run_hash").write_text(run.hash)

    return run


def aim_finalize(run: Run | None) -> None:
    """Close the Aim run, flushing all pending data."""
    if run is not None:
        try:
            run.close()
        except Exception:
            logger.warning("Failed to finalize Aim run", exc_info=True)


def get_aim_run() -> Run | None:
    """Return the current Aim run from the active :class:`RuntimeContext`, if any."""
    # Late import to avoid circular dependency
    from orbitdet.reproducibility.runtime import _CONTEXT

    if _CONTEXT is not None:
        return _CONTEXT.aim_run
    return None


def aim_log_metrics(
    metrics: dict[str, float],
    step: int | None = None,
    context: dict | None = None,
) -> None:
    """Log scalar metrics to the current Aim run.

    Parameters
    ----------
    metrics : dict[str, float]
        Metric name → value mapping, e.g. ``{"loss": 0.12, "accuracy": 0.98}``.
    step : int, optional
        Global step / iteration number.  Aim auto-increments if omitted.
    context : dict, optional
        Optional context dict for grouping (e.g. ``{"subset": "train"}``).
    """
    run = get_aim_run()
    if run is None:
        logger.warning("No active Aim run — cannot log metrics")
        return
    for name, value in metrics.items():
        run.track(value, name=name, step=step, context=context)


def aim_log_figure(
    fig: Figure,
    name: str = "figure",
    step: int | None = None,
    context: dict | None = None,
) -> None:
    """Log a matplotlib figure to the current Aim run.

    Logs both an interactive **Figure** (Figures tab) and a static **Image**
    (Images tab) so you can explore interactively *and* see thumbnails at a
    glance.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | plotly.graph_objects.Figure
        A matplotlib or Plotly figure to track.
    name : str
        Name for the figure/image series.
    step : int, optional
        Global step.
    context : dict, optional
        Optional context dict.
    """
    run = get_aim_run()
    if run is None:
        logger.warning("No active Aim run — cannot log figure")
        return

    # Interactive figure for the Figures tab
    from aim import Figure as AimFigure

    run.track(AimFigure(fig), name=name, step=step, context=context)

    # Static image for the Images tab (rasterize via canvas)
    from aim.sdk.objects import Image as AimImage

    fig.canvas.draw()
    run.track(AimImage(fig), name=f"{name}_static", step=step, context=context)


def aim_log_artifact(
    file_path: str | Path,
    artifact_name: str | None = None,
) -> None:
    """Attach an artifact (any file) to the current Aim run.

    The file is copied into the Aim repository's internal storage so it
    stays linked to the run even if the original file is moved.

    Parameters
    ----------
    file_path : str | Path
        Path to the file to attach.
    artifact_name : str, optional
        Display name inside Aim; defaults to the filename.
    """
    run = get_aim_run()
    if run is None:
        logger.warning("No active Aim run — cannot log artifact")
        return

    path = Path(file_path)
    if not path.exists():
        logger.warning("Artifact file not found, skipping: %s", path)
        return

    name = artifact_name or path.name

    # Ensure artifacts_uri is set (local directory for this project)
    if run.artifacts_uri is None:
        run.set_artifacts_uri("file://" + AIM_REPO_DIR)

    run.log_artifact(str(path), name)
