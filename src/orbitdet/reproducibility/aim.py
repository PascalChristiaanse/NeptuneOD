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

    from orbitdet.reproducibility.logging.config import LoggingSettings

    settings = LoggingSettings.from_config(cfg)

    # Use the actual Hydra job name (script name) as the experiment name
    experiment_name = HydraConfig.get().job.name
    run = Run(
        repo=AIM_REPO_DIR,
        experiment=experiment_name,
        # Terminal output reaches Aim through the AimLogHandler in the logging
        # pipeline. Aim's own capture monkey-patches sys.stdout/sys.stderr at
        # class level, which duplicated every line and conflicted with the other
        # capture mechanisms, so it stays off.
        capture_terminal_logs=settings.aim.capture_terminal_logs,
    )

    # Name the run after the script + git info for easy identification
    run.name = f"{experiment_name} ({git_commit})"

    # For sweep jobs, describe the parameter combination in the run description
    # so runs are identifiable in the Aim UI without opening the hparams tab.
    description = _sweep_description()
    if description:
        try:
            run.description = description
        except Exception:
            logger.warning("Failed to set Aim run description", exc_info=True)

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

    # Record Hydra/submitit job identity so a run can be traced back to the
    # exact job and parameter combination that produced it. `hydra.job.id` is
    # populated by the submitit launcher only; it is absent otherwise.
    job_identity = _job_identity()
    for key, value in job_identity.items():
        run[key] = value

    # Store the run hash inside the Hydra output dir so results on disk can
    # be linked back to the Aim run.
    _hash_path = Path(output_dir)
    _hash_path.mkdir(parents=True, exist_ok=True)
    (_hash_path / ".aim_run_hash").write_text(run.hash)

    return run


def _job_identity() -> dict:
    """Return the current Hydra job's identity for Aim tracking.

    Returns a dict with the keys that could be resolved. Never raises: job
    identity is useful metadata, not something that should break a run.

    Returns:
        Mapping of Aim parameter name to value, e.g.
        ``{"job_name": "solve_least_squares", "job_num": 3,
        "job_id": "12345", "override_dirname": "parameters=..."}``.
    """
    from hydra.core.hydra_config import HydraConfig

    try:
        job = HydraConfig.get().job
    except Exception:
        return {}

    # `job.id` exists only under the submitit launcher.
    candidates = {
        "job_name": _select_field(job, "name"),
        "job_num": _select_field(job, "num"),
        "job_id": _select_field(job, "id"),
        "override_dirname": _select_field(job, "override_dirname"),
    }
    return {key: value for key, value in candidates.items() if value is not None}


def format_sweep_description(
    config_name: str | None,
    overrides: Sequence[tuple[str, str]],
) -> str | None:
    """Format a sweep run description from a config name and its overrides.

    This is the single source of truth for the description format, shared by
    :func:`aim_start_run` (live runs) and ``scripts/backfill_sweep_descriptions.py``
    (retro-fitting descriptions onto existing runs).

    Args:
        config_name: The Hydra config name, e.g. ``"sweep/params_x_data"``.
        overrides: The sweep's task overrides as ``(key, value)`` pairs.

    Returns:
        A one-line description such as
        ``"sweep/params_x_data | parameters=initial_state, data=classical"``,
        or ``None`` when there is nothing to describe.
    """
    if not overrides:
        return None

    override_text = ", ".join(f"{key}={value}" for key, value in overrides)
    if config_name:
        return f"{config_name} | {override_text}"
    return override_text


def _sweep_description() -> str | None:
    """Build an Aim run description from the current Hydra sweep overrides.

    A sweep job is identified by ``hydra.mode == MULTIRUN`` (or, as a fallback,
    by the presence of ``hydra.job.num``, which only launchers populate). The
    description names the sweep config and lists every task override, so a run
    can be identified in the Aim UI without opening its hparams.

    Returns:
        A one-line description such as
        ``"sweep/params_x_data | parameters=initial_state, data=classical"``,
        or ``None`` when the job is not part of a sweep (single runs keep
        whatever description they already have).
    """
    from hydra.core.hydra_config import HydraConfig

    try:
        hydra_cfg = HydraConfig.get()
    except Exception:
        return None

    if not _is_sweep(hydra_cfg):
        return None

    return format_sweep_description(
        _select_field(hydra_cfg, "job.config_name"),
        _sweep_overrides(hydra_cfg),
    )


def _is_sweep(hydra_cfg: object) -> bool:
    """Return whether the current Hydra job belongs to a sweep."""
    mode = _select_field(hydra_cfg, "mode")
    if mode is not None and "MULTIRUN" in str(mode).upper():
        return True
    # Launchers set `hydra.job.num` for sweep jobs only.
    return _select_field(hydra_cfg, "job.num") is not None


def _sweep_overrides(hydra_cfg: object) -> list[tuple[str, str]]:
    """Return the sweep's task overrides as ``(key, value)`` pairs.

    Prefers ``hydra.overrides.task`` (the raw override strings) and falls back
    to splitting ``hydra.job.override_dirname`` with Hydra's configured
    separators.
    """
    task_overrides = _select_field(hydra_cfg, "overrides.task")
    if task_overrides:
        parsed = [_split_override(str(override)) for override in task_overrides]
        parsed = [pair for pair in parsed if pair is not None]
        if parsed:
            return parsed

    dirname = _select_field(hydra_cfg, "job.override_dirname")
    if not dirname:
        return []

    item_sep = _select_field(hydra_cfg, "job.config.override_dirname.item_sep") or ","
    kv_sep = _select_field(hydra_cfg, "job.config.override_dirname.kv_sep") or "="
    parsed = [_split_override(part, kv_sep=kv_sep) for part in str(dirname).split(item_sep)]
    return [pair for pair in parsed if pair is not None]


def _split_override(override: str, kv_sep: str = "=") -> tuple[str, str] | None:
    """Split a Hydra override string into ``(key, value)``.

    Returns ``None`` for overrides without a value separator (e.g. ``+foo``).
    """
    key, sep, value = override.partition(kv_sep)
    if not sep:
        return None
    return key.strip(), value.strip()


def _select_field(node: object, key: str) -> object | None:
    """Read a possibly dotted *key* from a Hydra config node.

    ``hydra.job`` is normally an OmegaConf ``DictConfig``, but it can be a plain
    object in tests or older Hydra versions. Missing keys and unexpected types
    yield ``None`` rather than raising.

    Args:
        node: The config node to read from.
        key: The field name, optionally dotted (e.g. ``"job.override_dirname"``).

    Returns:
        The field value, or ``None`` when absent or unreadable.
    """
    try:
        value = OmegaConf.select(node, key, default=None)
    except Exception:
        value = None
    if value is not None:
        return value

    # Fall back to attribute traversal for plain objects (tests, older Hydra).
    current: object | None = node
    for part in key.split("."):
        if current is None:
            return None
        try:
            current = getattr(current, part, None)
        except Exception:
            return None
    return current


def aim_add_tag(run: Run | None, tag: str) -> None:
    """Add a tag to the Aim run, if it is active."""
    if run is not None:
        try:
            run.add_tag(tag)
        except Exception:
            logger.warning("Failed to add tag '%s' to Aim run", tag, exc_info=True)


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

    Logs a static **Image** (Images tab) so you can see thumbnails at a glance.

    .. note::
        This deliberately does **not** log an interactive Aim ``Figure``.
        Interactive figures are ingested into the RocksDB-backed Aim repo and
        trigger a runaway compaction / disk-write storm. Logging only the
        static image keeps the Images tab populated without that cost.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | plotly.graph_objects.Figure
        A matplotlib or Plotly figure to track.
    name : str
        Name for the image series.
    step : int, optional
        Global step.
    context : dict, optional
        Optional context dict.
    """
    run = get_aim_run()
    if run is None:
        logger.warning("No active Aim run — cannot log figure")
        return

    # Static image for the Images tab (rasterize via canvas)
    from aim.sdk.objects import Image as AimImage

    try:
        fig.canvas.draw()
    except Exception as exc:
        logger.warning("Failed to render figure canvas for '%s': %s", name, exc)
        return
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


def aim_log_artifact_reference(
    file_path: str | Path,
    artifact_name: str | None = None,
) -> None:
    """Record a reference to a file on disk without ingesting it into Aim.

    Unlike :func:`aim_log_artifact`, this does **not** copy the file into the
    Aim repository's internal storage. Instead it stores the absolute path as a
    run parameter, so you can easily locate the file later without the disk
    cost of fully ingesting large objects (e.g. binary ``.tudat`` files).

    Parameters
    ----------
    file_path : str | Path
        Path to the file to reference.
    artifact_name : str, optional
        Display name inside Aim; defaults to the filename.
    """
    run = get_aim_run()
    if run is None:
        logger.warning("No active Aim run — cannot log artifact reference")
        return

    path = Path(file_path)
    if not path.exists():
        logger.warning("Artifact file not found, skipping reference: %s", path)
        return

    name = artifact_name or path.name
    run[f"artifact_refs/{name}"] = str(path.resolve())
