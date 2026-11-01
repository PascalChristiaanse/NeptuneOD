"""Stage: write a sweep combination's config slices.

A sweep combination is an experiment plus a set of Hydra overrides. Everything
the base experiment does *before* estimation depends only on the experiment, not
on the overrides -- for ``fit_to_ephemeris`` the overrides only toggle which
parameters are estimated -- so the expensive preprocessing is shared and each
combination only needs its own estimation-facing slices.

That is why this stage writes **estimation and covariance slices only**:

* ``observations``, ``prefit`` and the residual artefacts are identical across
  combinations, so recomputing them per combination would multiply the most
  expensive work by the size of the sweep for no benefit;
* ``estimation`` and ``covariance`` genuinely differ, because they read
  ``estimation.parameters_to_estimate``.

If a future sweep varies something the environment depends on (a dataset, an
integrator), its combinations *must* get their own preprocessing. The guard for
that is :data:`SHARED_STAGES`: any sweep key that could affect a shared stage
raises, rather than silently reusing a result computed with different settings.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from orbitdet.pipeline.config import load_config
from orbitdet.pipeline.resolve import STAGE_KEYS, _slice

logger = logging.getLogger(__name__)

#: Stages each combination writes for itself.
#:
#: Only stages that read sweep-varied keys belong here. Everything else is
#: shared with the base experiment.
PER_COMBO_STAGES = ("estimation", "covariance")

#: Top-level config prefixes that, if varied, would invalidate a shared stage.
#:
#: Guarding on the prefix rather than the full key is deliberate: it is a
#: conservative over-approximation. ``datasets`` appearing anywhere in an
#: override is refused, even though only some dataset fields affect the
#: environment build. Refusing too much is cheap; reusing an observation
#: collection built with different data is not.
_SHARED_STAGE_PREFIXES = (
    "datasets",
    "data",
    "collection",
    "kernels",
    "bodies_to_create",
    "bodies_to_propagate",
    "integrator",
    "propagation",
    "global_frame_origin",
    "global_frame_orientation",
    "start_date",
    "end_date",
    "initial_epoch",
    "seed",
    "outlier_rejection",
    "weighting",
)


def _refuse_shared_stage_overrides(overrides: list[str]) -> None:
    """Raise if any override could change a stage that combinations share."""
    offenders = []
    for override in overrides:
        key = override.partition("=")[0]
        root = key.split(".")[0]
        if root in _SHARED_STAGE_PREFIXES:
            offenders.append(override)

    if offenders:
        raise ValueError(
            "This sweep varies settings that a shared stage reads, so the "
            "combinations cannot reuse the base experiment's preprocessing:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\nEither restrict the sweep to estimation-only settings, or give "
            "each combination its own preprocessing stage. Reusing a shared "
            "artefact here would silently fit different data than the "
            "configuration describes."
        )


def run_resolve_sweep_config(
    experiment: str,
    output_dir: str | Path,
    overrides: list[str],
    only: str | None = None,
) -> list[Path]:
    """Compose one sweep combination and write its per-combination slices.

    Parameters
    ----------
    experiment:
        Base experiment name, e.g. ``"fit_to_ephemeris"``.
    output_dir:
        Directory for this combination's slices.
    overrides:
        Hydra override strings from the sweep manifest, e.g.
        ``["estimation.parameters_to_estimate.neptune_GM=false"]``.
    only:
        When given, write only this stage's slice (one rule per slice, for the
        same reason as in :mod:`orbitdet.pipeline.resolve`).

    Returns
    -------
    list[Path]
        Every file written.
    """
    _refuse_shared_stage_overrides(overrides)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if only is not None and only not in PER_COMBO_STAGES:
        raise ValueError(
            f"Stage {only!r} is not per-combination. Known: {sorted(PER_COMBO_STAGES)}"
        )

    cfg = load_config(experiment, overrides)

    stages = [only] if only is not None else list(PER_COMBO_STAGES)
    written: list[Path] = []
    for stage in stages:
        stage_cfg: DictConfig = _slice(cfg, STAGE_KEYS[stage])
        # Record which combination produced this slice, so a result directory
        # explains itself without needing the manifest.
        stage_cfg = OmegaConf.merge(
            stage_cfg,
            OmegaConf.create({"sweep_overrides": list(overrides)}),
        )
        path = output_dir / f"{stage}.yaml"
        path.write_text(OmegaConf.to_yaml(stage_cfg, resolve=True))
        written.append(path)

    logger.info(
        "Resolved sweep combination of %r (%d override(s)) -> %s",
        experiment,
        len(overrides),
        ", ".join(p.name for p in written),
    )
    return written
