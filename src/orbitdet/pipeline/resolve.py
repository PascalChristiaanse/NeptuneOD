"""Stage: resolve the Hydra config tree into per-stage tracked artifacts.

Snakemake only tracks its own ``configfile:``. The Hydra tree in ``conf/`` is
read *inside* the stage processes, so from Snakemake's perspective it is an
invisible side channel: editing ``conf/weighting/router.yaml`` would silently
leave every downstream output stale.

This rule closes that hole by composing the experiment config once and writing
**one slice per stage** to ``results/<experiment>/config/<stage>.yaml``. Each
stage rule declares only its own slice as an input, so:

* a change to ``conf/`` invalidates the slices that actually contain the
  changed key, and nothing else;
* each result directory carries a permanent record of the exact config that
  produced it.

Slicing matters. A single combined ``config.yaml`` would mean editing a figure
title invalidates the observation build and the estimation -- correct but
useless. With slices, editing ``conf/figures/residuals.yaml`` touches only the
``figures`` slice.

The written YAML has all interpolations resolved, so downstream stages load it
with plain ``OmegaConf.load`` and need no Hydra import at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from orbitdet.pipeline.config import load_config

logger = logging.getLogger(__name__)

# Keys every stage needs: ``build_context`` reads the epochs and seed, and
# ``setup_logging`` reads the logging section.
_CONTEXT_KEYS = ["seed", "start_date", "end_date", "initial_epoch", "logging"]

# ``KernelManager`` reads these.
_KERNEL_KEYS = [
    "kernel_folder",
    "kernels",
    "data_folder",
    "data_files",
    "use_default_kernels",
]

# ``get_environment`` reads these.
#
# ``datasets`` belongs here, not only on the observations stage. The environment
# build resolves a *tabulated* ephemeris by looking up
# ``cfg.datasets.<source_dataset>`` (see ``_configure_ephemeris_model``), so any
# stage that builds the environment needs the dataset config -- even one that
# loads an already-built observation collection from disk. Omitting it fails
# with "Voyager dataset configuration ... is required to build a tabulated
# ephemeris", which is exactly what happened for classic_triton_state.
_ENVIRONMENT_KEYS = [
    "bodies_to_create",
    "use_default_body_settings",
    "global_frame_origin",
    "global_frame_orientation",
    "datasets",
]

# Retained as a named group for readability at the call sites below.
_DATASET_KEYS = ["datasets"]

# ``get_dynamical_model``, ``get_integrator_settings`` and
# ``get_propagator_settings`` read these, as do ``get_estimatable_parameters``
# and the a priori covariance builder.
_DYNAMICS_KEYS = ["bodies_to_propagate", "integrator", "propagation"]

# ``get_estimatable_parameters`` and ``get_apriori_covariance_matrix``.
_ESTIMATION_KEYS = ["estimation", "apriori"]

#: Top-level config keys each stage actually reads.
#:
#: Every stage that calls ``prepare_environment`` needs the full environment
#: key set *including* ``datasets``, because the tabulated-ephemeris path reads
#: ``cfg.datasets``. ``figures`` is the exception: it never builds an
#: environment, so it needs no environmental keys at all.
#:
#: ``figures`` needs ``data_folder`` because the residual plotters resolve
#: ground-station labels through ``get_observatory_info``, which reads
#: ``observatories.txt`` from there.
#:
#: ``weighting`` and ``outlier_rejection`` each need their own subsystem config
#: *and* ``grouping``: the outlier engine may be given its own ``grouping:`` node
#: but falls back to the top-level one, and the weighting engine receives the
#: top-level one explicitly.
STAGE_KEYS: dict[str, list[str]] = {
    "kernels": _CONTEXT_KEYS + _KERNEL_KEYS,
    "observations": _CONTEXT_KEYS + _KERNEL_KEYS + _ENVIRONMENT_KEYS + _DATASET_KEYS,
    "outlier_rejection": (
        _CONTEXT_KEYS
        + _KERNEL_KEYS
        + _ENVIRONMENT_KEYS
        + ["outlier_rejection", "grouping"]
    ),
    "weighting": _CONTEXT_KEYS
    + _KERNEL_KEYS
    + _ENVIRONMENT_KEYS
    + ["weighting", "grouping"],
    "dynamics": _CONTEXT_KEYS + _KERNEL_KEYS + _ENVIRONMENT_KEYS + _DYNAMICS_KEYS,
    "prefit": _CONTEXT_KEYS + _KERNEL_KEYS + _ENVIRONMENT_KEYS + _DYNAMICS_KEYS,
    "estimation": _CONTEXT_KEYS
    + _KERNEL_KEYS
    + _ENVIRONMENT_KEYS
    + _DYNAMICS_KEYS
    + _ESTIMATION_KEYS,
    "covariance": _CONTEXT_KEYS
    + _KERNEL_KEYS
    + _ENVIRONMENT_KEYS
    + _DYNAMICS_KEYS
    + _ESTIMATION_KEYS,
    "figures": ["figures", "data_folder"],
}


def _slice(cfg: DictConfig, keys: list[str]) -> DictConfig:
    """Return a new config containing only *keys* that are present in *cfg*."""
    present = {key: cfg[key] for key in keys if key in cfg}
    return OmegaConf.create(present)


def run_resolve_config(
    experiment: str,
    output_dir: str | Path,
    overrides: list[str] | None = None,
    only: str | None = None,
) -> list[Path]:
    """Compose the experiment config and write one slice per stage.

    Parameters
    ----------
    experiment:
        Experiment config name, e.g. ``"classic_triton_state"``.
    output_dir:
        Directory to write the slices into.
    overrides:
        Hydra-style overrides applied during composition.
    only:
        When given, write only this stage's slice. Snakemake uses this so that
        each stage's slice is produced by its own rule: a single rule emitting
        every slice would make Snakemake treat all of them as "will be
        updated", scheduling the whole downstream DAG even when only one
        slice's content actually changes.

    Returns
    -------
    list[Path]
        Every file written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if only is not None and only not in STAGE_KEYS:
        raise ValueError(f"Unknown stage {only!r}. Known stages: {sorted(STAGE_KEYS)}")

    cfg = load_config(experiment, overrides)

    stages = [only] if only is not None else list(STAGE_KEYS)
    written: list[Path] = []
    for stage in stages:
        stage_cfg = _slice(cfg, STAGE_KEYS[stage])
        path = output_dir / f"{stage}.yaml"
        # ``resolve=True`` evaluates every interpolation (``${now:...}``,
        # ``${git_commit:}``, cross-references) so the written file is
        # self-contained and needs no resolvers to read back.
        path.write_text(OmegaConf.to_yaml(stage_cfg, resolve=True))
        written.append(path)

    logger.info(
        "Resolved experiment %r (%d override(s)) -> %s",
        experiment,
        len(overrides or []),
        ", ".join(p.name for p in written),
    )
    return written
