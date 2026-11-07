"""Shared plumbing for the per-family figure scripts.

Each ``plot_*_figure.py`` script draws exactly one figure per process from a
saved artifact.  They all share the same skeleton:

* pick a matplotlib backend (interactive when a display exists, else WebAgg),
* load the figure's own config file and merge it into ``cfg.figures.<name>``,
* load the artifact(s) the figure needs,
* build the plotter and publish it.

This module holds the parts that do not depend on the artifact type.  The
artifact loading and the figure registry stay in each script, because they are
what actually differ between the residual, estimation and simulation families.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def select_matplotlib_backend() -> None:
    """Pick an interactive backend when a display exists, else WebAgg.

    Mirrors the selection used by the other plotting scripts so a figure run
    from a terminal behaves the same as one run from the debugger.
    """
    display = os.environ.get("DISPLAY")
    is_headless_display = display in (":99", "localhost:99", "127.0.0.1:99")

    matplotlib.rcParams["webagg.port"] = 8988
    matplotlib.rcParams["webagg.open_in_browser"] = False

    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or is_headless_display:
        matplotlib.use("WebAgg", force=True)
    elif display or os.environ.get("WAYLAND_DISPLAY"):
        try:
            matplotlib.use("QtAgg", force=True)
        except Exception:
            matplotlib.use("TkAgg", force=True)
    else:
        matplotlib.use("WebAgg", force=True)


def apply_figure_style(cfg: DictConfig, figure_cfg: DictConfig | None = None) -> str:
    """Apply the journal style named by the figure config, or the default.

    The style is read from ``figure_cfg.style`` (the figure's own block) and
    falls back to :data:`~orbitdet.visualization.style.DEFAULT_STYLE`.  It is
    applied once per process, before any figure is built, so every figure in a
    run shares one typographic style.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration, consulted for a top-level ``figure_style``.
    figure_cfg : DictConfig, optional
        The figure's own config block, whose ``style`` key takes precedence.

    Returns
    -------
    str
        The style name that was applied.
    """
    from orbitdet.visualization.style import apply_style

    style = None
    if figure_cfg is not None:
        style = figure_cfg.get("style")
    if style is None:
        style = cfg.get("figure_style")

    return apply_style(style)


def attach_figure_config(cfg: DictConfig, figure_name: str) -> DictConfig:
    """Merge the figure's own config file into ``cfg.figures.<figure_name>``.

    The plotters read their styling from the figure's own config block.  Rather
    than compose every figure config into one experiment file, each figure rule
    passes the path of the single file it needs and it is merged in here.  That
    keeps a figure's styling next to the rule that draws it, and means editing
    one figure cannot invalidate another.
    """
    path = cfg.figure_config
    logger.info("Loading figure config from %s", path)

    figure_cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    cfg.figures = {figure_name: figure_cfg}
    OmegaConf.set_struct(cfg, True)

    return cfg


def resolve_builder(registry: dict, figure_cfg):
    """Return the plotter class declared by the figure config.

    The figure config names its plotter with a ``plotter`` key (e.g.
    ``plotter: "Residuals"``).  This is what lets one config file per figure
    instance (``prefit_residuals_spice``, ``postfit_residuals``, ...) share a
    single plotter class without a hand-maintained name -> class registry.

    Parameters
    ----------
    registry : dict
        Maps plotter class name to the class, e.g. ``{"Residuals": Residuals}``.
    figure_cfg : DictConfig
        The figure's own config block, which must carry a ``plotter`` key.

    Returns
    -------
    type
        The plotter class to instantiate.
    """
    plotter_name = figure_cfg.get("plotter")
    if plotter_name is None:
        raise ValueError(
            "Figure config has no 'plotter' key. Add e.g. 'plotter: \"Residuals\"' "
            f"to the config. Available plotters: {sorted(registry)}."
        )
    builder = registry.get(plotter_name)
    if builder is None:
        raise ValueError(f"Unknown plotter '{plotter_name}'. Available: {sorted(registry)}.")
    return builder


def running_under_snakemake() -> bool:
    """Whether this process was launched by a Snakemake job.

    Snakemake does not export a marker variable of its own, so detection uses
    the two signals that are actually available:

    * ``SNAKEMAKE_*`` environment variables, in case a future version adds one;
    * a non-tty stdout.  Every rule in this project redirects the stage's output
      to its log file (``> {log} 2>&1``), so a job's stdout is never a terminal
      while an interactive run's is.

    The second signal is the one that fires today.  It is also why a manually
    piped run is treated as non-interactive, which is the desired behaviour:
    there is no display to show a figure on.
    """
    if any(key.startswith("SNAKEMAKE_") for key in os.environ):
        return True
    return not sys.stdout.isatty()


def show_or_serve(cfg: DictConfig | None = None) -> None:
    """Show the figure interactively, or log where to find it.

    A figure is only shown when the run is interactive.  Under Snakemake the
    figure has already been written to the run directory, and blocking on
    ``plt.show()`` would hang the job until the window is closed.

    ``figure_show`` in the run config overrides the detection: ``true`` always
    shows, ``false`` never does.
    """
    override = None if cfg is None else cfg.get("figure_show")
    if override is not None:
        if not bool(override):
            logger.info("figure_show is false; not opening the figure.")
            return
    elif running_under_snakemake():
        logger.info("Non-interactive run; not opening the figure.")
        return

    backend = plt.get_backend().lower()
    if backend == "webagg":
        logger.info("Open the interactive plot at http://localhost:8988")
    elif backend not in ("agg",) and "inline" not in backend:
        plt.show(block=True)


def output_dir() -> str:
    """Return the Hydra run directory the figure was written to."""
    return HydraConfig.get().runtime.output_dir


__all__ = [
    "apply_figure_style",
    "attach_figure_config",
    "output_dir",
    "resolve_builder",
    "running_under_snakemake",
    "select_matplotlib_backend",
    "show_or_serve",
]
