"""Shared base class for the estimation-output-driven plots.

Every plot in this family is driven by exactly one
:class:`tudatpy.estimation.estimation_analysis.EstimationOutput` and exactly one
figure configuration block.  The block is passed directly as ``cfg`` (the
contents of ``conf/figures/<name>.yaml``), so there is no ``use_figure_name``
routing: a plotter always draws a single figure from a single config.

Subclasses only implement :meth:`Plot._make_figure`.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.visualization.base import Plot


def _cfg_get(cfg, *keys, default=None):
    """Read a nested config value, returning ``default`` when any key is missing."""
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        try:
            cur = cur.get(key)
        except Exception:
            try:
                cur = cur[key]
            except Exception:
                return default
    return default if cur is None else cur


def _output_name(cfg) -> str | None:
    """Derive the output stem from the config's ``output_file``, if present."""
    output_file = _cfg_get(cfg, "output_file", default=None)
    if not output_file:
        return None
    return Path(str(output_file)).stem


class EstimationPlot(Plot):
    """Base class for plots driven by an :class:`EstimationOutput`.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (the contents of
        ``conf/figures/<name>.yaml``).  ``output_file`` names the output.
    estimation_output : est_an.EstimationOutput
        The estimation output whose covariance / history / residuals are plotted.
    """

    #: Whether this plot needs the system of bodies and runtime context.
    requires_environment: bool = False

    def __init__(self, cfg: DictConfig, estimation_output: est_an.EstimationOutput):
        super().__init__(cfg, name=_output_name(cfg))
        self.estimation_output = estimation_output

    def _figure_size(self, default_width: float, default_height: float) -> tuple[float, float]:
        """Return the configured ``(width, height)``, falling back to defaults."""
        width = float(_cfg_get(self.cfg, "figure", "width", default=default_width))
        height = float(_cfg_get(self.cfg, "figure", "height", default=default_height))
        return width, height

    def _parameter_names(self, n_params: int, default_names: list[str]) -> list[str]:
        """Resolve the parameter tick labels, padded/truncated to ``n_params``."""
        names = list(_cfg_get(self.cfg, "parameter_names", default=default_names))
        if len(names) < n_params:
            names += [f"p{i}" for i in range(len(names), n_params)]
        return names[:n_params]


__all__ = ["EstimationPlot", "_cfg_get", "_output_name"]
