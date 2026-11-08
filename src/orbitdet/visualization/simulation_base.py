"""Shared base class for the propagation-result-driven plots.

Every plot in this family is driven by exactly one
:class:`tudatpy.dynamics.propagation.SimulationResults` (or an object exposing a
``propagation_results`` attribute, such as an ``EstimationOutput``) and exactly
one figure configuration block.  The block is passed directly as ``cfg`` (the
contents of ``conf/figures/<name>.yaml``), so there is no ``use_figure_name``
routing: a plotter always draws a single figure from a single config.

Subclasses only implement :meth:`Plot._make_figure`.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from tudatpy.dynamics import propagation as prop

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


def _as_simulation_results(result):
    """Return a :class:`SimulationResults`, unwrapping ``.propagation_results``."""
    if hasattr(result, "propagation_results"):
        return result.propagation_results
    return result


class SimulationPlot(Plot):
    """Base class for plots driven by a :class:`SimulationResults`.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (the contents of
        ``conf/figures/<name>.yaml``).  ``output_file`` names the output.
    result : prop.SimulationResults
        The propagation result to plot.  An object exposing a
        ``propagation_results`` attribute (e.g. an ``EstimationOutput``) is also
        accepted and unwrapped.
    """

    def __init__(self, cfg: DictConfig, result: prop.SimulationResults):
        super().__init__(cfg, name=_output_name(cfg))
        self.result = _as_simulation_results(result)


class ArrayPlot(Plot):
    """Base class for plots driven by pre-computed arrays.

    These plots (propagated formal errors, RSW distance with uncertainty) are fed
    the epochs and the derived arrays directly, because the arrays are computed
    from a covariance propagation rather than read off a single result object.

    Parameters
    ----------
    cfg : DictConfig
        The figure's own configuration block (the contents of
        ``conf/figures/<name>.yaml``).  ``output_file`` names the output.
    epochs : np.ndarray
        Epochs in seconds since J2000.
    """

    def __init__(self, cfg: DictConfig, epochs):
        super().__init__(cfg, name=_output_name(cfg))
        self.epochs = epochs


__all__ = [
    "SimulationPlot",
    "ArrayPlot",
    "_as_simulation_results",
    "_cfg_get",
    "_output_name",
]
