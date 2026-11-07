"""Pipeline stage: draw one propagation figure from a saved simulation result.

Loads a serialized :class:`~tudatpy.dynamics.propagation.SimulationResults` and
draws exactly one figure from it.  No propagation is re-run; the state history
and dependent-variable history are read straight off the artifact.

Runnable standalone::

    python scripts/plot_simulation_figure.py \\
        figure=state_history_3d \\
        figure_config=conf/figures/state_history_3d.yaml \\
        simulation_result=results/pipeline/chapter_3/final_propagation_result.tudat

or as a Snakemake rule, which passes the same overrides.

The figure is written to ``<hydra.run.dir>/<figure name>.pdf`` plus a ``.fig``
pickle for interactive inspection.

Dependent-variable figures
--------------------------
``dependent_variable`` and ``rsw_distance`` plot one dependent variable from the
result.  Which one is selected by ``dependent_variable_index`` in the figure
config: an index into the result's ``ordered_dependent_variable_settings`` (the
same order the propagation saved them in).  This keeps the script free of any
hard-coded body names.

Plotter selection
-----------------
Each figure config names its plotter class with a ``plotter`` key (e.g.
``plotter: "StateHistory3D"``).  The registry maps that class name to the class,
so one config file per figure instance can share a single plotter.

Array figures
-------------
``propagated_formal_errors_cartesian``, ``propagated_formal_errors_rsw`` and
``rsw_distance_with_uncertainty`` are fed pre-computed arrays rather than a
result object, because the arrays come from a covariance propagation.  They read
an ``.npz`` archive named by ``cfg.formal_errors`` (written by the estimation
stage) with the keys ``epochs``, ``formal_errors``, ``formal_errors_rsw``,
``rsw_components`` and ``rsw_sigma``.
"""

import logging

import hydra
import numpy as np
from _figure_common import (
    attach_figure_config,
    output_dir,
    resolve_builder,
    select_matplotlib_backend,
    show_or_serve,
)
from omegaconf import DictConfig
from tudatpy.dynamics import propagation as prop

from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.utility import load_tudat_object
from orbitdet.visualization import (
    DependentVariable,
    PropagatedFormalErrorsCartesian,
    PropagatedFormalErrorsRSW,
    RSWDistance,
    RSWDistanceWithUncertainty,
    StateHistory3D,
)

logger = logging.getLogger(__name__)


#: Every simulation plotter this script can draw, keyed by the class name a
#: figure config declares with its ``plotter`` key.  ``result`` plotters take
#: ``(figure_cfg, result)``; ``array`` plotters take ``(figure_cfg, epochs, ...)``
#: and are fed from the ``.npz`` archive.
FIGURE_BUILDERS = {
    "StateHistory3D": StateHistory3D,
    "DependentVariable": DependentVariable,
    "RSWDistance": RSWDistance,
    "PropagatedFormalErrorsCartesian": PropagatedFormalErrorsCartesian,
    "PropagatedFormalErrorsRSW": PropagatedFormalErrorsRSW,
    "RSWDistanceWithUncertainty": RSWDistanceWithUncertainty,
}

#: Plotters that read the ``.npz`` archive instead of the simulation result.
ARRAY_PLOTTERS = {
    "PropagatedFormalErrorsCartesian",
    "PropagatedFormalErrorsRSW",
    "RSWDistanceWithUncertainty",
}

#: Plotters that plot a single dependent variable selected by index.
DEPENDENT_VARIABLE_PLOTTERS = {"DependentVariable", "RSWDistance"}


def _load_result(cfg: DictConfig) -> prop.SimulationResults:
    """Load the serialized simulation result named by ``cfg.simulation_result``."""
    path = cfg.simulation_result
    logger.info("Loading simulation result from %s", path)
    return load_tudat_object(prop.SimulationResults, path)


def _load_formal_errors(cfg: DictConfig) -> dict:
    """Load the ``.npz`` archive of pre-computed formal errors."""
    path = cfg.formal_errors
    logger.info("Loading formal errors from %s", path)
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _select_dependent_variable(result: prop.SimulationResults, index: int):
    """Return the dependent-variable settings at ``index`` in the saved order."""
    settings = list(result.ordered_dependent_variable_settings.values())
    if not 0 <= index < len(settings):
        raise IndexError(
            f"dependent_variable_index={index} is out of range; the result saved "
            f"{len(settings)} dependent variable(s)."
        )
    return settings[index]


def _build_array_figure(plotter_name: str, figure_cfg: DictConfig, arrays: dict):
    """Construct an array-driven plotter from the ``.npz`` archive."""
    epochs = arrays["epochs"]
    if plotter_name == "PropagatedFormalErrorsCartesian":
        return PropagatedFormalErrorsCartesian(figure_cfg, epochs, arrays["formal_errors"])
    if plotter_name == "PropagatedFormalErrorsRSW":
        return PropagatedFormalErrorsRSW(figure_cfg, epochs, arrays["formal_errors_rsw"])
    if plotter_name == "RSWDistanceWithUncertainty":
        return RSWDistanceWithUncertainty(
            figure_cfg, epochs, arrays["rsw_components"], arrays["rsw_sigma"]
        )
    raise ValueError(f"'{plotter_name}' is not an array plotter.")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="plot_simulation_figure",
)
@enforce_initialization
def main(cfg: DictConfig):
    select_matplotlib_backend()

    figure_name = cfg.figure

    cfg = attach_figure_config(cfg, figure_name)
    figure_cfg = cfg.figures[figure_name]
    builder = resolve_builder(FIGURE_BUILDERS, figure_cfg)
    plotter_name = figure_cfg.plotter

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the figure config has been merged in.
    initialize(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    if plotter_name in ARRAY_PLOTTERS:
        arrays = _load_formal_errors(cfg)
        _build_array_figure(plotter_name, figure_cfg, arrays).plot()
    else:
        result = _load_result(cfg)
        if plotter_name in DEPENDENT_VARIABLE_PLOTTERS:
            index = int(figure_cfg.get("dependent_variable_index", 0))
            dependent_variable = _select_dependent_variable(result, index)
            builder(figure_cfg, result, dependent_variable).plot()
        else:
            builder(figure_cfg, result).plot()
    logger.info("Figure '%s' written to %s", figure_name, output_dir())

    show_or_serve()


if __name__ == "__main__":
    main()
