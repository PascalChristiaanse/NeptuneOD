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

``differenced_dependent_variables`` plots the difference between two dependent
variables of the *same* result: ``dependent_variable_index`` is the reference
and ``comparison_dependent_variable_index`` the subtrahend.  That is how the
propagated Triton state is differenced against the SPICE ephemeris without a
second propagation.

SPICE kernels
-------------
``create_dependent_variable_dictionary`` builds a degenerate system of bodies to
resolve each variable's shape, which reads body radii from the kernel pool.  The
script therefore furnishes the kernels named by ``cfg.kernels`` before drawing
anything; without them the dependent-variable figures fail with
``BODY10_RADII could not be found``.

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
    apply_figure_style,
    attach_figure_config,
    output_dir,
    resolve_builder,
    select_matplotlib_backend,
    show_or_serve,
)
from omegaconf import DictConfig
from tudatpy.dynamics import propagation as prop

from orbitdet.data import KernelManager
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.utility import load_tudat_object
from orbitdet.visualization import (
    DependentVariable,
    DifferencedDependentVariables,
    PropagatedFormalErrorsCartesian,
    PropagatedFormalErrorsRSW,
    RSWDistance,
    RSWDistanceComparison,
    RSWDistanceDifference,
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
    "DifferencedDependentVariables": DifferencedDependentVariables,
    "RSWDistance": RSWDistance,
    "RSWDistanceComparison": RSWDistanceComparison,
    "RSWDistanceDifference": RSWDistanceDifference,
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

#: Plotters that difference two dependent variables of the same result.
DIFFERENCED_PLOTTERS = {"DifferencedDependentVariables"}

#: Plotters that overlay several results, read from ``cfg.simulation_results``.
MULTI_RESULT_PLOTTERS = {"RSWDistanceComparison"}

#: Plotters that subtract comparison results from a reference result.  The
#: reference is ``cfg.simulation_result``; the comparisons are
#: ``cfg.simulation_results``.
DIFFERENCE_PLOTTERS = {"RSWDistanceDifference"}


def _load_result(cfg: DictConfig) -> prop.SimulationResults:
    """Load the serialized simulation result named by ``cfg.simulation_result``.

    The propagation stage saves a :class:`SingleArcSimulationResults`, which is
    a subclass of :class:`SimulationResults`.  TudatPy's ``load_from_binary``
    checks the concrete type, so the subclass must be named explicitly.
    """
    path = cfg.simulation_result
    logger.info("Loading simulation result from %s", path)
    return load_tudat_object(prop.SingleArcSimulationResults, path)


def _load_results(cfg: DictConfig) -> list[prop.SimulationResults]:
    """Load every result named by ``cfg.simulation_results``, in order."""
    paths = list(cfg.simulation_results)
    if not paths:
        raise ValueError("cfg.simulation_results is empty; nothing to overlay.")
    results = []
    for path in paths:
        logger.info("Loading simulation result from %s", path)
        results.append(load_tudat_object(prop.SingleArcSimulationResults, path))
    return results


def _furnish_kernels(cfg: DictConfig) -> None:
    """Furnish SPICE before any dependent-variable figure is drawn.

    ``create_dependent_variable_dictionary`` builds a degenerate system of
    bodies to resolve each variable's shape, which reads body radii from the
    kernel pool.  Without kernels furnished it fails with
    ``BODY10_RADII could not be found``.  Loading is cheap and idempotent, so it
    runs for every figure rather than only the ones that need it.
    """
    logger.info("Furnishing SPICE kernels...")
    km: KernelManager = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()


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

    # The journal style must be applied before any figure is built, so it runs
    # once the figure's own config (which may name a style) is merged in.
    apply_figure_style(cfg, figure_cfg)

    # ``initialize`` writes the run metadata and freezes the config, so it runs
    # after the figure config has been merged in.
    initialize(cfg)

    _furnish_kernels(cfg)

    logger.info("Plotting figure '%s'...", figure_name)
    if plotter_name in ARRAY_PLOTTERS:
        arrays = _load_formal_errors(cfg)
        _build_array_figure(plotter_name, figure_cfg, arrays).plot()
    elif plotter_name in MULTI_RESULT_PLOTTERS:
        # Several results overlaid on one figure.  The labels come from the
        # figure config so the rule stays free of model names.
        results = _load_results(cfg)
        index = int(figure_cfg.get("dependent_variable_index", 0))
        dependent_variable = _select_dependent_variable(results[0], index)
        labels = figure_cfg.get("labels")
        builder(
            figure_cfg,
            results,
            dependent_variable,
            list(labels) if labels is not None else None,
        ).plot()
    elif plotter_name in DIFFERENCE_PLOTTERS:
        # One reference result minus one or more comparison results.  The
        # reference is ``simulation_result``; the comparisons are
        # ``simulation_results``.
        reference = _load_result(cfg)
        comparisons = _load_results(cfg)
        index = int(figure_cfg.get("dependent_variable_index", 0))
        dependent_variable = _select_dependent_variable(reference, index)
        labels = figure_cfg.get("labels")
        builder(
            figure_cfg,
            reference,
            comparisons,
            dependent_variable,
            list(labels) if labels is not None else None,
        ).plot()
    else:
        result = _load_result(cfg)
        if plotter_name in DEPENDENT_VARIABLE_PLOTTERS:
            index = int(figure_cfg.get("dependent_variable_index", 0))
            dependent_variable = _select_dependent_variable(result, index)
            builder(figure_cfg, result, dependent_variable).plot()
        elif plotter_name in DIFFERENCED_PLOTTERS:
            # Both variables come from the same result: the reference is one
            # saved dependent variable, the comparison is another.  This is how
            # the propagated Triton state is differenced against the SPICE
            # ephemeris without a second propagation.
            reference_index = int(figure_cfg.get("dependent_variable_index", 0))
            comparison_index = int(figure_cfg.get("comparison_dependent_variable_index", 1))
            reference_dv = _select_dependent_variable(result, reference_index)
            comparison_dv = _select_dependent_variable(result, comparison_index)
            builder(
                figure_cfg,
                result,
                [result],
                reference_dv,
                [comparison_dv],
            ).plot()
        else:
            builder(figure_cfg, result).plot()
    logger.info("Figure '%s' written to %s", figure_name, output_dir())

    show_or_serve()


if __name__ == "__main__":
    main()
