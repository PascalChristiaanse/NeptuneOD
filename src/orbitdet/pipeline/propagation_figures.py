"""Stage: trajectory, RSW and formal-error figures.

All four families are fed by propagation and covariance artifacts rather than by
an observation collection:

``state_history_3d``
    3D view of the propagated trajectory.
``prefit_rsw`` / ``postfit_rsw``
    Offset between the ephemeris and the propagated body, in the RSW frame.
``propagated_formal_errors_cartesian`` / ``..._rsw``
    Cartesian and RSW formal errors from the covariance stage.
``RSW_distance_with_uncertainty``
    RSW offset with the position formal errors as an envelope.

Figure names are passed in rather than derived, because the DAG declares exact
output filenames such as ``prefit_rsw.pdf`` while the plotters would otherwise
name themselves after their class.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from orbitdet.pipeline.artifacts import load_propagation_results
from orbitdet.visualization.propagated_formal_errors_cartesian import (
    PropagatedFormalErrorsCartesian,
)
from orbitdet.visualization.propagated_formal_errors_rsw import PropagatedFormalErrorsRSW
from orbitdet.visualization.RSW_distance import RSWDistance
from orbitdet.visualization.rsw_distance_with_uncertainty import RSWDistanceWithUncertainty
from orbitdet.visualization.state_history_3d import StateHistory3D

logger = logging.getLogger(__name__)


def _publish(plotter, output_dir: Path, name: str) -> Path:
    """Run a plotter and verify its PDF landed where the DAG expects.

    The plotter's ``name`` is set explicitly: several plotters derive their own
    name from the class name, while the DAG declares an output filename such as
    ``prefit_rsw.pdf``. Setting it here keeps the two in step.
    """
    plotter.name = name
    plotter.output_dir = output_dir
    plotter.plot()
    pdf_path = output_dir / f"{name}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"Expected {pdf_path} after plotting, but it was not created.")
    logger.info("Wrote %s", pdf_path.name)
    return pdf_path


def run_state_history_figure(
    cfg: DictConfig,
    propagated_states_path: str | Path,
    output_dir: str | Path,
    figure_name: str = "state_history_3d",
) -> Path:
    """Render the 3D trajectory figure from the propagation results."""
    output_dir = Path(output_dir)
    results = load_propagation_results(propagated_states_path)
    return _publish(StateHistory3D(cfg, results), output_dir, figure_name)


def run_rsw_figure(
    cfg: DictConfig,
    propagated_states_path: str | Path,
    output_dir: str | Path,
    figure_name: str,
    kernels_cfg: DictConfig | None = None,
) -> Path:
    """Render the RSW distance figure from the propagation results.

    Requires the relative-position dependent variable, which the dynamics stage
    saves as the first dependent variable -- and **a furnished SPICE pool**.

    That second requirement is not obvious. ``RSWDistance`` resolves its
    dependent variables through ``create_dependent_variable_dictionary``, which
    builds a throwaway system of bodies to look up variable shapes. Building
    that system reads body radii from the kernel pool, so without kernels the
    figure dies with ``SpiceKERNELVARNOTFOUND: BODY10_RADII`` even though it
    never needs an ephemeris.

    ``kernels_cfg`` is therefore the kernel config slice, separate from the
    figure styling slice. Passing it explicitly rather than widening the figures
    slice keeps kernel-config edits from invalidating the other nineteen
    figures, which need no SPICE at all.
    """
    import tudatpy.dynamics.propagation_setup as prop_setup

    from orbitdet.pipeline.setup import furnish_kernels

    output_dir = Path(output_dir)
    if kernels_cfg is None:
        raise ValueError(
            f"{figure_name} needs a furnished SPICE pool (RSWDistance builds a "
            "system of bodies to resolve dependent variables), so its kernel "
            "config slice must be supplied."
        )
    furnish_kernels(kernels_cfg)

    results = load_propagation_results(propagated_states_path)

    # Rebuild the same dependent-variable setting the propagation used, so the
    # plotter can find the right entry in the results dictionary.
    relative_position = prop_setup.dependent_variable.relative_position("Triton Spice", "Triton")

    return _publish(
        RSWDistance(
            cfg,
            results,
            relative_position,
            central_body="Neptune",
            use_figure_name=figure_name,
        ),
        output_dir,
        figure_name,
    )


def run_formal_errors_figure(
    cfg: DictConfig,
    formal_errors_path: str | Path,
    output_dir: str | Path,
    figure_name: str,
) -> Path:
    """Render a formal-error figure from the covariance stage's archive.

    ``propagated_formal_errors_cartesian`` uses the Cartesian array;
    ``propagated_formal_errors_rsw`` uses the RSW-rotated one. Both live in the
    same ``.npz``, so one stage feeds both figures.
    """
    output_dir = Path(output_dir)
    archive = np.load(formal_errors_path)

    if figure_name == "propagated_formal_errors_cartesian":
        plotter = PropagatedFormalErrorsCartesian(cfg, archive["epochs"], archive["formal_errors"])
    elif figure_name == "propagated_formal_errors_rsw":
        plotter = PropagatedFormalErrorsRSW(
            cfg, archive["epochs"], archive["formal_errors_rsw"]
        )
    else:
        raise ValueError(
            f"Unknown formal-error figure {figure_name!r}. Known: "
            "propagated_formal_errors_cartesian, propagated_formal_errors_rsw"
        )

    return _publish(plotter, output_dir, figure_name)


def run_rsw_uncertainty_figure(
    cfg: DictConfig,
    formal_errors_path: str | Path,
    output_dir: str | Path,
    figure_name: str = "RSW_distance_with_uncertainty",
) -> Path:
    """Render the RSW distance plot with formal-error envelopes."""
    output_dir = Path(output_dir)
    archive = np.load(formal_errors_path)
    plotter = RSWDistanceWithUncertainty(
        cfg,
        archive["epochs"],
        archive["rsw_distance"],
        archive["formal_errors_rsw"][:, :3],
    )
    return _publish(plotter, output_dir, figure_name)
