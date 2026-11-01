"""Stage: render figures that come from the estimation output.

These figures need an ``EstimationOutput`` rather than an observation
collection: parameter correlation, parameter history, and residual RMS per
iteration. All three are read from the serialized output, so they re-run when
the estimation changes and not when a residual figure's styling changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig
from tudatpy.estimation import estimation_analysis as est_an

from orbitdet.pipeline.artifacts import tudat_stem
from orbitdet.visualization.parameter_correlation_heatmap import ParameterCorrelationHeatmap
from orbitdet.visualization.parameter_history_per_iteration import ParameterHistoryPerIteration
from orbitdet.visualization.residual_rms_per_iteration import ResidualRMSPerIteration

logger = logging.getLogger(__name__)

#: Figure name -> plotter class, for the estimation-output family.
ESTIMATION_PLOTTERS = {
    "parameter_correlation_heatmap": ParameterCorrelationHeatmap,
    "parameter_history_per_iteration": ParameterHistoryPerIteration,
    "residual_rms_per_iteration": ResidualRMSPerIteration,
}


def run_estimation_figure(
    cfg: DictConfig,
    estimation_path: str | Path,
    output_dir: str | Path,
    figure_name: str,
) -> Path:
    """Render one estimation-output figure.

    Parameters
    ----------
    cfg:
        The ``figures`` config slice.
    estimation_path:
        Serialized ``EstimationOutput``.
    output_dir:
        Directory for the PDF.
    figure_name:
        Key under ``cfg.figures`` and base name of the output PDF.

    Returns
    -------
    Path
        The written PDF.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotter_cls = ESTIMATION_PLOTTERS.get(figure_name)
    if plotter_cls is None:
        raise ValueError(
            f"Unknown estimation figure {figure_name!r}. "
            f"Known: {sorted(ESTIMATION_PLOTTERS)}"
        )

    estimation_output = est_an.EstimationOutput.load_from_binary(tudat_stem(estimation_path))
    logger.info("Loaded estimation output for %s.", figure_name)

    plotter = plotter_cls(cfg, estimation_output)
    # These plotters derive their own name from the class name, so the output
    # filename is the snake_case class name plus ".pdf" ("parameter_history_per_iteration.pdf",
    # etc.). Set it explicitly so the DAG's declared output matches exactly.
    plotter.name = figure_name
    plotter.output_dir = output_dir
    plotter.plot()

    pdf_path = output_dir / f"{figure_name}.pdf"
    if not pdf_path.exists():
        raise RuntimeError(f"Expected {pdf_path} after plotting, but it was not created.")
    logger.info("Wrote %s", pdf_path.name)
    return pdf_path
