"""Stage: assemble the experiment's summary report.

Collects the small JSON summaries each stage writes -- estimation metrics,
covariance formal errors, outlier counts -- into one Markdown document that
Snakemake's ``--report`` renders alongside the figures.

Why a separate stage rather than reading the JSONs in the Snakefile
------------------------------------------------------------------
The Snakefile is a DAG description, not a place for formatting logic. Keeping
the assembly here means the report content is testable without a Snakemake run,
and the numbers are formatted once rather than in a template.

The report deliberately quotes *numbers* rather than embedding figures: the
figures are attached by the ``report()`` directives on the figure rules, so
Snakemake owns their layout and this file only supplies the narrative.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load(path: Path) -> dict | None:
    """Load a JSON summary, returning ``None`` when it is absent."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse %s; omitting from the report.", path)
        return None


def _fmt(value: float, unit: str = "") -> str:
    """Format a number in scientific notation with an optional unit."""
    return f"{value:.6g}{(' ' + unit) if unit else ''}"


def run_report(
    experiment: str,
    results_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Write the experiment's summary report.

    Parameters
    ----------
    experiment:
        Experiment name, used as the report title.
    results_dir:
        The experiment's result directory.
    output_path:
        Where to write the Markdown report.

    Returns
    -------
    Path
        The written report.
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    estimation = _load(results_dir / "estimation" / "estimation_summary.json")
    covariance = _load(results_dir / "covariance" / "covariance_summary.json")
    outliers = _load(results_dir / "preprocessing" / "outlier_rejection_metadata.json")

    lines: list[str] = [f"# {experiment}", ""]

    if estimation:
        lines += [
            "## Estimation",
            "",
            f"- Observations: {estimation['num_observations']}",
            f"- Iterations: {estimation['num_iterations']} "
            f"(best: {estimation['best_iteration']})",
            f"- Residual RMS: {_fmt(estimation['residuals_rms'])}",
            f"- Residual RMS at best iteration: "
            f"{_fmt(estimation['postfit_rms_best_iteration'])}",
            f"- Residual max |value|: {_fmt(estimation['residuals_max_abs'])}",
            f"- Duration: {_fmt(estimation['estimation_duration_s'], 's')}",
            "",
        ]

    if outliers:
        if outliers.get("enabled", True):
            lines += [
                "## Outlier rejection",
                "",
                f"- Accepted: {outliers['n_accepted']}",
                f"- Rejected: {outliers['n_rejected']}",
                f"- Total: {outliers['n_total_observations']}",
                "",
            ]
        else:
            lines += ["## Outlier rejection", "", "Disabled for this experiment.", ""]

    if covariance:
        lines += ["## Propagated formal errors", ""]
        if covariance.get("formal_errors_are_finite", True):
            lines += [
                f"- Epochs: {covariance['num_epochs']}",
                f"- Peak position sigma: {_fmt(covariance['max_position_sigma_m'], 'm')}",
                f"- Peak velocity sigma: {_fmt(covariance['max_velocity_sigma_m_s'], 'm/s')}",
                f"- Median position sigma: {_fmt(covariance['median_position_sigma_m'], 'm')}",
                "",
            ]
        else:
            # A degenerate covariance inverts to NaN rather than raising, so
            # say so explicitly. Printing the NaN would present a numerical
            # failure as if it were a measurement.
            lines += [
                "**Not available.** The estimated covariance is degenerate: the "
                "normal matrix is singular, so its inversion produced NaN. At "
                "least one estimated parameter is not constrained by these "
                "observations.",
                "",
                f"- Epochs propagated: {covariance['num_epochs']}",
                "- To obtain usable formal errors, remove the unconstrained "
                "parameters from `estimation.parameters_to_estimate`, or "
                "constrain them with an a priori covariance.",
                "",
            ]

    if estimation and estimation.get("covariance_is_finite") is False:
        lines += [
            "## Estimation covariance",
            "",
            "The normal matrix is singular (see the formal-errors section). The "
            "residual statistics above remain valid, but every uncertainty "
            "derived from the covariance is not.",
            "",
        ]

    if not any((estimation, covariance, outliers)):
        lines += [
            "No stage summaries were found. Run the pipeline before generating the report.",
            "",
        ]

    output_path.write_text("\n".join(lines))
    logger.info("Report written to %s", output_path)
    return output_path
