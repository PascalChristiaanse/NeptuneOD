"""Stage: aggregate a sweep's per-combination telemetry into one table.

Each sweep combination produces the same small JSON summaries as a standalone
experiment. This stage collects them into a single tidy CSV so the comparison
figure and the report have one thing to read.

The "telemetry" is deliberately the metrics every run produces regardless of
what it varied:

``residuals_rms`` / ``postfit_rms_best_iteration``
    Fit quality.
``num_iterations`` / ``best_iteration``
    How hard the solver had to work.
``estimation_duration_s``
    Cost.
``covariance_is_finite``
    Whether the covariance is usable at all.

That last one matters for the ``fit_to_ephemeris`` sweep specifically. The base
experiment estimates a parameter set the observations cannot constrain (the
normal matrix comes out singular, so the inversion yields NaN rather than
raising). Different parameter subsets have different ranks, so which combinations
produce a usable covariance is a *result* of the sweep, not a footnote.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from orbitdet.pipeline.sweeps import load_manifest, sweep_output_dir

logger = logging.getLogger(__name__)

#: Columns taken from the manifest rather than a summary.
MANIFEST_COLUMNS = ("combo", "overrides")

#: Columns taken from the estimation summary.
ESTIMATION_COLUMNS = (
    "num_observations",
    "num_parameters",
    "num_iterations",
    "best_iteration",
    "residuals_rms",
    "postfit_rms_best_iteration",
    "residuals_mean",
    "residuals_std",
    "residuals_max_abs",
    "estimation_duration_s",
    "covariance_is_finite",
)

#: Columns taken from the covariance summary. Absent when a combination has no
#: covariance archive, which is itself a result worth recording.
COVARIANCE_COLUMNS = (
    "formal_errors_are_finite",
    "max_position_sigma_m",
)

#: Columns of the aggregated telemetry table, in order. Declared explicitly so
#: an empty sweep still writes a valid header.
BASE_COLUMNS = MANIFEST_COLUMNS + ESTIMATION_COLUMNS + COVARIANCE_COLUMNS


def _load_summary(path: Path) -> dict | None:
    """Load one combination's estimation summary, or ``None`` if absent."""
    if not path.exists():
        logger.warning("No estimation summary at %s; skipping this combination.", path)
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Could not parse %s; skipping this combination.", path)
        return None


def _override_value(manifest: dict, combo: dict) -> str:
    """Render a combination's varied parameters as one readable string.

    Uses the short last path component of each key, so an axis reads
    ``neptune_GM=false`` rather than
    ``estimation.parameters_to_estimate.neptune_GM=false``.
    """
    parts = []
    for key, value in combo["values"].items():
        parts.append(f"{key.rsplit('.', 1)[-1]}={value}")
    return ",".join(parts)


def collect_telemetry(
    sweep: str,
    output_csv: str | Path,
) -> Path:
    """Aggregate a sweep's per-combination telemetry into one CSV.

    Parameters
    ----------
    sweep:
        Sweep name, e.g. ``"fit_to_ephemeris"``. Results are read from
        ``sweeps/<sweep>/<combo>/``.
    output_csv:
        Where to write the aggregated table.

    Returns
    -------
    Path
        The written CSV.
    """
    base = sweep_output_dir(sweep)
    manifest = load_manifest(base / "manifest.json")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for combo in manifest["combos"]:
        combo_dir = base / combo["id"]
        summary = _load_summary(combo_dir / "estimation" / "estimation_summary.json")
        if summary is None:
            continue

        # The covariance summary is optional: a combination whose estimation
        # produced no usable covariance payload has no archive to summarise.
        covariance = _load_summary(combo_dir / "covariance" / "covariance_summary.json") or {}

        row = {
            "combo": combo["id"],
            "overrides": _override_value(manifest, combo),
            **{key: summary.get(key, "") for key in ESTIMATION_COLUMNS},
            **{key: covariance.get(key, "") for key in COVARIANCE_COLUMNS},
        }
        # Record every varied parameter as its own column so the figure can
        # group by any single axis without re-parsing the overrides string.
        for key, value in combo["values"].items():
            row[key] = value
        rows.append(row)

    # Columns: fixed ones first, then the varied parameters in declaration order.
    varied = sorted({key for combo in manifest["combos"] for key in combo["values"]})
    columns = list(BASE_COLUMNS) + varied

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "Sweep %r telemetry: %d combination(s) -> %s",
        sweep,
        len(rows),
        output_csv.name,
    )
    return output_csv
