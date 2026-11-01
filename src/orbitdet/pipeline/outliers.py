"""Stage: reject outliers from the observation collection.

Runs before weighting and after the residuals stage, because the rejection
strategies compare residuals against thresholds. Produces the filtered
collection plus a JSON rejection report.

Why this is its own file boundary
---------------------------------
``ObservationCollection`` serializes its weights *and* its residuals, so the
outlier-filtered collection is a complete input to the weighting stage. That
makes outlier rejection separately invalidatable: editing an outlier threshold
re-runs this stage and weighting, but not the expensive observation build.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import DictConfig

from orbitdet.observations import OutlierEngine
from orbitdet.pipeline.artifacts import load_collection, load_metadata, save_collection
from orbitdet.pipeline.setup import prepare_environment

logger = logging.getLogger(__name__)


def run_outlier_rejection(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    output_name: str = "outliers_applied.tudat",
) -> tuple[Path, Path | None]:
    """Apply configured outlier rejection and persist the filtered collection.

    Parameters
    ----------
    cfg:
        Resolved configuration containing the ``outlier_rejection`` section.
    observations_path:
        Collection carrying residuals (from the ephemeris residuals stage).
    metadata_path:
        ``dataset_metadata.json``.
    output_dir:
        Directory this stage writes into.
    output_name:
        Filename for the filtered collection.

    Returns
    -------
    tuple[Path, Path]
        ``(collection_path, report_path)``. When rejection is disabled the
        collection is forwarded unchanged and the report records
        ``enabled: false``, so the DAG can declare both outputs unconditionally.
    """
    output_dir = Path(output_dir)
    observations = load_collection(observations_path)

    outlier_cfg = cfg.get("outlier_rejection") if "outlier_rejection" in cfg else None
    enabled = outlier_cfg is not None and bool(outlier_cfg.get("enabled", False))

    collection_path = output_dir / output_name
    report_path = output_dir / "outlier_rejection_metadata.json"

    if not enabled:
        logger.info("Outlier rejection disabled; forwarding collection unchanged.")
        save_collection(observations, collection_path)
        # The report is written even when disabled, so the DAG can declare it
        # unconditionally. A missing-but-optional output would otherwise force
        # every caller to know whether rejection is enabled.
        report_path.write_text(
            json.dumps(
                {
                    "enabled": False,
                    "n_accepted": len(observations.concatenated_times),
                    "n_rejected": 0,
                    "n_total_observations": len(observations.concatenated_times),
                },
                indent=2,
            )
        )
        return collection_path, report_path

    _, bodies = prepare_environment(cfg, output_dir, observations)
    metadata = load_metadata(metadata_path)

    engine = OutlierEngine.from_config(outlier_cfg)
    filtered, rejection_metadata = engine.apply(
        observations, bodies, dataset_metadata=metadata
    )
    logger.info(
        "Outlier rejection complete: %d accepted, %d rejected out of %d",
        rejection_metadata["n_accepted"],
        rejection_metadata["n_rejected"],
        rejection_metadata["n_total_observations"],
    )

    save_collection(filtered, collection_path)
    rejection_metadata["enabled"] = True
    report_path.write_text(json.dumps(rejection_metadata, indent=2, default=str))
    logger.info("Rejection report written to %s", report_path.name)

    return collection_path, report_path
