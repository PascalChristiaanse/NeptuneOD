"""Stage: assign observation weights.

Runs after outlier rejection. Produces the weighted collection plus the weight
DataFrames the summary tables are built from.

The weights live on the collection itself (``set_tabulated_weights``), and that
state survives ``save_to_binary``, so the estimation and figure stages can read
the weighted collection directly instead of recomputing weights.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from orbitdet.observations import WeightEngine
from orbitdet.observations.weighting.dataframe import WEIGHTS_DF_COLUMNS
from orbitdet.pipeline.artifacts import load_collection, load_metadata, save_collection
from orbitdet.pipeline.setup import prepare_environment

logger = logging.getLogger(__name__)


def run_weighting(
    cfg: DictConfig,
    observations_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    output_name: str = "weighted.tudat",
) -> tuple[Path, Path]:
    """Apply the configured weighting strategy and persist the result.

    Parameters
    ----------
    cfg:
        Resolved configuration with ``weighting`` and (optionally) ``grouping``.
    observations_path:
        Collection carrying residuals.
    metadata_path:
        ``dataset_metadata.json``.
    output_dir:
        Directory this stage writes into.
    output_name:
        Filename for the weighted collection.

    Returns
    -------
    tuple[Path, Path]
        ``(collection_path, weights_csv_path)``. When weighting is disabled the
        collection is forwarded unchanged and the CSV is written empty (headers
        only), so the DAG can declare both outputs unconditionally.
    """
    output_dir = Path(output_dir)
    observations = load_collection(observations_path)

    weighting_cfg = cfg.get("weighting") if "weighting" in cfg else None
    enabled = weighting_cfg is not None and bool(weighting_cfg.get("enabled", False))

    collection_path = output_dir / output_name
    weights_path = output_dir / "observation_weights.csv"

    if not enabled:
        logger.info("Weighting disabled; forwarding collection unchanged.")
        save_collection(observations, collection_path)
        # Headers only, so downstream table rules still find a parseable CSV.
        pd.DataFrame(columns=WEIGHTS_DF_COLUMNS).to_csv(weights_path, index=False)
        return collection_path, weights_path

    _, bodies = prepare_environment(cfg, output_dir, observations)
    metadata = load_metadata(metadata_path)

    # The outlier engine may carry its own ``grouping`` node, but weighting uses
    # the shared top-level one. Passing it explicitly keeps the two subsystems
    # independent -- an outlier group need not equal a weighting group.
    default_grouping = cfg.get("grouping") if "grouping" in cfg else None

    engine = WeightEngine.from_config(weighting_cfg, default_grouping=default_grouping)
    weighted, weights_df = engine.apply(
        observations, bodies, output_dir=output_dir, dataset_metadata=metadata
    )
    logger.info(
        "Weighting complete: %s strategy applied, %d observations weighted",
        weighting_cfg.get("strategy", "unknown"),
        len(weights_df) if not weights_df.empty else 0,
    )

    save_collection(weighted, collection_path)
    weights_df.to_csv(weights_path, index=False)
    logger.info("Per-observation weights written to %s", weights_path.name)

    return collection_path, weights_path


def load_weights(output_dir: str | Path) -> pd.DataFrame:
    """Load the per-observation weight DataFrame written by :func:`run_weighting`.

    Used by the summary-table stage, which is a pure function of this CSV.
    """
    path = Path(output_dir) / "observation_weights.csv"
    if not path.exists():
        logger.warning("No weights DataFrame at %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)
