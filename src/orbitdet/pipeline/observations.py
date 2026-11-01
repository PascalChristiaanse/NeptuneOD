"""Stage: build the observation collection.

This is the expensive stage that everything downstream depends on. It produces
two artifacts:

``observations.tudat``
    The ``ObservationCollection``, serialized with tudat's native
    ``save_to_binary``. Reloadable via ``ObservationCollection.load_from_binary``.
``dataset_metadata.json``
    The per-set metadata dict (``dataset_id``, ``index``, coordinate labels)
    that ``select_observation_collection_by_tag`` and the weighting/outlier
    engines need. It is not recoverable from the serialized collection, so it
    travels alongside it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from orbitdet.observations import create_observation_collection
from orbitdet.pipeline.artifacts import save_collection, save_metadata
from orbitdet.pipeline.context import build_context
from orbitdet.pipeline.setup import furnish_kernels
from orbitdet.simulation import get_environment

logger = logging.getLogger(__name__)


def run_observations(cfg: DictConfig, output_dir: str | Path) -> tuple[Path, Path]:
    """Build the observation collection and persist it.

    Returns
    -------
    tuple[Path, Path]
        ``(observations_path, metadata_path)``.
    """
    output_dir = Path(output_dir)
    ctx = build_context(cfg, output_dir)

    # Each rule is a separate process, so the SPICE pool starts empty.
    furnish_kernels(cfg)

    bodies = get_environment(cfg, ctx)
    logger.info("Environment built.")

    observations, observation_models, dataset_metadata = create_observation_collection(cfg, bodies)
    logger.info(
        "Observation collection built: %d observations across %d dataset(s).",
        len(observations.concatenated_times),
        len(dataset_metadata),
    )

    obs_path = save_collection(observations, output_dir / "observations.tudat")
    meta_path = save_metadata(dataset_metadata, output_dir / "dataset_metadata.json")

    logger.info("Wrote %s and %s", obs_path.name, meta_path.name)
    return obs_path, meta_path
