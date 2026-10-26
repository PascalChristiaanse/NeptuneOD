"""Stage: build the observation collection.

This is the expensive stage that everything downstream depends on. It produces
two artifacts:

``observations.tudat``
    The ``ObservationCollection``, serialized with tudat's native
    ``save_to_binary``. Reloadable via ``ObservationCollection.load_from_binary``.
``dataset_metadata.json``
    The per-set metadata dict (``dataset_id``, ``index``, coordinate labels)
    that ``select_observation_collection_by_tag`` needs. It is not recoverable
    from the serialized collection, so it travels alongside it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import DictConfig

from orbitdet.observations import create_observation_collection
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

    obs_path = output_dir / "observations.tudat"
    # tudat's save_to_binary appends ".tudat" unconditionally, so passing a path
    # that already ends in ".tudat" yields "observations.tudat.tudat". Pass the
    # stem and let tudat add the extension.
    observations.save_to_binary(str(obs_path.with_suffix("")))
    if not obs_path.exists():
        raise RuntimeError(
            f"Expected {obs_path} after save_to_binary, but it was not created. "
            f"Directory contains: {sorted(p.name for p in output_dir.iterdir())}"
        )

    meta_path = output_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(dataset_metadata, indent=2, default=str))

    logger.info("Wrote %s and %s", obs_path.name, meta_path.name)
    return obs_path, meta_path
