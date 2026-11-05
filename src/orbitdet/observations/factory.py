"""Central factory for creating observation datasets."""

import inspect
import logging

import pandas as pd
import tudatpy.dynamics.environment as env
from omegaconf import DictConfig

from .registry import resolve_factory

logger = logging.getLogger(__name__)


def create_observation_dataset(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    frame: pd.DataFrame | None = None,
) -> tuple:
    """Create an observation dataset from configuration.

    This is the central dispatcher, and the single entry point for turning a
    dataset config into observation sets and models.  It:

    1. Accepts a dataset config (Hydra DictConfig or structured dataclass)
    2. Selects the factory: by the standard frame's ``(observation_type,
       coordinate_kind)`` when present, otherwise by the legacy ``type`` string
    3. Invokes the factory to construct the observation dataset object

    Keying on the two normalised fields rather than the fused ``type`` string is
    what lets one factory serve every dataset sharing an observation model: the
    fused names encode category, coordinates *and* receptor, so they proliferate
    (15+ in the corpus, typos included) while only the first two select a model.

    The frame is passed explicitly rather than through the config, because it is a
    dataframe and not configuration.  A factory that consumes a standard frame
    declares a ``frame`` parameter and receives it; the modalities that do not
    (Gaia, Voyager, simulated) simply do not declare one, so the same call site
    serves both.

    Args:
        cfg: Experiment configuration.
        dataset_cfg: Dataset configuration.  Must have a ``type`` field, or both
            ``observation_type`` and ``coordinate_kind``.
        system_of_bodies: The environment containing the bodies.
        frame: The dataset's standard observation frame, when it has one.

    Returns:
        Constructed observation dataset object (type depends on modality factory).

    Raises:
        ValueError: If no factory can be resolved for the config.

    Example:
        cfg = OmegaConf.create({'type': 'ground_ccd', 'file': 'data.csv', 'weight': 1.0})
        dataset = create_observation_dataset(cfg)
    """
    logger.debug("Creating observation dataset for '%s'", dataset_cfg.get("identifier", "?"))

    try:
        factory = resolve_factory(dataset_cfg)
    except ValueError as e:
        logger.error(f"Failed to find a factory for dataset config. Error: {e}")
        raise

    if "frame" in inspect.signature(factory).parameters:
        dataset, model_settings = factory(cfg, dataset_cfg, system_of_bodies, frame=frame)
    else:
        dataset, model_settings = factory(cfg, dataset_cfg, system_of_bodies)

    logger.debug(f"Successfully created observation dataset {dataset_cfg.get('identifier', '?')}")
    return dataset, model_settings
