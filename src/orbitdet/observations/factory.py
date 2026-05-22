"""Central factory for creating observation datasets."""

import logging

import tudatpy.dynamics.environment as env
from omegaconf import DictConfig, OmegaConf

from .registry import get_factory

logger = logging.getLogger(__name__)


def create_observation_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple:
    """Create an observation dataset from configuration.

    This is the central dispatcher that:
    1. Accepts a dataset config (Hydra DictConfig or structured dataclass)
    2. Looks up the correct factory based on cfg.type
    3. Invokes the factory to construct the observation dataset object

    Args:
        cfg: Dataset configuration. Must have a 'type' field.

    Returns:
        Constructed observation dataset object (type depends on modality factory).

    Raises:
        ValueError: If cfg.type is not registered.
        KeyError: If cfg does not have a 'type' field.

    Example:
        cfg = OmegaConf.create({'type': 'ground_ccd', 'file': 'data.csv', 'weight': 1.0})
        dataset = create_observation_dataset(cfg)
    """
    # Extract type from config
    dataset_type = OmegaConf.select(dataset_cfg, "type")
    if not dataset_type:
        raise ValueError("Dataset config must have a 'type' field")

    logger.debug(f"Creating observation dataset of type '{dataset_type}'")

    # Get the factory for this type
    try:
        factory = get_factory(dataset_type)

        # Invoke the factory
        dataset, model_settings = factory(cfg, dataset_cfg, system_of_bodies)

        logger.debug(f"Successfully created observation dataset of type '{dataset_type}'")
        return dataset, model_settings

    except ValueError as e:
        logger.error(f"Failed to find factory for dataset type '{dataset_type}'. Error: {e}")
        raise
