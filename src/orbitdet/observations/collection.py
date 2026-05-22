"""Observation collection builder."""

import logging
from typing import Any

import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, OmegaConf

from .factory import create_observation_dataset

logger = logging.getLogger(__name__)


def create_observation_collection(
    cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, list[Any]]:
    """Build an observation collection from multiple dataset configs.

    Iterates through dataset configurations in a collection, dispatches each through
    the central factory, and aggregates the resulting observation sets.

    Args:
        cfg: Hydra config with 'datasets' list. Each entry should be a
                       dataset config with a 'type' field.
        system_of_bodies: The environment containing the bodies for which to create observations.

    Returns:
        List of observation dataset objects. The structure depends on the registered
        factories and Tudatpy's actual observation API.

    Raises:
        ValueError: If collection_cfg does not have 'datasets' key or if any
                   dataset factory fails.

    Example:
        collection_cfg = OmegaConf.create({
            'datasets': [
                {'type': 'ground_ccd', 'file': 'data1.csv', 'weight': 1.0},
                {'type': 'space_ccd', 'file': 'data2.csv', 'weight': 0.8},
            ]
        })
        observations = create_observation_collection(collection_cfg)
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    datasets = OmegaConf.select(cfg, "datasets")
    if datasets is None:
        raise ValueError("Collection config must have a 'datasets' list")

    logger.info(f"Creating observation collection with {len(datasets)} dataset(s)")

    observation_sets = []
    model_setting = []
    for idx, (set_name, dataset_cfg) in enumerate(datasets.items()):
        try:
            logger.debug(f"Creating dataset {set_name} ({idx + 1}/{len(datasets)})")
            dataset, model_settings = create_observation_dataset(cfg, dataset_cfg, system_of_bodies)
            observation_sets.append(dataset)
            model_setting.append(model_settings)
            logger.debug(f"Successfully created dataset {set_name} ({idx + 1}/{len(datasets)})")
        except Exception as e:
            logger.error(
                f"Failed to create dataset {idx + 1}/{len(datasets)}: {dataset_cfg}. Error: {e}"
            )
            raise

    paired_sets = [
        (dataset, settings)
        for dataset, settings in zip(observation_sets, model_setting)
        if dataset is not None and settings is not None
    ]
    observation_sets = [dataset for dataset, _ in paired_sets]
    model_setting = [settings for _, settings in paired_sets]
    logger.info(f"Successfully created observation collection with {len(observation_sets)} set(s)")
    observation_collection = obs.merge_observation_collections(observation_sets)

    return observation_collection, model_setting
