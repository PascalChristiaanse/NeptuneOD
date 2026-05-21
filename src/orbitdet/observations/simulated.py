"""Simulated observation dataset factory."""

import logging

import numpy as np
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
import tudatpy.estimation.observations_setup as obs_setup
from omegaconf import DictConfig
from tudatpy.astro.time_representation import iso_string_to_epoch

from .registry import register_dataset_factory

logger = logging.getLogger(__name__)


@register_dataset_factory("simulated")
def create_simulated_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, obs_model_setup.model_settings.ObservationModelSettings]:
    """Create a simulated observation dataset.

    This factory generates synthetic observations from existing propagation
    results or ephemeris data. Allows hybrid experiments mixing real and
    simulated observations in a single collection.

    Args:
        cfg: Simulated observation configuration with observable types and cadence.
        system_of_bodies: The environment containing the bodies for which to simulate observations.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings) for the simulated dataset.
    Example:
        cfg = OmegaConf.create({
            'type': 'simulated',
            'observable_type': 'relative_cartesian_position',
            'target': 'Triton',
            'observer': 'Neptune',
            'cadence': 3600,  # seconds
            'start_date_observation_period': '2025-01-01T00:00:00',
            'end_date_observation_period': '2025-01-10T00:00:00',
            'noise_sigma': 100.0,  # meters
        })
        dataset = create_simulated_dataset(cfg)
    """
    logger.info(
        f"""Creating simulated observation dataset with cadence={dataset_cfg.cadence} """
        f"""for {dataset_cfg.target} w.r.t. {dataset_cfg.observer}."""
    )

    start_epoch = iso_string_to_epoch(dataset_cfg.start_date_observation_period)
    end_epoch = iso_string_to_epoch(dataset_cfg.end_date_observation_period)

    observation_times = np.linspace(
        start_epoch,
        end_epoch,
        int(np.ceil((end_epoch - start_epoch) / dataset_cfg.cadence)) + 1,
    )

    logger.info(
        f"""Generating observations at {len(observation_times)} epochs """
        f"""from {dataset_cfg.start_date_observation_period}"""
        f"""to {dataset_cfg.end_date_observation_period} with cadence {dataset_cfg.cadence} seconds."""
    )

    # Setup link ends
    link_ends = dict()
    link_ends[obs_model_setup.links.observed_body] = obs_model_setup.links.body_origin_link_end_id(
        dataset_cfg.target
    )
    link_ends[obs_model_setup.links.observer] = obs_model_setup.links.body_origin_link_end_id(
        dataset_cfg.observer
    )
    link_definition = obs_model_setup.links.LinkDefinition(link_ends)

    match dataset_cfg.observable_type:
        case "relative_cartesian_position":
            # Create observation model
            observation_model = obs_model_setup.model_settings.relative_cartesian_position(
                link_definition
            )
            # Create simulation settings
            single_setting = (
                obs_setup.observations_simulation_settings.tabulated_simulation_settings(
                    obs_model_setup.model_settings.relative_position_observable_type,
                    link_definition,
                    observation_times,
                    reference_link_end_type=obs_model_setup.links.LinkEndType.observed_body,
                )
            )
            obs_setup.random_noise.add_gaussian_noise_to_observable(
                [single_setting],
                dataset_cfg.noise_sigma,
                obs_model_setup.model_settings.relative_position_observable_type,
            )
    # Create observation simulators
    ephemeris_observation_simulators = (
        obs_setup.observations_simulation_settings.create_observation_simulators(
            [observation_model], system_of_bodies
        )
    )

    # Get ephemeris states as ObservationCollection
    simulated_pseudo_observations = obs_setup.observations_wrapper.simulate_observations(
        [single_setting],
        ephemeris_observation_simulators,
        system_of_bodies,
    )

    return (simulated_pseudo_observations, observation_model)
