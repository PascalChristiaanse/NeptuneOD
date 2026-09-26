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

    Supports two modes:

    **Continuous mode** (default):
    Uses ``start_date_observation_period`` / ``end_date_observation_period`` and
    ``cadence`` to produce evenly spaced observations over the whole interval.

    **Burst mode** (when ``bursts`` list is present):
    Each burst is a dict with:

    - ``start`` (str): ISO start date of the burst
    - ``duration_hours`` (float): how long the burst lasts [h]
    - ``cadence`` (float): time between observations within the burst [s]

    All burst observation times are concatenated and sorted.

    Args:
        cfg: Top-level configuration.
        dataset_cfg: Dataset-specific configuration.
        system_of_bodies: The environment containing the bodies.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings).
    """
    logger.info(
        f"Creating simulated observation dataset for "
        f"{dataset_cfg.target} w.r.t. {dataset_cfg.observer}."
    )

    # -- Build observation times ------------------------------------------------
    bursts = dataset_cfg.get("bursts", None)
    if bursts is not None and len(bursts) > 0:
        # Burst mode: several short dense windows
        obs_time_list: list[np.ndarray] = []
        for b in bursts:
            b_start = iso_string_to_epoch(b.start)
            b_duration_s = b.duration_hours * 3600.0
            b_cadence = b.cadence
            b_times = np.arange(b_start, b_start + b_duration_s, b_cadence)
            obs_time_list.append(b_times)
            logger.info(
                "Burst at %s: %d observations, cadence %.0f s, duration %.1f h",
                b.start, len(b_times), b_cadence, b.duration_hours,
            )
        observation_times = np.sort(np.concatenate(obs_time_list))
        logger.info(
            "Total observations from %d bursts: %d",
            len(bursts), len(observation_times),
        )
    else:
        # Continuous mode: uniform cadence over full interval
        start_epoch = iso_string_to_epoch(dataset_cfg.start_date_observation_period)
        end_epoch = iso_string_to_epoch(dataset_cfg.end_date_observation_period)
        observation_times = np.linspace(
            start_epoch,
            end_epoch,
            int(np.ceil((end_epoch - start_epoch) / dataset_cfg.cadence)) + 1,
        )
        logger.info(
            "Generating observations at %d epochs from %s to %s with cadence %.0f s.",
            len(observation_times),
            dataset_cfg.start_date_observation_period,
            dataset_cfg.end_date_observation_period,
            dataset_cfg.cadence,
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
