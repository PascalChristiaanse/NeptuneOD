import logging

import numpy as np
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations_setup as obs_setup
from omegaconf import DictConfig

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)


def generate_observations(
    cfg: DictConfig, ctx: RuntimeContext, system_of_bodies: env.SystemOfBodies
):
    """Generate observations based on the configuration settings."""

    observation_times = np.linspace(
        ctx.start_epoch.to_float(),
        ctx.end_epoch.to_float(),
        int(
            np.ceil(
                (ctx.end_epoch.to_float() - ctx.start_epoch.to_float()) / cfg.observations.cadence
            )
        )
        + 1,
    )
    observation_times = np.arange(
        ctx.start_epoch.to_float() + cfg.observations.cadence,
        ctx.end_epoch.to_float() - cfg.observations.cadence,
        cfg.observations.cadence,
    )

    observation_times = np.array([t for t in observation_times])

    logger.info(
        f"""Generating observations at {len(observation_times)} epochs from {ctx.start_epoch}"""
        """to {ctx.end_epoch} with cadence {cfg.observations.cadence} seconds."""
    )

    observation_models = []
    observation_simulation_settings = []
    for body in cfg.bodies_to_propagate.keys():
        # Setup link ends
        link_ends = dict()
        link_ends[obs_model_setup.links.observed_body] = (
            obs_model_setup.links.body_origin_link_end_id(body)
        )
        link_ends[obs_model_setup.links.observer] = obs_model_setup.links.body_origin_link_end_id(
            "Neptune"
        )
        link_definition = obs_model_setup.links.LinkDefinition(link_ends)

        for observable_type in cfg.observations.type:
            match observable_type:
                case "relative_position":
                    observation_models.append(
                        obs_model_setup.model_settings.relative_cartesian_position(link_definition)
                    )
                    observation_simulation_settings.append(
                        obs_setup.observations_simulation_settings.tabulated_simulation_settings(
                            obs_model_setup.model_settings.relative_position_observable_type,
                            link_definition,
                            observation_times,
                            reference_link_end_type=obs_model_setup.links.LinkEndType.observed_body,
                        )
                    )  # estimation_setup.observation.observed_body

    # Create observation simulators
    ephemeris_observation_simulators = (
        obs_setup.observations_simulation_settings.create_observation_simulators(
            observation_models, system_of_bodies
        )
    )

    # Get ephemeris states as ObservationCollection
    print("Checking spice for position pseudo observations...")
    simulated_pseudo_observations = obs_setup.observations_wrapper.simulate_observations(
        observation_simulation_settings,
        ephemeris_observation_simulators,
        system_of_bodies,
    )

    return simulated_pseudo_observations, observation_models
