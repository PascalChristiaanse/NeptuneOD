"""Gaia observation dataset factory.

Creates a Tudat observation collection from Gaia astrometric observations of a
solar-system body (e.g. Triton).  The raw archive pull is handled by
:class:`~orbitdet.data.gaia.GaiaQuery`, which caches the pull to disk to reduce
internet traffic.
"""

import logging

import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig

from orbitdet.data.gaia_data import GaiaQuery
from orbitdet.observations.registry import register_dataset_factory

logger = logging.getLogger(__name__)


@register_dataset_factory("gaia")
def create_gaia_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, obs_model_setup.model_settings.ObservationModelSettings]:
    """Create a dataset from Gaia astrometric observations of a target body.

    Args:
        cfg: Experiment configuration with necessary metadata.
        dataset_cfg: Dataset configuration.  Must contain a ``source_id`` (or
            ``mpc_numbers``) and a ``cache_file`` path.
        system_of_bodies: The environment containing the bodies for which to
            create the dataset.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings) for the Gaia
        dataset.
    """
    logger.info(f"Creating Gaia observation dataset: {dataset_cfg.identifier}.")

    target_name = str(getattr(dataset_cfg, "target", "Triton"))
    source_ids = getattr(dataset_cfg, "source_ids", None)
    mpc_numbers = getattr(dataset_cfg, "mpc_numbers", None)
    cache_file = getattr(dataset_cfg, "cache_file", None)
    epoch_of_equinox = str(getattr(dataset_cfg, "epoch_of_equinox", "ICRS"))
    filter_outcomes = bool(getattr(dataset_cfg, "filter_outcomes", True))
    correct_photocenter = bool(getattr(dataset_cfg, "correct_photocenter", False))
    light_deflection = getattr(dataset_cfg, "light_deflection", ("Sun",))
    # Normalise light_deflection to a (possibly empty) list of body names.
    if isinstance(light_deflection, str):
        light_deflection = [light_deflection] if light_deflection.strip() else []
    else:
        try:
            light_deflection = list(light_deflection)
        except TypeError:
            raise ValueError(
                f"Gaia dataset {dataset_cfg.identifier}: 'light_deflection' must be a "
                "list/tuple of body names or a string, got "
                f"{type(light_deflection).__name__}."
            )
    diameter = float(getattr(dataset_cfg, "diameter", 0.0) or 0.0)

    query = GaiaQuery()
    if source_ids is not None:
        query.retrieve_data(
            source_ids=source_ids, cache_file=cache_file, filter_outcomes=filter_outcomes
        )
    elif mpc_numbers is not None:
        query.retrieve_data(
            mpc_numbers=mpc_numbers, cache_file=cache_file, filter_outcomes=filter_outcomes
        )
    else:
        raise ValueError(
            f"Gaia dataset {dataset_cfg.identifier} must specify either "
            f"'source_ids' or 'mpc_numbers'."
        )

    # Apply astrometric corrections (light deflection and/or photocenter offset)
    # to the observed RA/DEC so they refer to the centre of mass of the target.
    if correct_photocenter or light_deflection:
        query.correct_observations(
            target_name,
            system_of_bodies,
            light_deflection=light_deflection,
            correct_photocenter=correct_photocenter,
            diameter=diameter,
        )

    observation_collection = query.to_tudat(
        system_of_bodies,
        target_name=target_name,
        input_frame=epoch_of_equinox,
        output_frame=cfg.global_frame_orientation,
    )

    # Build the observation model settings for the simulator
    link_ends = dict()
    link_ends[obs_model_setup.links.transmitter] = obs_model_setup.links.body_origin_link_end_id(
        target_name
    )
    link_ends[obs_model_setup.links.receiver] = obs_model_setup.links.body_origin_link_end_id(
        "Gaia"
    )
    link_definition = obs_model_setup.links.LinkDefinition(link_ends)
    observation_model = obs_model_setup.model_settings.angular_position(link_definition)

    logger.info(
        f"Gaia observation dataset {dataset_cfg.identifier} created with "
        f"{len(query.observation_table)} rows."
    )

    return (observation_collection, observation_model)