import logging

import numpy as np
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig

from orbitdet.data.voyager_data import load_and_merge_voyager_tables
from orbitdet.observations.registry import register_dataset_factory
from orbitdet.transformations import convert_radec_frame

from .nsdb_helpers import set_ra_dec_columns

logger = logging.getLogger(__name__)


@register_dataset_factory("voyager")
def create_voyager_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, obs_model_setup.model_settings.ObservationModelSettings]:
    """Create a dataset from Voyager 2 observations of Triton.

    Args:
        cfg: relative CCD observation with differential RA/DEC coordinates configuration with
        necessary metadata.
        system_of_bodies: The environment containing the bodies for which to create the dataset.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings) for the relative CCD dataset.
    """
    logger.info(f"Creating Voyager observation dataset: {dataset_cfg.identifier}.")

    merged_data = load_and_merge_voyager_tables(cfg, dataset_cfg)

    receiver_name = dataset_cfg.observatory.name
    receiver_link_end = obs_model_setup.links.body_origin_link_end_id(receiver_name)

    # Convert RA/DEC columns to appropriate frame
    ra_column, dec_column = set_ra_dec_columns(merged_data)
    merged_data = convert_radec_frame(
        merged_data,
        ra_column,
        dec_column,
        input_frame=dataset_cfg.epoch_of_equinox,
        output_frame=cfg.global_frame_orientation,
    )

    valid_rows = merged_data[["epoch_TDB", ra_column, dec_column]].dropna()
    times = valid_rows["epoch_TDB"].to_numpy()

    if len(list(dataset_cfg.satellites.keys())) > 1:
        raise NotImplementedError("Multiple satellites in one Voyager file not supported yet.")

    target = list(dataset_cfg.satellites.keys())[0]
    link_ends = dict()
    link_ends[obs_model_setup.links.transmitter] = obs_model_setup.links.body_origin_link_end_id(
        target
    )
    link_ends[obs_model_setup.links.receiver] = receiver_link_end
    link_definition = obs_model_setup.links.LinkDefinition(link_ends)
    observation_model = obs_model_setup.model_settings.angular_position(link_definition)

    observation_set = obs.create_single_observation_set(
        obs_model_setup.model_settings.angular_position_type,
        link_ends,
        [
            np.deg2rad(row[[ra_column, dec_column]].to_numpy(dtype=float)).reshape(2, 1)
            for _, row in valid_rows.iterrows()
        ],
        times,
        obs_model_setup.links.LinkEndType.receiver,
    )

    logger.info(
        f"Voyager observation dataset {dataset_cfg.identifier} created with {len(valid_rows)} rows."
    )

    return (observation_set, observation_model)
