"""Absolute CCD NSDB observation dataset factory."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig

from orbitdet.transformations import convert_radec_frame

from .helpers import (
    add_observatory_to_SOB,
    convert_time_to_seconds_since_j2000_TDB,
)
from .nsdb_helpers import (
    group_rows_by_observatory,
    resolve_observatory_codes,
    set_iso_time_column,
    set_ra_dec_columns,
)
from .registry import register_dataset_factory

logger = logging.getLogger(__name__)


def _build_absolute_observation_set(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    data_file: pd.DataFrame,
    station_name: str,
    ra_column: str,
    dec_column: str,
) -> tuple[obs.SingleObservationSet, obs_model_setup.model_settings.ObservationModelSettings]:
    """Build a single absolute-angular observation set for one observatory station."""
    # Ensure observatory exists in the system of bodies
    add_observatory_to_SOB(cfg, system_of_bodies, station_name)

    # Convert times to seconds since J2000 epoch TDB for Tudat using station position
    convert_time_to_seconds_since_j2000_TDB(
        data_file, station_name, system_of_bodies, dataset_cfg.time_scale
    )

    valid_rows = data_file[["epoch_TDB", ra_column, dec_column]].dropna()
    times = valid_rows["epoch_TDB"].tolist()
    data = [
        np.deg2rad(row[[ra_column, dec_column]].to_numpy(dtype=float)).reshape(2, 1)
        for _, row in valid_rows.iterrows()
    ]

    # Setup link ends
    link_ends = dict()
    target = list(dataset_cfg.satellites.keys())[0]
    link_ends[obs_model_setup.links.transmitter] = obs_model_setup.links.body_origin_link_end_id(
        target
    )
    if str(dataset_cfg.center_of_frame).lower() == "geocentric":
        link_ends[obs_model_setup.links.receiver] = obs_model_setup.links.body_origin_link_end_id(
            "Earth"
        )
    else:
        link_ends[obs_model_setup.links.receiver] = (
            obs_model_setup.links.body_reference_point_link_end_id("Earth", station_name)
        )
    link_definition = obs_model_setup.links.LinkDefinition(link_ends)
    observation_model = obs_model_setup.model_settings.angular_position(link_definition)

    observation_set = obs.create_single_observation_set(
        obs_model_setup.model_settings.angular_position_type,
        link_ends,
        data,
        times,
        obs_model_setup.links.LinkEndType.receiver,
    )

    return (observation_set, observation_model)


@register_dataset_factory("absolute_absolute_CCD_nsdb")
@register_dataset_factory("absolute_absolute_photographic_nsdb")
def create_absolute_ccd_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, list[Any]]:
    """Create a dataset from absolute CCD observations.

    Args:
        cfg: Absolute CCD observation configuration with necessary metadata.
        system_of_bodies: The environment containing the bodies for which to create the dataset.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings) for the absolute CCD dataset.
    """
    logger.info(f"""Creating absolute CCD observation dataset: {dataset_cfg.identifier}.""")

    data_file = pd.read_csv(dataset_cfg.file, sep=r"\s+", header=None, comment="#", engine="python")
    # Set DataFrame column names from cfg.format mapping (index -> name).
    # Ensure the mapping is applied deterministically by sorting keys numerically
    # when possible, and overlaying names into the default positional columns.
    fmt = dict(dataset_cfg.format_columns)
    col_names = list(data_file.columns)

    def _keyfunc(k):
        try:
            return int(k)
        except Exception:
            return str(k)

    for index in sorted(fmt.keys(), key=_keyfunc):
        name = fmt.get(index, fmt.get(str(index), None))
        # Try interpret index as 1-based first (NSDB formats often number from 1)
        pos = None
        try:
            pos = int(index) - 1
        except Exception:
            try:
                pos = int(index)
            except Exception:
                pos = None

        if pos is not None and 0 <= pos < len(col_names):
            col_names[pos] = name if name is not None else col_names[pos]

    data_file.columns = col_names

    # Station-independent preprocessing on the full dataframe.
    set_iso_time_column(data_file)
    ra_column, dec_column = set_ra_dec_columns(data_file)
    data_file = convert_radec_frame(
        data_file,
        ra_column,
        dec_column,
        input_frame=dataset_cfg.epoch_of_equinox,
        output_frame=cfg.global_frame_orientation,
    )

    # Fail if file contains multiple satellites, not implemented yet
    if len(list(dataset_cfg.satellites.keys())) > 1:
        raise NotImplementedError("Multiple satellites in one NSDB file not supported yet.")

    observatories = list(dataset_cfg.observatory)
    telescope_index = dict(dataset_cfg.get("telescope_index", {}) or {})
    observatory_codes = resolve_observatory_codes(data_file, observatories, telescope_index)

    observation_sets = []
    observation_models = []
    for station_name, group in group_rows_by_observatory(data_file, observatory_codes):
        observation_set, observation_model = _build_absolute_observation_set(
            cfg,
            dataset_cfg,
            system_of_bodies,
            group,
            station_name,
            ra_column,
            dec_column,
        )
        observation_sets.append(observation_set)
        observation_models.append(observation_model)

    if len(observation_sets) == 1:
        return observation_sets[0], observation_models[0]

    collection = obs.ObservationCollection(observation_sets)
    return collection, observation_models
