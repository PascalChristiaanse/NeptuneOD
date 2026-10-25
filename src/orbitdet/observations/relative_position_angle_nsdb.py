"""NSDB factory for relative position-angle and separation observations."""

import logging
from typing import Any

import pandas as pd
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
import tudatpy.estimation.observations_setup.ancillary_settings as ancillary
from omegaconf import DictConfig

from .helpers import add_observatory_to_SOB, convert_time_to_seconds_since_j2000_TDB
from .nsdb_helpers import (
    group_rows_by_observatory,
    resolve_observatory_codes,
    set_iso_time_column,
    set_position_angle_separation_columns,
)
from .registry import register_dataset_factory

logger = logging.getLogger(__name__)


def _format_columns(dataframe: pd.DataFrame, format_columns: dict) -> None:
    names = list(dataframe.columns)
    for index, name in sorted(format_columns.items(), key=lambda item: int(item[0])):
        position = int(index) - 1
        if 0 <= position < len(names):
            names[position] = name
    dataframe.columns = names


def _reference_frame(dataset_cfg: DictConfig):
    equinox = str(dataset_cfg.get("epoch_of_equinox", "J2000")).strip().lower()
    if equinox in {"date", "of date"}:
        return ancillary.PositionAngleReferenceFrame.mean_of_date_iau_2006
    if equinox.startswith("b1950"):
        return ancillary.PositionAngleReferenceFrame.b1950
    return ancillary.PositionAngleReferenceFrame.j2000


def _observatories(dataset_cfg: DictConfig) -> list:
    value = dataset_cfg.observatory
    if isinstance(value, DictConfig):
        return [value]
    return list(value)


def _build_set(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    dataframe: pd.DataFrame,
    station_name: str,
    angle_column: str,
    separation_column: str,
) -> tuple[list[obs.SingleObservationSet], list[Any]]:
    add_observatory_to_SOB(cfg, system_of_bodies, station_name)
    convert_time_to_seconds_since_j2000_TDB(
        dataframe, station_name, system_of_bodies, dataset_cfg.time_scale
    )
    target = list(dataset_cfg.satellites.keys())[0]
    planet = dataset_cfg.planet.name
    link_ends = {
        obs_model_setup.links.transmitter: obs_model_setup.links.body_origin_link_end_id(planet),
        obs_model_setup.links.transmitter2: obs_model_setup.links.body_origin_link_end_id(target),
    }
    center_of_frame = str(dataset_cfg.get("center_of_frame", "topocentre")).lower()
    if center_of_frame in {"geocentric", "geocentre", "geoocentre"}:
        link_ends[obs_model_setup.links.receiver] = obs_model_setup.links.body_origin_link_end_id(
            "Earth"
        )
    else:
        link_ends[obs_model_setup.links.receiver] = (
            obs_model_setup.links.body_reference_point_link_end_id("Earth", station_name)
        )

    link_definition = obs_model_setup.links.LinkDefinition(link_ends)
    ancillary_settings = ancillary.position_angle_ancillary_settings(
        reference_frame=_reference_frame(dataset_cfg),
        use_stellar_aberration=bool(dataset_cfg.get("use_stellar_aberration", True)),
    )
    observation_sets = []
    models = []

    angle_rows = dataframe[["epoch_TDB", angle_column]].dropna()
    if not angle_rows.empty:
        observation_sets.append(
            obs.create_single_observation_set(
                obs_model_setup.model_settings.position_angle_type,
                link_ends,
                [
                    row[[angle_column]].to_numpy(dtype=float).reshape(1, 1)
                    for _, row in angle_rows.iterrows()
                ],
                angle_rows["epoch_TDB"].tolist(),
                obs_model_setup.links.LinkEndType.receiver,
                ancillary_settings,
            )
        )
        models.append(obs_model_setup.model_settings.position_angle(link_definition))

    separation_rows = dataframe[["epoch_TDB", separation_column]].dropna()
    if not separation_rows.empty:
        observation_sets.append(
            obs.create_single_observation_set(
                obs_model_setup.model_settings.separation_distance_type,
                link_ends,
                [
                    row[[separation_column]].to_numpy(dtype=float).reshape(1, 1)
                    for _, row in separation_rows.iterrows()
                ],
                separation_rows["epoch_TDB"].tolist(),
                obs_model_setup.links.LinkEndType.receiver,
                ancillary_settings,
            )
        )
        models.append(obs_model_setup.model_settings.separation_distance(link_definition))

    return observation_sets, models


@register_dataset_factory("relative_position_angle_and_separation_micrometric_nsdb")
@register_dataset_factory("relative_position_angle_and_separation_photographs_nsdb")
@register_dataset_factory("relative_position_angle_and_separation_photographic_nsdb")
@register_dataset_factory("relative_position_angle_and_separation_photograpihc_nsdb")
@register_dataset_factory("relative_position_angle_and_separation_no inf_nsdb")
@register_dataset_factory("relative_position_angle_and_distance_micrometric, photographs_nsdb")
@register_dataset_factory("relative_Position_angle,_Distance_visual_nsdb")
@register_dataset_factory("1919-1923_position_angle_and_separation_micrometric_nsdb")
def create_relative_position_angle_nsdb_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, list[Any]]:
    """Create combined position-angle/separation observations from an NSDB file."""
    dataframe = pd.read_csv(
        dataset_cfg.file, sep=r"\s+", header=None, comment="#", engine="python"
    )
    _format_columns(dataframe, dict(dataset_cfg.format_columns))
    set_iso_time_column(dataframe)
    angle_column, separation_column = set_position_angle_separation_columns(dataframe)

    observatories = _observatories(dataset_cfg)
    telescope_index = dict(dataset_cfg.get("telescope_index", {}) or {})
    codes = resolve_observatory_codes(dataframe, observatories, telescope_index)
    sets = []
    models = []
    for station_name, group in group_rows_by_observatory(dataframe, codes):
        observation_sets, observation_models = _build_set(
            cfg,
            dataset_cfg,
            system_of_bodies,
            group,
            station_name,
            angle_column,
            separation_column,
        )
        sets.extend(observation_sets)
        models.extend(observation_models)
    if len(sets) == 1:
        return sets[0], models[0]
    return obs.ObservationCollection(sets), models