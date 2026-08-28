"""Relative X/Y NSDB observation dataset converter and factory.

Handles NSDB datasets that store the satellite position as *relative X/Y offsets*
(in arcseconds) with respect to the planet, e.g.:

- ``relative_X,_Y_photographic_nsdb`` (nm0008, nm0009, nm0010)
- ``relative_X,Y_CCD_nsdb`` (nm0003, nm0004)

The raw files differ in how the observation time is encoded (Julian Date vs.
year/month/day) and whether a satellite-number column is present, but the
relative coordinates are always X/Y in arcseconds.  The converter normalises
these into a common dataframe (ISO time + relative position in radians), and the
factory builds a Tudat relative-angular-position observation set per observatory.

The X/Y offsets are interpreted as *standard (gnomonic tangent-plane) coordinates*
about the topocentric position of the planet, as defined in e.g. "Satellite Orbits:
Models, Methods and Applications" (Montenbruck & Gill).  Before building the
observation set they are projected back through the exact gnomonic inverse into
relative right-ascension/declination offsets, using the topocentric RA/Dec of the
planet (from the system-of-bodies ephemerides) as the tangent point at each epoch.
This guarantees the measured vector is a true relative RA/Dec, consistent with what
Tudat's ``relative_angular_position`` model computes from the same geometry.
"""

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
    normalize_observatory_code,
)
from .nsdb_helpers import (
    group_rows_by_observatory,
    resolve_observatory_codes,
    set_iso_time_column,
    set_relative_position_columns,
)
from .registry import register_dataset_factory

logger = logging.getLogger(__name__)

# TudatPy in this environment binds relative_angular_position_type to angular_position_type.
# Use the underlying C++ enum value directly so relative-angular sets are tagged correctly.
RELATIVE_ANGULAR_POSITION_TYPE = obs_model_setup.model_settings.ObservableType(9)


def _apply_format_columns(data_file: pd.DataFrame, format_columns: dict) -> pd.DataFrame:
    """Overlay the NSDB format-column mapping (index -> name) onto a positional DataFrame.

    Args:
        data_file: The raw DataFrame read without a header.
        format_columns: Mapping from 1-based column index to column name.

    Returns:
        The DataFrame with columns renamed in place.
    """
    fmt = dict(format_columns)
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
    return data_file


def convert_relative_xy_data(
    cfg: DictConfig, dataset_cfg: DictConfig
) -> tuple[pd.DataFrame, str, str]:
    """Read and normalise a relative X/Y NSDB data file.

    Applies the format-column mapping, infers the observation time (JD or
    year/month/day), and converts the relative X/Y offsets (in arcseconds) to
    radians.  The returned dataframe has ``iso_time``, ``relative_position_x``
    and ``relative_position_y`` columns.

    Args:
        cfg: Top-level configuration (used for observatory metadata).
        dataset_cfg: Dataset configuration with ``file`` and ``format_columns``.

    Returns:
        Tuple of (dataframe, relative_x_column, relative_y_column).
    """
    data_file = pd.read_csv(
        dataset_cfg.file, sep=r"\s+", header=None, comment="#", engine="python"
    )
    _apply_format_columns(data_file, dataset_cfg.format_columns)

    # Station-independent preprocessing on the full dataframe.
    set_iso_time_column(data_file)
    relative_x_column, relative_y_column = set_relative_position_columns(data_file)
    return data_file, relative_x_column, relative_y_column


def _resolve_relative_to_body_name(dataset_cfg: DictConfig) -> str:
    """Resolve the name of the body the relative coordinates are referenced to.

    NSDB encodes the reference body in ``relative_to.designation`` in several
    ways, e.g. ``"planet"``, ``"8 - Neptune"`` or ``"N1-Triton"``.  This returns
    the plain body name (e.g. ``"Neptune"``) that matches the system-of-bodies.
    """
    designation = str(dataset_cfg.relative_to.designation).strip()

    if designation.lower() == "planet":
        return str(dataset_cfg.planet.name)

    if " - " in designation:
        return designation.split(" - ")[1].strip()

    if "-" in designation:
        return designation.split("-")[-1].strip()

    return designation


def _topocentric_planet_direction(
    system_of_bodies: env.SystemOfBodies,
    planet_name: str,
    station_name: str,
    epoch_tdb: float,
) -> np.ndarray:
    """Unit direction from the observatory to the planet centre in the global frame.

    Uses the system-of-bodies ephemerides for the planet and Earth centres, and
    Earth's rotation model plus the ground-station body-fixed position for the
    observatory location.  This matches the geometry used by Tudat's
    ``relative_angular_position`` model (receiver = topocentric observatory).

    Args:
        system_of_bodies: The system containing the bodies.
        planet_name: Name of the reference body (e.g. ``"Neptune"``).
        station_name: Normalized observatory code (ground-station name).
        epoch_tdb: Observation time in seconds since J2000 TDB.

    Returns:
        3-vector unit direction from the observatory toward the planet centre, in
        the global frame orientation.
    """
    earth = system_of_bodies.get_body("Earth")
    planet_body = system_of_bodies.get_body(planet_name)
    ground_station = earth.get_ground_station(station_name)

    station_body_fixed = np.ravel(
        ground_station.station_state.get_cartesian_position(epoch_tdb)
    )
    rotation = np.array(earth.rotation_model.body_fixed_to_inertial_rotation(epoch_tdb))
    station_inertial = np.ravel(earth.ephemeris.cartesian_position(epoch_tdb)) + (
        rotation @ station_body_fixed
    )

    planet_inertial = np.ravel(planet_body.ephemeris.cartesian_position(epoch_tdb))
    direction = planet_inertial - station_inertial
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.array([np.nan, np.nan, np.nan])
    return direction / norm


def _standard_to_relative_radec(
    xi_rad, eta_rad, ra0, dec0
) -> tuple[float, float]:
    """Project gnomonic standard coordinates into relative RA/Dec offsets.

    Given standard (tangent-plane) coordinates ``(xi, eta)`` measured about a
    reference direction ``(ra0, dec0)``, return the exact relative offsets
    ``(Delta_alpha = alpha - alpha_0,  Delta_delta = delta - delta_0)`` in radians.

    This is the inverse of the gnomonic projection of the standard coordinates
    (see e.g. "Satellite Orbits: Models, Methods and Applications", Montenbruck
    & Gill):

        Delta_alpha = atan( xi / (cos(delta0) - eta * sin(delta0)) )
        sin(delta)  = (sin(delta0) + eta * cos(delta0)) / sqrt(1 + xi^2 + eta^2)

    Accepts scalars or equal-length arrays for each argument; array input is
    evaluated element-wise and returns arrays.

    Args:
        xi_rad: Standard X coordinate (along increasing RA) in radians.
        eta_rad: Standard Y coordinate (along increasing Dec) in radians.
        ra0: Reference right ascension in radians.
        dec0: Reference declination in radians.

    Returns:
        Tuple ``(Delta_alpha, Delta_delta)`` in radians (scalars or arrays).
    """
    cos_d0 = np.cos(dec0)
    sin_d0 = np.sin(dec0)

    del_alpha = np.arctan2(xi_rad, cos_d0 - eta_rad * sin_d0)
    sin_dec = (sin_d0 + eta_rad * cos_d0) / np.sqrt(1.0 + xi_rad ** 2 + eta_rad ** 2)
    del_dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0)) - dec0
    return del_alpha, del_dec


def _build_relative_observation_set(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    data_file: pd.DataFrame,
    station_name: str,
    relative_x_column: str,
    relative_y_column: str,
) -> tuple[obs.SingleObservationSet, obs_model_setup.model_settings.ObservationModelSettings]:
    """Build a single relative-angular observation set for one observatory station."""
    # Ensure observatory exists in the system of bodies
    add_observatory_to_SOB(cfg, system_of_bodies, station_name)

    # Convert times to seconds since J2000 epoch TDB for Tudat using station position
    convert_time_to_seconds_since_j2000_TDB(
        data_file, station_name, system_of_bodies, dataset_cfg.time_scale
    )

    # The X/Y offsets are gnomonic standard-coordinate offsets about the
    # topocentric position of the planet, anchored to the mean equator and equinox
    # of the observation date (``epoch_of_equinox: Date``, matching the absolute
    # CCD factory).  For each epoch we:
    #   1. form the topocentric planet direction in the global frame,
    #   2. express it as RA/Dec in the Date frame (the tangent-plane reference),
    #   3. obtain the satellite RA/Dec in the Date frame via the exact gnomonic
    #      inverse,
    #   4. rotate that same direction back into the global frame, and
    #   5. take the relative spherical offset (Delta_alpha, Delta_delta) in the
    #      global frame - the exact convention Tudat's relative_angular_position
    #      model uses when it reads RA/Dec off the global-frame state vectors.
    planet_name = _resolve_relative_to_body_name(dataset_cfg)
    station_code = normalize_observatory_code(station_name)
    global_frame = cfg.global_frame_orientation

    valid_rows = data_file[["epoch_TDB", relative_x_column, relative_y_column]].dropna()
    times = valid_rows["epoch_TDB"].tolist()
    epochs = valid_rows["epoch_TDB"].to_numpy(dtype=float)
    xi = valid_rows[relative_x_column].to_numpy(dtype=float)
    eta = valid_rows[relative_y_column].to_numpy(dtype=float)

    # Planet topocentric direction and its spherical angles in the global frame.
    u_planet = np.array(
        [
            _topocentric_planet_direction(system_of_bodies, planet_name, station_code, t)
            for t in epochs
        ]
    )
    valid = np.isfinite(u_planet).all(axis=1)
    if not valid.all():
        logger.warning(
            f"Skipped {int((~valid).sum())} rows with non-finite topocentric planet direction "
            f"for station {station_name}."
        )
    times = [t for t, ok in zip(times, valid) if ok]
    xi = xi[valid]
    eta = eta[valid]
    epochs = epochs[valid]
    u_planet = u_planet[valid]

    ra_p_global = np.arctan2(u_planet[:, 1], u_planet[:, 0])
    dec_p_global = np.arcsin(u_planet[:, 2])

    # Reference RA/Dec of the planet in the Date (mean equator/equinox of date) frame.
    planet_date = pd.DataFrame(
        {
            "ra_deg": np.rad2deg(ra_p_global),
            "dec_deg": np.rad2deg(dec_p_global),
            "epoch_TDB": epochs,
        }
    )
    planet_date = convert_radec_frame(
        planet_date,
        "ra_deg",
        "dec_deg",
        input_frame=global_frame,
        output_frame="Date",
        time_column="epoch_TDB",
    )
    ra0 = np.deg2rad(planet_date["ra_deg"].to_numpy(dtype=float))
    dec0 = np.deg2rad(planet_date["dec_deg"].to_numpy(dtype=float))

    # Satellite absolute RA/Dec in the Date frame from the gnomonic inverse.
    del_alpha_date, del_dec_date = _standard_to_relative_radec(xi, eta, ra0, dec0)
    satellite_date = pd.DataFrame(
        {
            "ra_deg": np.rad2deg(ra0 + del_alpha_date),
            "dec_deg": np.rad2deg(dec0 + del_dec_date),
            "epoch_TDB": epochs,
        }
    )
    satellite_date = convert_radec_frame(
        satellite_date,
        "ra_deg",
        "dec_deg",
        input_frame="Date",
        output_frame=global_frame,
        time_column="epoch_TDB",
    )
    ra_s_global = np.deg2rad(satellite_date["ra_deg"].to_numpy(dtype=float))
    dec_s_global = np.deg2rad(satellite_date["dec_deg"].to_numpy(dtype=float))

    # Relative spherical offset in the global frame (the model's convention).
    del_alpha = np.arctan2(
        np.sin(ra_s_global - ra_p_global), np.cos(ra_s_global - ra_p_global)
    )
    del_dec = dec_s_global - dec_p_global

    data = [np.array([[a], [d]]) for a, d in zip(del_alpha.tolist(), del_dec.tolist())]

    # Setup link ends
    link_ends = dict()
    target = list(dataset_cfg.satellites.keys())[0]
    relative_to = _resolve_relative_to_body_name(dataset_cfg)
    link_ends[obs_model_setup.links.transmitter] = obs_model_setup.links.body_origin_link_end_id(
        relative_to
    )
    link_ends[obs_model_setup.links.transmitter2] = obs_model_setup.links.body_origin_link_end_id(
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
    observation_model = obs_model_setup.model_settings.relative_angular_position(link_definition)

    observation_set = obs.create_single_observation_set(
        RELATIVE_ANGULAR_POSITION_TYPE,
        link_ends,
        data,
        times,
        obs_model_setup.links.LinkEndType.receiver,
    )

    return (observation_set, observation_model)


@register_dataset_factory("relative_X,_Y_photographic_nsdb")
@register_dataset_factory("relative_X,Y_CCD_nsdb")
def create_relative_xy_nsdb_dataset(
    cfg: DictConfig, dataset_cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, list[Any]]:
    """Create a dataset from relative X/Y NSDB observations.

    Args:
        cfg: Relative X/Y observation configuration with necessary metadata.
        system_of_bodies: The environment containing the bodies for which to create the dataset.

    Returns:
        Tuple of (ObservationCollection, ObservationModelSettings) for the relative X/Y dataset.
    """
    logger.info(
        f"""Creating relative X/Y observation dataset: {dataset_cfg.identifier}."""
    )
    
    data_file, relative_x_column, relative_y_column = convert_relative_xy_data(cfg, dataset_cfg)

    # Fail if file contains multiple satellites, not implemented yet
    if len(list(dataset_cfg.satellites.keys())) > 1:
        raise NotImplementedError("Multiple satellites in one NSDB file not supported yet.")

    observatories = list(dataset_cfg.observatory)
    telescope_index = dict(dataset_cfg.get("telescope_index", {}) or {})
    observatory_codes = resolve_observatory_codes(data_file, observatories, telescope_index)

    observation_sets = []
    observation_models = []
    for station_name, group in group_rows_by_observatory(data_file, observatory_codes):
        observation_set, observation_model = _build_relative_observation_set(
            cfg,
            dataset_cfg,
            system_of_bodies,
            group,
            station_name,
            relative_x_column,
            relative_y_column,
        )
        observation_sets.append(observation_set)
        observation_models.append(observation_model)

    if len(observation_sets) == 1:
        return observation_sets[0], observation_models[0]

    collection = obs.ObservationCollection(observation_sets)
    return collection, observation_models