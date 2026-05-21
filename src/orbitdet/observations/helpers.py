import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tudatpy.astro import element_conversion
from tudatpy.astro import time_representation as time_rep
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import environment_setup as env_setup

logger = logging.getLogger(__name__)

ISO_TIME_COLUMN = "iso_time"


def convert_time_to_seconds_since_j2000_TDB(
    dataframe: pd.DataFrame,
    observatory_code: int,
    system_of_bodies: env.SystemOfBodies,
    timescale: str,
) -> pd.Series:
    """Convert the observation times in the dataframe to seconds since J2000 epoch TDB, using the
    position of the observatory as saved in system_of_bodies. Adds epoch_TDB column to the dataframe
    in place.

    Args:
        dataframe: The DataFrame containing the observation data, with an ISO-8601 timestamp column.
        observatory_code: The code of the observatory where the observations were made.
        system_of_bodies: The system of bodies containing the observatory's position.
        timescale: The timescale of the input timestamps (e.g., "UTC", "TT", "TDB").

    Returns:
        A pandas Series containing the observation times in seconds since J2000 epoch TDB.
    """

    # Convert iso string to epoch using time_rep.iso_string_to_epoch
    iso_times = dataframe[ISO_TIME_COLUMN].tolist()
    epochs = [time_rep.iso_string_to_epoch(iso_time) for iso_time in iso_times]
    # Get observatory position
    observatory = system_of_bodies.get_body("Earth").get_ground_station(observatory_code)
    obs_position = observatory.station_state.cartesian_position_at_reference_epoch

    # Reuse TudatPy's default time scale converter instance for each conversion.
    tsc = time_rep.default_time_scale_converter()
    # Convert epochs to seconds since J2000 TDB
    match timescale:
        case "UT1":
            tudat_timescale = time_rep.TimeScales.ut1_scale
        case "UTC":
            tudat_timescale = time_rep.TimeScales.utc_scale
        case "TT":
            tudat_timescale = time_rep.TimeScales.tt_scale
        case "TDB":
            tudat_timescale = time_rep.TimeScales.tdb_scale
        case _:
            raise ValueError(f"Unsupported timescale: {timescale}")

    seconds_since_j2000_TDB = []
    for epoch in epochs:
        # Convert input epoch to TDB
        epoch_tdb = tsc.convert_time(
            tudat_timescale,
            time_rep.TimeScales.tdb_scale,
            epoch,
            obs_position.reshape(3, 1),
        )
        seconds_since_j2000_TDB.append(epoch_tdb)

    dataframe["epoch_TDB"] = seconds_since_j2000_TDB
    return pd.Series(seconds_since_j2000_TDB)


OBSERVATORY_INFO_FILE = "data/Observatories.txt"  # https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt


def _observatory_info(
    cfg: DictConfig,
    observatory_code: int,
) -> tuple[float, float, float]:  # Positive to north and east
    """Retrieve the station position from observatories.txt

    Args:
        observatory_code (int): observatory code

    Returns:
        tuple: longitude, latitude, and altitude of the observatory
    """
    if len(observatory_code) == 2:  # Making sure 098 and 98 are the same
        observatory_code = "0" + observatory_code
    elif len(observatory_code) == 1:  # Making sure 098 and 98 are the same
        observatory_code = "00" + observatory_code

    observatories_file = Path(cfg.data_folder) / 'observatories.txt'

    with open(
        observatories_file, "r"
    ) as file:  # https://www.projectpluto.com/obsc.htm, https://www.projectpluto.com/mpc_stat.txt
        lines = file.readlines()
        for line in lines[1:]:  # Ignore the first line
            columns = line.split()
            if columns[1] == observatory_code:
                longitude = float(columns[2])
                latitude = float(columns[3])
                altitude = float(columns[4])
                return np.deg2rad(longitude), np.deg2rad(latitude), altitude
        print("No matching Observatory found")


def add_observatory_to_SOB(
    cfg: DictConfig, system_of_bodies: env.SystemOfBodies, observatory_code: int
):
    """Add the observatory as a ground station to the system of bodies.

    Args:
        system_of_bodies: The system of bodies to which the observatory will be added.
        observatory_code: The code of the observatory to be added.

    Returns:
        None. The system_of_bodies is modified in place.
    """

    # Test if ground station already exists
    try:
        system_of_bodies.get_body("Earth").get_ground_station(str(observatory_code))
    except RuntimeError:
        pass
    else:
        logger.info(
            f"Ground station with code {observatory_code} already exists in the system of bodies."
        )
        return

    # Define the position of the observatory on Earth
    observatory_longitude, observatory_latitude, observatory_altitude = _observatory_info(
        cfg, observatory_code
    )

    env_setup.add_ground_station(
        system_of_bodies.get_body("Earth"),
        observatory_code,
        [observatory_altitude, observatory_latitude, observatory_longitude],
        element_conversion.geodetic_position_type,
    )
