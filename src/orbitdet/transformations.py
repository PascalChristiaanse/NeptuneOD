"""Coordinate transformation utilities (Cartesian and RA/Dec frame conversions).

Includes Astropy-based precession for handling NSDB 'Date' (mean equator and
equinox of date) frames that SPICE does not recognize.
"""

import logging

import numpy as np
import pandas as pd
from tudatpy.interface import spice

logger = logging.getLogger(__name__)

# Julian date of J2000.0 epoch
JD_J2000 = 2451545.0


def _seconds_since_j2000_to_jd(et_s: float | np.ndarray) -> float | np.ndarray:
    """Convert seconds since J2000.0 TDB to Julian Date (TDB)."""
    return et_s / 86400.0 + JD_J2000


def _precess_radec_date_to_j2000(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Precess RA/Dec from mean-of-date (Date equinox) to J2000 using Astropy.

    The NSDB 'Date' / 'date' epoch-of-equinox means coordinates are in the mean
    equator and equinox of the observation date (FK5 system).  SPICE does not
    recognise this frame, so we use Astropy to precess to J2000 first.

    Args:
        data: DataFrame with RA (deg), Dec (deg), and epoch (s since J2000 TDB).
        ra_column: Column name for right ascension in degrees.
        dec_column: Column name for declination in degrees.
        time_column: Column name for epoch in seconds since J2000.0 TDB.

    Returns:
        Copy of *data* with RA/Dec columns precessed from mean-of-date to J2000.
    """
    from astropy import units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    ras = data[ra_column].to_numpy(copy=True)
    decs = data[dec_column].to_numpy(copy=True)
    ets = data[time_column].to_numpy(copy=True)

    mask = ~(np.isnan(ras) | np.isnan(decs) | np.isnan(ets))
    if not mask.any():
        return data.copy()

    jds = _seconds_since_j2000_to_jd(ets[mask])
    obstimes = Time(jds, format="jd", scale="tdb")

    coord_date = SkyCoord(ras[mask], decs[mask], unit=u.deg, frame=FK5(equinox=obstimes))
    coord_j2000 = coord_date.transform_to(FK5(equinox="J2000"))

    out = data.copy()
    out.loc[mask, ra_column] = coord_j2000.ra.deg
    out.loc[mask, dec_column] = coord_j2000.dec.deg
    return out


def _precess_radec_j2000_to_date(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    time_column: str,
) -> pd.DataFrame:
    """Precess RA/Dec from J2000 to mean-of-date (Date equinox) using Astropy.

    Args:
        data: DataFrame with RA (deg), Dec (deg), and epoch (s since J2000 TDB).
        ra_column: Column name for right ascension in degrees.
        dec_column: Column name for declination in degrees.
        time_column: Column name for epoch in seconds since J2000.0 TDB.

    Returns:
        Copy of *data* with RA/Dec columns precessed from J2000 to mean-of-date.
    """
    from astropy import units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    ras = data[ra_column].to_numpy(copy=True)
    decs = data[dec_column].to_numpy(copy=True)
    ets = data[time_column].to_numpy(copy=True)

    mask = ~(np.isnan(ras) | np.isnan(decs) | np.isnan(ets))
    if not mask.any():
        return data.copy()

    jds = _seconds_since_j2000_to_jd(ets[mask])
    obstimes = Time(jds, format="jd", scale="tdb")

    coord_j2000 = SkyCoord(ras[mask], decs[mask], unit=u.deg, frame=FK5(equinox="J2000"))
    coord_date = coord_j2000.transform_to(FK5(equinox=obstimes))

    out = data.copy()
    out.loc[mask, ra_column] = coord_date.ra.deg
    out.loc[mask, dec_column] = coord_date.dec.deg
    return out


def convert_cartesian_frame(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    input_frame: str,
    output_frame: str,
    time_column: str = None,
) -> pd.DataFrame:
    """Rotate Cartesian coordinates from input_frame to output_frame using SPICE rotation matrices.

    Args:
        data (pd.DataFrame): input data containing Cartesian coordinates and optionally time.
        Modified in-place.
        x_column (str): column name for x coordinate in data.
        y_column (str): column name for y coordinate in data.
        z_column (str): column name for z coordinate in data.
        input_frame (str): identifier of the input reference frame
        (e.g. "B1950", "FK4", "J2000", "ICRS").
        output_frame (str): identifier of the output reference frame
        (e.g. "B1950", "FK4", "J2000", "ICRS").
        time_column (str, optional): . Defaults to None.

    Returns:
        pd.DataFrame: data with Cartesian coordinates rotated to the output frame.
        The original x/y/z columns are overwritten.
    """
    if input_frame == output_frame:
        logger.info(f"Input and output frames are the same ({input_frame}), no conversion applied.")
        return data
    if time_column is None:
        time_column = pd.Series([0.0] * len(data))

    def rotate_row(row):
        epoch = row[time_column]
        rotation_matrix = spice.compute_rotation_matrix_between_frames(
            input_frame, output_frame, epoch
        )
        vector = np.array([row[x_column], row[y_column], row[z_column]])
        transformed_vector = rotation_matrix @ vector
        return transformed_vector

    transformed_vectors = data.apply(rotate_row, axis=1, result_type="expand")
    converted_data = data.copy()
    converted_data[x_column] = transformed_vectors[0]
    converted_data[y_column] = transformed_vectors[1]
    converted_data[z_column] = transformed_vectors[2]
    return converted_data


def convert_radec_frame(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    input_frame: str,
    output_frame: str,
    time_column: str = None,
    ra_wrap: bool = True,
) -> pd.DataFrame:
    """Convert RA/DEC between reference frames using SPICE and/or Astropy.

    Supports known SPICE frames (J2000, ICRS, B1950, etc.) and the NSDB
    ``'Date'`` / ``'date'`` frame (mean equator and equinox of the observation
    date).  When *input_frame* or *output_frame* is ``'Date'``, Astropy FK5
    precession is used because SPICE does not recognise this frame.

    The method:
        RA/DEC -> unit vector -> rotate -> RA/DEC

    Args:
        data: input data containing RA/DEC (degrees).
        ra_column: column name for right ascension (deg).
        dec_column: column name for declination (deg).
        input_frame: input reference frame name (or ``'Date'``).
        output_frame: output reference frame name (or ``'Date'``).
        time_column: epoch column (seconds since J2000 TDB) required for
            time-dependent frame conversions.  **Must** be provided when
            either frame is ``'Date'``.
        ra_wrap: wrap RA into [0, 360). Default True.

    Returns:
        Copy of input data with transformed RA/DEC.
    """
    if input_frame == output_frame:
        return data.copy()

    input_is_date = input_frame.lower() == "date"
    output_is_date = output_frame.lower() == "date"

    # If neither frame is Date, use pure SPICE as before
    if not input_is_date and not output_is_date:
        return _radec_spice_convert(
            data, ra_column, dec_column, input_frame, output_frame, time_column, ra_wrap
        )

    if time_column is None:
        raise ValueError("time_column is required when converting to/from the 'Date' frame.")

    # Strategy: chain through J2000 as an intermediate.
    #   Date -> J2000 : Astropy precession
    #   J2000 -> *    : SPICE (normal path)
    #   *    -> J2000 : SPICE (normal path)
    #   J2000 -> Date : Astropy precession

    if input_is_date and not output_is_date:
        # Date -> J2000 (precess)
        out = _precess_radec_date_to_j2000(data, ra_column, dec_column, time_column)
        # J2000 -> output_frame (SPICE)
        if output_frame.lower() != "j2000":
            out = _radec_spice_convert(
                out, ra_column, dec_column, "J2000", output_frame, time_column, ra_wrap
            )
        return out

    if output_is_date and not input_is_date:
        # input_frame -> J2000 (SPICE)
        if input_frame.lower() != "j2000":
            out = _radec_spice_convert(
                data, ra_column, dec_column, input_frame, "J2000", time_column, ra_wrap
            )
        else:
            out = data.copy()
        # J2000 -> Date (precess)
        out = _precess_radec_j2000_to_date(out, ra_column, dec_column, time_column)
        return out

    # Date -> Date: no-op (already same frame), but just in case
    return data.copy()


def _radec_spice_convert(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    input_frame: str,
    output_frame: str,
    time_column: str = None,
    ra_wrap: bool = True,
) -> pd.DataFrame:
    """Convert RA/DEC between two SPICE-recognised frames.

    This is the original SPICE-based conversion extracted from
    *convert_radec_frame* for reuse.
    """
    if input_frame == "J2000.0":
        input_frame = "J2000"

    if input_frame == "J2015.0":
        logger.warning(
            "Input frame is J2015.0, which is not a standard SPICE frame. "
            "Using J2000 as an approximation for the transformation."
            "This should be fixed later, J2015 is a precession, orientation is just J2000.0"
        )
        input_frame = "J2000"

    def radec_to_vector(ra_deg, dec_deg):
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)

        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return np.array([x, y, z])

    def vector_to_radec(v):
        x, y, z = v
        r = np.linalg.norm(v)
        if r == 0:
            return np.nan, np.nan

        x, y, z = x / r, y / r, z / r

        dec = np.arcsin(z)
        ra = np.arctan2(y, x)

        ra_deg = np.rad2deg(ra)
        dec_deg = np.rad2deg(dec)

        if ra_wrap:
            ra_deg = ra_deg % 360.0

        return ra_deg, dec_deg

    def transform_row(row):
        epoch = row[time_column] if time_column is not None else 0.0

        rotation_matrix = spice.compute_rotation_matrix_between_frames(
            input_frame, output_frame, epoch
        )

        vec = radec_to_vector(row[ra_column], row[dec_column])
        vec_out = rotation_matrix @ vec
        return vector_to_radec(vec_out)

    transformed = data.apply(transform_row, axis=1, result_type="expand")

    out = data.copy()
    out[ra_column] = transformed[0]
    out[dec_column] = transformed[1]

    return out


def convert_observation_to_apparent_direction(
    data: pd.DataFrame, ra_column: str, dec_column: str
) -> pd.DataFrame:
    """Convert RA/DEC in degrees to unit Cartesian vectors representing the apparent direction to
    the target. Introduces new columns "obs_x", "obs_y", "obs_z" in the data.

    Args:
        data (pd.DataFrame): input data containing RA and DEC columns. Modified in-place.
        ra_column (str): column name for right ascension in degrees.
        dec_column (str): column name for declination in degrees.

    Returns:
        pd.DataFrame: data with RA/DEC columns replaced by x/y/z unit vector columns.
    """

    def convert_row(row):
        ra_rad = np.deg2rad(row[ra_column])
        dec_rad = np.deg2rad(row[dec_column])
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)
        return np.array([x, y, z])

    transformed_vectors = data.apply(convert_row, axis=1, result_type="expand")
    converted_data = data.copy()
    converted_data["obs_x"] = transformed_vectors[0]
    converted_data["obs_y"] = transformed_vectors[1]
    converted_data["obs_z"] = transformed_vectors[2]
    return converted_data
