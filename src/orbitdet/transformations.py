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


def _radec_icrs_to_j2000(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    angle_unit: str = "deg",
) -> pd.DataFrame:
    """Apply the ICRS -> FK5 (equinox J2000) frame-bias rotation via Astropy.

    ICRS and the FK5 mean equator/equinox of J2000 differ by a small constant
    rotation (frame bias, ~17-20 mas).  SPICE has no exact ICRS frame, so this
    uses Astropy to transform the coordinates exactly.  The input column unit
    is preserved in the output.

    Args:
        data: DataFrame with RA/DEC columns.
        ra_column: Column name for right ascension.
        dec_column: Column name for declination.
        angle_unit: Unit of the RA/DEC columns, ``"deg"`` or ``"rad"``.

    Returns:
        Copy of *data* with RA/DEC rotated from ICRS to FK5-J2000, in *angle_unit*.
    """
    from astropy import units as u
    from astropy.coordinates import FK5, ICRS
    from astropy.time import Time

    ras = data[ra_column].to_numpy(copy=True)
    decs = data[dec_column].to_numpy(copy=True)

    angle_unit = str(angle_unit).lower()
    if angle_unit == "rad":
        unit = u.rad
    else:
        unit = u.deg

    icrs_coords = ICRS(ras * unit, decs * unit)
    j2000_coords = icrs_coords.transform_to(FK5(equinox=Time("J2000", scale="tdb")))

    out = data.copy()
    if angle_unit == "rad":
        new_ra = j2000_coords.ra.to(u.rad).value
        new_dec = j2000_coords.dec.to(u.rad).value
        # Astropy reports RA in [0, 2*pi); rewrap to the source convention
        # (principal interval [-pi, pi]) so the small frame bias leaves the
        # angles near their original values.
        new_ra = np.remainder(new_ra + np.pi, 2.0 * np.pi) - np.pi
        out[ra_column] = new_ra
        out[dec_column] = new_dec
    else:
        out[ra_column] = j2000_coords.ra.to(u.deg).value
        out[dec_column] = j2000_coords.dec.to(u.deg).value

    logger.info("Applied ICRS -> FK5 (J2000) frame-bias rotation to RA/DEC.")
    return out


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
    angle_unit: str = "deg",
) -> pd.DataFrame:
    """Convert RA/DEC between reference frames using SPICE and/or Astropy.

    Supports known SPICE frames (J2000, ICRS, B1950, etc.) and the NSDB
    ``'Date'`` / ``'date'`` frame (mean equator and equinox of the observation
    date).  When *input_frame* or *output_frame* is ``'Date'``, Astropy FK5
    precession is used because SPICE does not recognise this frame.

    ICRS is *not* identical to J2000: there is a real frame-bias rotation
    (~17-20 mas) between the ICRS and the FK5 mean equator/equinox of J2000.
    Because SPICE has no exact ICRS sprite, an ``ICRS``/``ICRF`` *input* is
    rotated to J2000 with an Astropy frame-bias transform before any onward
    SPICE conversion.

    The method:
        RA/DEC -> unit vector -> rotate -> RA/DEC

    Args:
        data: input data containing RA/DEC.
        ra_column: column name for right ascension.
        dec_column: column name for declination.
        input_frame: input reference frame name (or ``'Date'``).
        output_frame: output reference frame name (or ``'Date'``).
        time_column: epoch column (seconds since J2000 TDB) required for
            time-dependent frame conversions.  **Must** be provided when
            either frame is ``'Date'``.
        ra_wrap: wrap RA into [0, 360). Default True.
        angle_unit: unit of the RA/DEC columns. Either ``"deg"`` (default) or
            ``"rad"``.  The returned frame has the same unit as the input.

    Returns:
        Copy of input data with transformed RA/DEC (same angular unit as input).
    """
    if input_frame == output_frame:
        return data.copy()

    input_is_date = str(input_frame).lower() == "date"
    output_is_date = str(output_frame).lower() == "date"

    # ICRS/ICRF input: apply the real ICRS->J2000 frame-bias rotation (Astropy),
    # then (if needed) continue from J2000 to the requested output via SPICE.
    input_is_icrs = str(input_frame).strip().upper() in {"ICRS", "ICRF"}
    if input_is_icrs and not output_is_date and not str(output_frame).lower() in {"date"}:
        output_norm = str(output_frame).strip().upper()
        if output_norm in {"J2000", "J2000.0"}:
            return _radec_icrs_to_j2000(data, ra_column, dec_column, angle_unit)

        # ICRS -> J2000 (frame bias), then J2000 -> output_frame (SPICE).
        # Do the frame bias in radians, run SPICE on degrees, restore input unit.
        inter = _radec_icrs_to_j2000(data, ra_column, dec_column, "rad")
        temp = inter.copy()
        temp[ra_column] = np.rad2deg(temp[ra_column])
        temp[dec_column] = np.rad2deg(temp[dec_column])
        out = _radec_spice_convert(
            temp, ra_column, dec_column, "J2000", output_frame, time_column, ra_wrap
        )
        if angle_unit == "rad":
            out[ra_column] = np.deg2rad(out[ra_column])
            out[dec_column] = np.deg2rad(out[dec_column])
        return out

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


def _canonicalize_frame(frame: str) -> str:
    """Map non-SPICE frame names to the closest SPICE frame.

    ``ICRS``/``ICRF`` should have been handled by the Astropy frame-bias path
    in :func:`convert_radec_frame`; if one still reaches SPICE directly it is
    approximated as J2000 (they coincide to ~17-20 mas, not 0.1 arcsec).
    ``J2015.0`` is a precession of the J2000 orientation.
    """
    frame = str(frame).strip()
    if frame.upper() in {"ICRS", "ICRF"}:
        if frame != "J2000":
            logger.warning(
                "Frame %s reached the SPICE path without the Astropy frame-bias "
                "correction; approximating as J2000 (frames differ by ~20 mas).",
                frame,
            )
        return "J2000"
    if frame == "J2015.0":
        logger.warning(
            "Input frame is J2015.0, which is not a standard SPICE frame. "
            "Using J2000 as an approximation for the transformation."
            "This should be fixed later, J2015 is a precession, orientation is just J2000.0"
        )
        return "J2000"
    if frame == "J2000.0":
        return "J2000"
    return frame


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
    input_frame = _canonicalize_frame(input_frame)
    output_frame = _canonicalize_frame(output_frame)

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
