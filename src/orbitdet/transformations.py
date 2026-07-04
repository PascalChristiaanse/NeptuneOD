import logging

import numpy as np
import pandas as pd
from tudatpy.interface import spice

logger = logging.getLogger(__name__)


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
    """Convert RA/DEC between reference frames using SPICE rotation matrices.

    The method:
        RA/DEC -> unit vector -> rotate -> RA/DEC

    Args:
        data (pd.DataFrame): input data containing RA/DEC (degrees).
        ra_column (str): column name for right ascension (deg).
        dec_column (str): column name for declination (deg).
        input_frame (str): input reference frame name.
        output_frame (str): output reference frame name.
        time_column (str, optional): epoch column for time-dependent frames.
        ra_wrap (bool): wrap RA into [0, 360). Default True.

    Returns:
        pd.DataFrame: copy of input data with transformed RA/DEC.
    """
    if input_frame == output_frame:
        return data.copy()
    if input_frame == "J2000.0":
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
