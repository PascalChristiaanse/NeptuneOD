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
