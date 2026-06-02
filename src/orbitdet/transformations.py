import logging

import pandas as pd
from astropy import units as u
from astropy.coordinates import FK4, ICRS, SkyCoord
from astropy.time import Time

logger = logging.getLogger(__name__)


def convert_fk4_b1950_to_icrs_j2000(
    data: pd.DataFrame,
    ra_column: str,
    dec_column: str,
    epoch_of_equinox,
) -> pd.DataFrame:
    if epoch_of_equinox is None:
        return data

    equinox_text = str(epoch_of_equinox).strip().upper()
    if not equinox_text.startswith("B"):
        return data

    try:
        equinox = Time(float(equinox_text[1:]), format="byear")
    except ValueError as exc:
        raise ValueError(
            f"Could not parse epoch_of_equinox='{epoch_of_equinox}' as a Besselian year."
        ) from exc

    skycoord = SkyCoord(
        ra=data[ra_column].to_numpy(dtype=float) * u.deg,
        dec=data[dec_column].to_numpy(dtype=float) * u.deg,
        frame=FK4(equinox=equinox),
    ).transform_to(ICRS())

    converted_data = data.copy()
    converted_data[ra_column] = skycoord.ra.deg
    converted_data[dec_column] = skycoord.dec.deg
    logger.info(
        "Converted Voyager coordinates from FK4/%s to ICRS/J2000 using epoch_of_equinox.",
        equinox_text,
    )
    return converted_data


def convert_fk4_b1950_cartesian_to_icrs_j2000(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    epoch_of_equinox,
) -> pd.DataFrame:
    if epoch_of_equinox is None:
        return data

    equinox_text = str(epoch_of_equinox).strip().upper()
    if not equinox_text.startswith("B"):
        return data

    try:
        equinox = Time(float(equinox_text[1:]), format="byear")
    except ValueError as exc:
        raise ValueError(
            f"Could not parse epoch_of_equinox='{epoch_of_equinox}' as a Besselian year."
        ) from exc

    skycoord = SkyCoord(
        x=data[x_column].to_numpy(dtype=float) * u.km,
        y=data[y_column].to_numpy(dtype=float) * u.km,
        z=data[z_column].to_numpy(dtype=float) * u.km,
        frame=FK4(equinox=equinox),
        representation_type="cartesian",
    ).transform_to(ICRS())

    converted_data = data.copy()
    converted_data[x_column] = skycoord.cartesian.x.to_value(u.km)
    converted_data[y_column] = skycoord.cartesian.y.to_value(u.km)
    converted_data[z_column] = skycoord.cartesian.z.to_value(u.km)
    logger.info(
        "Converted Voyager Cartesian coordinates from FK4/%s to ICRS/J2000 using epoch_of_equinox.",
        equinox_text,
    )
    return converted_data
