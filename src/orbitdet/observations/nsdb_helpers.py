import re

import numpy as np
import pandas as pd

ISO_TIME_COLUMN = "iso_time"


def _normalize_column_name(column_name: object) -> str:
    return " ".join(str(column_name).strip().lower().replace("_", " ").split())


def _find_column(
    dataframe: pd.DataFrame, *needles: str, excludes: list[str] | None = None
) -> str | None:
    excludes = excludes or []
    for column_name in dataframe.columns:
        normalized = _normalize_column_name(column_name)

        # Skip columns containing any excluded keywords
        if any(re.search(rf"\b{re.escape(ex)}\b", normalized) for ex in excludes):
            continue

        matched_all = True
        for needle in needles:
            if needle == "jd":
                if not ("jd" in normalized or "julian" in normalized):
                    matched_all = False
                    break
            else:
                if not re.search(rf"\b{re.escape(needle)}\b", normalized):
                    matched_all = False
                    break

        if matched_all:
            return str(column_name)

    return None


def _numeric_series(dataframe: pd.DataFrame, column_name: str | None) -> pd.Series:
    if column_name is None:
        return pd.Series(0.0, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[column_name], errors="coerce")


def _to_iso_string(timestamp: object) -> str | pd._libs.missing.NAType:
    if pd.isna(timestamp):
        return pd.NA
    return pd.Timestamp(timestamp).isoformat()


def set_iso_time_column(dataframe: pd.DataFrame) -> pd.Series:
    """
    Infer the observation time columns and add an ISO-8601 timestamp column in place.

    Args:
        dataframe: The DataFrame containing the observation data, with columns named according to
        the format mapping.

    Returns:
        The name of the newly created ISO time column.

    Examples:
        A dataframe with columns such as JD, or year/month/day[/hour/minute/second], is
        converted to a new "iso_time" column containing ISO-8601 strings.
    """
    if ISO_TIME_COLUMN in dataframe.columns:
        return ISO_TIME_COLUMN

    jd_column = _find_column(dataframe, "jd")
    if jd_column is not None:
        timestamps = pd.to_datetime(
            pd.to_numeric(dataframe[jd_column], errors="coerce"),
            unit="D",
            origin="julian",
            errors="coerce",
        )
        dataframe[ISO_TIME_COLUMN] = timestamps.map(_to_iso_string)
        return ISO_TIME_COLUMN

    year_column = _find_column(dataframe, "year")
    month_column = _find_column(dataframe, "month")
    day_column = _find_column(dataframe, "day")
    if year_column is None or month_column is None or day_column is None:
        raise ValueError(
            "Could not infer an observation time column set. Expected JD or year/month/day columns."
        )

    # Avoid matching right-ascension / declination columns when looking for
    # observation time hour/minute/second.
    ra_excludes = ["right ascension", "alpha", "ra", "declination", "delta"]
    hour_column = _find_column(dataframe, "hour", excludes=ra_excludes)
    minute_column = _find_column(dataframe, "minute", excludes=ra_excludes)
    second_column = _find_column(dataframe, "second", excludes=ra_excludes)

    year = _numeric_series(dataframe, year_column)
    month = _numeric_series(dataframe, month_column)
    day = _numeric_series(dataframe, day_column)
    day_integer = np.floor(day)
    day_fraction_seconds = (day - day_integer) * 86400.0

    hour = _numeric_series(dataframe, hour_column)
    minute = _numeric_series(dataframe, minute_column)
    second = _numeric_series(dataframe, second_column)

    timestamps = pd.to_datetime(
        pd.DataFrame(
            {
                "year": year,
                "month": month,
                "day": day_integer,
            }
        ),
        errors="coerce",
    )
    timestamps = timestamps + pd.to_timedelta(
        hour * 3600.0 + minute * 60.0 + second + day_fraction_seconds,
        unit="s",
    )
    dataframe[ISO_TIME_COLUMN] = timestamps.map(_to_iso_string)
    return ISO_TIME_COLUMN


def set_ra_dec_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """Infer the right ascension and declination columns in the dataframe.

    Args:
        dataframe: The DataFrame containing the observation data, with columns named according to
        the format mapping.

    Returns:
        A tuple of (ra_column_name, dec_column_name) for the right ascension
        and declination columns.

    Examples:
        5. Hour   of right ascension (alpha, h)
        6. Minute of right ascension (alpha, m)
        7. Second of right ascension (alpha, s)
        8. Degree of declination (delta, deg)
        9. Minute of declination (delta, '  )
        10. Second of declination (delta, '' )
        ---

    """
    ra_excludes = ["declination", "delta"]
    dec_excludes = ["ra", "right", "ascension", "alpha"]
    ra_component_excludes = [
        "hour of",
        "minute of",
        "second of",
        "observation time",
        "utc",
    ]

    # Try to find RA column already in decimal degrees
    ra_column = _find_column(dataframe, "ra")
    if ra_column is None:
        ra_column = _find_column(dataframe, "alpha")
    if ra_column is None:
        ra_column = _find_column(dataframe, "right", "ascension")

    # Try to find Dec column already in decimal degrees
    dec_column = _find_column(dataframe, "dec")
    if dec_column is None:
        dec_column = _find_column(dataframe, "delta")
    if dec_column is None:
        dec_column = _find_column(dataframe, "declination")

    # If we found both in decimal degrees, return them
    if ra_column is not None and dec_column is not None:
        return (ra_column, dec_column)

    # Avoid matching time hour/minute/second columns when looking for RA components
    time_excludes = ra_component_excludes

    # Try to find component columns for RA (hour, minute, second)
    ra_hour_column = _find_column(
        dataframe,
        "ra",
        "hour",
        excludes=time_excludes + ra_excludes,
    )
    if ra_hour_column is None:
        ra_hour_column = _find_column(
            dataframe,
            "alpha",
            "hour",
            excludes=time_excludes + ra_excludes,
        )
    if ra_hour_column is None:
        ra_hour_column = _find_column(
            dataframe,
            "right",
            "ascension",
            "hour",
            excludes=time_excludes,
        )

    ra_minute_column = _find_column(
        dataframe,
        "ra",
        "minute",
        excludes=time_excludes + ra_excludes,
    )
    if ra_minute_column is None:
        ra_minute_column = _find_column(
            dataframe,
            "alpha",
            "minute",
            excludes=time_excludes + ra_excludes,
        )
    if ra_minute_column is None:
        ra_minute_column = _find_column(
            dataframe,
            "right",
            "ascension",
            "minute",
            excludes=time_excludes,
        )

    ra_second_column = _find_column(
        dataframe,
        "ra",
        "second",
        excludes=time_excludes + ra_excludes,
    )
    if ra_second_column is None:
        ra_second_column = _find_column(
            dataframe,
            "alpha",
            "second",
            excludes=time_excludes + ra_excludes,
        )
    if ra_second_column is None:
        ra_second_column = _find_column(
            dataframe,
            "right",
            "ascension",
            "second",
            excludes=time_excludes,
        )

    # Try to find component columns for Dec (degree, minute, second)
    dec_degree_column = _find_column(
        dataframe,
        "dec",
        "degree",
        excludes=dec_excludes,
    )
    if dec_degree_column is None:
        dec_degree_column = _find_column(
            dataframe,
            "delta",
            "degree",
            excludes=dec_excludes,
        )
    if dec_degree_column is None:
        dec_degree_column = _find_column(dataframe, "declination", "degree")

    dec_minute_column = _find_column(
        dataframe,
        "dec",
        "minute",
        excludes=dec_excludes,
    )
    if dec_minute_column is None:
        dec_minute_column = _find_column(
            dataframe,
            "delta",
            "minute",
            excludes=dec_excludes,
        )
    if dec_minute_column is None:
        dec_minute_column = _find_column(dataframe, "declination", "minute")

    dec_second_column = _find_column(
        dataframe,
        "dec",
        "second",
        excludes=dec_excludes,
    )
    if dec_second_column is None:
        dec_second_column = _find_column(
            dataframe,
            "delta",
            "second",
            excludes=dec_excludes,
        )
    if dec_second_column is None:
        dec_second_column = _find_column(dataframe, "declination", "second")

    # If we have component columns for RA, convert to decimal degrees
    if (
        ra_hour_column is not None
        or ra_minute_column is not None
        or ra_second_column is not None
    ):
        ra_hours = _numeric_series(dataframe, ra_hour_column)
        ra_minutes = _numeric_series(dataframe, ra_minute_column)
        ra_seconds = _numeric_series(dataframe, ra_second_column)

        # Convert from hours, minutes, seconds to decimal degrees
        # RA is in hours (0-24), so convert to degrees (0-360) by multiplying by 15
        ra_decimal = (ra_hours + ra_minutes / 60.0 + ra_seconds / 3600.0) * 15.0
        ra_column = "ra"
        dataframe[ra_column] = ra_decimal

    # If we have component columns for Dec, convert to decimal degrees
    if (
        dec_degree_column is not None
        or dec_minute_column is not None
        or dec_second_column is not None
    ):
        dec_degrees = _numeric_series(dataframe, dec_degree_column)
        dec_minutes = _numeric_series(dataframe, dec_minute_column)
        dec_seconds = _numeric_series(dataframe, dec_second_column)

        # Handle sign for declination (minutes and seconds should be positive)
        # The sign of declination comes from the degree component
        sign = np.sign(dec_degrees).replace(0, 1)
        dec_decimal = sign * (
            np.abs(dec_degrees)
            + np.abs(dec_minutes) / 60.0
            + np.abs(dec_seconds) / 3600.0
        )
        dec_column = "dec"
        dataframe[dec_column] = dec_decimal

    if ra_column is None or dec_column is None:
        raise ValueError(
            "Could not infer right ascension and/or declination columns. "
            "Expected RA/Alpha/Right Ascension and Dec/Delta/Declination columns "
            "either in decimal degrees or as hour/minute/second and "
            "degree/minute/second."
        )

    return (ra_column, dec_column)
