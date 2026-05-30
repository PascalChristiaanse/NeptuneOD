import logging
import re

import numpy as np
import pandas as pd

ISO_TIME_COLUMN = "iso_time"
RELATIVE_POSITION_X_COLUMN = "relative_position_x"
RELATIVE_POSITION_Y_COLUMN = "relative_position_y"

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


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


def set_iso_time_column(dataframe: pd.DataFrame) -> str:
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
    """Infer RA/Dec component columns and convert them into decimal-degree columns.

    The dataframe is scanned for column names containing any RA labels (ra, right ascension,
    alpha) and Dec labels (dec, declination, delta), then matched to hour/minute/second or
    degree/minute/second components. The converted values are written to new `ra` and `dec`
    columns.
    """

    ra_hour_column = None
    ra_minute_column = None
    ra_second_column = None
    dec_degree_column = None
    dec_minute_column = None
    dec_second_column = None

    for column_name in dataframe.columns:
        normalized = _normalize_column_name(column_name)

        has_ra_label = (
            re.search(r"\bra\b", normalized)
            or "right ascension" in normalized
            or re.search(r"\balpha\b", normalized)
        )
        has_dec_label = (
            re.search(r"\bdec\b", normalized)
            or re.search(r"\bdeclination\b", normalized)
            or re.search(r"\bdelta\b", normalized)
        )

        if has_ra_label:
            if ra_hour_column is None and re.search(r"\bhour\b", normalized):
                ra_hour_column = str(column_name)
            elif ra_minute_column is None and re.search(r"\bminute\b", normalized):
                ra_minute_column = str(column_name)
            elif ra_second_column is None and re.search(r"\bsecond\b", normalized):
                ra_second_column = str(column_name)

        if has_dec_label:
            if dec_degree_column is None and re.search(r"\bdegree\b|\bdeg\b", normalized):
                dec_degree_column = str(column_name)
            elif dec_minute_column is None and re.search(r"\bminute\b", normalized):
                dec_minute_column = str(column_name)
            elif dec_second_column is None and re.search(r"\bsecond\b", normalized):
                dec_second_column = str(column_name)

    missing_ra = [
        part
        for part, column_name in (
            ("hour", ra_hour_column),
            ("minute", ra_minute_column),
            ("second", ra_second_column),
        )
        if column_name is None
    ]
    missing_dec = [
        part
        for part, column_name in (
            ("degree", dec_degree_column),
            ("minute", dec_minute_column),
            ("second", dec_second_column),
        )
        if column_name is None
    ]

    if missing_ra or missing_dec:
        message = (
            "Could not infer right ascension and declination component columns from the dataframe."
        )
        if missing_ra:
            message += f" Missing RA components: {', '.join(missing_ra)}."
        if missing_dec:
            message += f" Missing Dec components: {', '.join(missing_dec)}."
        logger.error(message)
        raise RuntimeError(message)

    ra_hours = _numeric_series(dataframe, ra_hour_column)
    ra_minutes = _numeric_series(dataframe, ra_minute_column)
    ra_seconds = _numeric_series(dataframe, ra_second_column)
    dataframe["ra"] = (ra_hours + ra_minutes / 60.0 + ra_seconds / 3600.0) * 15.0

    dec_degrees = _numeric_series(dataframe, dec_degree_column)
    dec_minutes = _numeric_series(dataframe, dec_minute_column)
    dec_seconds = _numeric_series(dataframe, dec_second_column)
    sign = np.sign(dec_degrees).replace(0, 1)
    dataframe["dec"] = sign * (
        np.abs(dec_degrees) + np.abs(dec_minutes) / 60.0 + np.abs(dec_seconds) / 3600.0
    )

    return ("ra", "dec")
