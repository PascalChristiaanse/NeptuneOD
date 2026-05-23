import pandas as pd
import pytest

from orbitdet.observations.nsdb_helpers import (
    ISO_TIME_COLUMN,
    set_iso_time_column,
    set_ra_dec_columns,
)


def test_set_iso_time_column_from_julian_date():
    dataframe = pd.DataFrame({"JD of the moment of observation": [2451545.0]})

    column_name = set_iso_time_column(dataframe)

    assert column_name == ISO_TIME_COLUMN
    assert dataframe[column_name].tolist() == ["2000-01-01T12:00:00"]


def test_set_iso_time_column_from_calendar_parts():
    dataframe = pd.DataFrame(
        {
            "Year of the moment of observation": [2001],
            "Month of the moment of observation": [1],
            "Day of the moment of observation with decimals": [2.5],
        }
    )

    column_name = set_iso_time_column(dataframe)

    assert column_name == ISO_TIME_COLUMN
    assert dataframe[column_name].tolist() == ["2001-01-02T12:00:00"]


def test_set_ra_dec_columns_from_nsdb_style_components():
    dataframe = pd.DataFrame(
        {
            "Hour of right ascension (alpha, h)": [5],
            "Minute of right ascension (alpha, m)": [30],
            "Second of right ascension (alpha, s)": [0],
            "Degree of declination (delta, deg)": [-12],
            "Minute of declination (delta, arcmin)": [30],
            "Second of declination (delta, arcsec)": [15],
        }
    )

    ra_column, dec_column = set_ra_dec_columns(dataframe)

    assert (ra_column, dec_column) == ("ra", "dec")
    assert dataframe[ra_column].iloc[0] == pytest.approx(82.5)
    assert dataframe[dec_column].iloc[0] == pytest.approx(-12.5041666667)


def test_set_ra_dec_columns_raises_when_components_are_missing(caplog):
    dataframe = pd.DataFrame({"unrelated": [1]})

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="Could not infer right ascension"):
            set_ra_dec_columns(dataframe)

    assert any("Could not infer right ascension" in record.message for record in caplog.records)
