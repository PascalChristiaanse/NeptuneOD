import pandas as pd

from orbitdet.observations.nsdb_helpers import ISO_TIME_COLUMN, set_iso_time_column


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
