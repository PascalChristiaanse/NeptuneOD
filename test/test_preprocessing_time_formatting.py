"""Tests for ``orbitdet.preprocessing.time_formatting``.

The reference throughout is TudatPy's own ``DateTime``, because that is what the
conversion now *is*: the module hands each source layout to
``DateTime.from_iso_string`` / ``from_julian_day`` / ``from_modified_julian_day``
and only decides which columns carry the epoch and what the source's quirks mean.
These tests therefore pin the two things that are still ours to get wrong - the
layout detection, and the reduction of a source's components to something
``DateTime`` accepts - rather than re-deriving the epoch arithmetic.

Astropy is deliberately *not* a reference: its UTC Julian Date is
leap-second-aware and differs from TudatPy by up to ~0.47 s on a leap-second day.
"""

import numpy as np
import pandas as pd
import pytest
from tudatpy.astro.time_representation import DateTime

from orbitdet.preprocessing.time_formatting import (
    EPOCH_COLUMN,
    JD_J2000,
    MJD_OFFSET_DAYS,
    SECONDS_PER_DAY,
    EpochLayout,
    detect_epoch_layout,
    seconds_since_j2000_to_datetime,
    seconds_since_j2000_to_julian_date,
    to_seconds_since_j2000,
)


def tudat_seconds(year, month, day, hour=0, minute=0, second=0.0):
    """Reference value from TudatPy, the authority for every epoch conversion."""
    return DateTime.from_iso_string(
        f"{int(year):04d}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute):02d}:{second:09.6f}"
    ).to_epoch()


def calendar_seconds(year, month, day, hour=None, minute=None, second=None):
    """Convert calendar components the way ``to_seconds_since_j2000`` would.

    Wraps the components in a frame so the tests exercise the public entry point
    rather than a helper that no longer exists.  ``None`` marks a component the
    source does not have, and so produces no column.
    """
    frame = pd.DataFrame({"Year": [year], "Month": [month], "Day": [day]})
    for name, value in (("Hour", hour), ("Minute", minute), ("Second", second)):
        if value is not None:
            frame[name] = [value]
    return to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]


def datetime_to_seconds(moment):
    """Read a naive calendar instant back as an epoch, for round-trip tests."""
    return DateTime.from_iso_string(moment.isoformat()).to_epoch()


class TestCalendarComponents:
    def test_j2000_epoch_is_zero(self):
        out = calendar_seconds(2000, 1, 1, 12)

        assert out == pytest.approx(0.0)

    def test_one_day_later_is_86400(self):
        out = calendar_seconds(2000, 1, 2, 12)

        assert out == pytest.approx(SECONDS_PER_DAY)

    def test_midnight_is_minus_half_a_day(self):
        out = calendar_seconds(2000, 1, 1)

        assert out == pytest.approx(-SECONDS_PER_DAY / 2)

    @pytest.mark.parametrize(
        "year, month, day, hour, minute, second",
        [
            (2000, 1, 1, 12, 0, 0.0),
            (1900, 1, 1, 0, 0, 0.0),
            (2012, 6, 30, 11, 11, 37.0),
            (2015, 6, 15, 8, 30, 15.0),
            (1898, 9, 9, 22, 44, 30.25),
            (1850, 12, 31, 23, 59, 59.0),
            (2100, 3, 1, 6, 0, 0.0),
            (2020, 6, 15, 4, 20, 1.5),
        ],
    )
    def test_matches_tudat_reference(self, year, month, day, hour, minute, second):
        out = calendar_seconds(year, month, day, hour, minute, second)

        assert out == pytest.approx(tudat_seconds(year, month, day, hour, minute, second))

    def test_fractional_day_equals_explicit_hour(self):
        # NSDB writes either "day with decimals" or separate h/m/s columns for
        # the same instant, so both must land on the same epoch.
        fractional = calendar_seconds(2012, 6, 30.5)
        explicit = calendar_seconds(2012, 6, 30, 12)

        assert fractional == pytest.approx(explicit)

    def test_fractional_day_adds_to_explicit_hour(self):
        both = calendar_seconds(2012, 6, 30.25, 12)
        just_day = calendar_seconds(2012, 6, 30.25)

        assert both - just_day == pytest.approx(12 * 3600.0)

    def test_missing_components_default_to_zero(self):
        with_none = calendar_seconds(2000, 1, 1)
        with_zeros = calendar_seconds(2000, 1, 1, 0, 0, 0.0)

        assert with_none == pytest.approx(with_zeros)

    def test_nan_year_yields_nan(self):
        out = calendar_seconds(np.nan, 1, 1)

        assert np.isnan(out)

    def test_nan_month_yields_nan(self):
        out = calendar_seconds(2000, np.nan, 1)

        assert np.isnan(out)

    def test_nan_day_yields_nan(self):
        out = calendar_seconds(2000, 1, np.nan)

        assert np.isnan(out)

    def test_nan_propagates_row_wise_not_globally(self):
        frame = pd.DataFrame({"Year": [2000, np.nan, 2001], "Month": [1, 1, 1], "Day": [1, 1, 1]})

        out = to_seconds_since_j2000(frame)[EPOCH_COLUMN].to_numpy()

        assert np.isfinite(out[0])
        assert np.isnan(out[1])
        assert np.isfinite(out[2])

    def test_nan_time_component_is_treated_as_zero(self):
        # A blank hour is a missing time of day, not a missing epoch: the date
        # is still meaningful and must not become NaN.
        with_nan = calendar_seconds(2000, 1, 1, np.nan)

        assert with_nan == pytest.approx(-SECONDS_PER_DAY / 2)

    def test_infinite_time_component_yields_nan_not_a_crash(self):
        # Infinity is not a time of day; the record has no usable epoch.
        with_inf = calendar_seconds(2000, 1, 1, np.inf)

        assert np.isnan(with_inf)

    def test_hour_beyond_a_day_carries_into_the_date(self):
        # Some sources write hour 24 for the following midnight.
        out = calendar_seconds(2000, 1, 1, 24)

        assert out == pytest.approx(tudat_seconds(2000, 1, 2, 0))

    def test_leap_day_is_representable(self):
        out = calendar_seconds(2000, 2, 29, 12)

        assert out == pytest.approx(tudat_seconds(2000, 2, 29, 12))

    def test_century_non_leap_year(self):
        # 1900 was not a leap year.
        out = calendar_seconds(1900, 3, 1, 12)

        assert out == pytest.approx(tudat_seconds(1900, 3, 1, 12))

    def test_january_and_february_roll_into_previous_year(self):
        for month, day in ((1, 15), (2, 15)):
            out = calendar_seconds(1999, month, day, 12)
            assert out == pytest.approx(tudat_seconds(1999, month, day, 12))

    def test_impossible_date_raises(self):
        # A date that does not exist means the source is being misread, and a
        # silently wrong epoch is indistinguishable from a wrong orbit after.
        with pytest.raises(RuntimeError):
            calendar_seconds(2012, 2, 30)

    @pytest.mark.parametrize("second", [59.9999999994, 59.9999999995, 59.9999999999])
    def test_seconds_rounding_up_to_a_minute_is_not_rendered_negative(self, second):
        # Rounding must happen before the seconds are split into m/s: splitting
        # first and then subtracting 60 leaves a negative zero, which renders as
        # "-0.000000000" and is rejected by the parser.
        out = calendar_seconds(2000, 1, 1, 12, 30, second)

        assert out == pytest.approx(tudat_seconds(2000, 1, 1, 12, 31, 0.0))

    def test_seconds_rounding_up_at_end_of_day_advances_the_date(self):
        out = calendar_seconds(2000, 1, 1, 23, 59, 59.9999999999)

        assert out == pytest.approx(tudat_seconds(2000, 1, 2, 0, 0, 0.0))

    @pytest.mark.parametrize(("month", "day"), [(1, -1), (-1, 1), (-1, -1)])
    def test_negative_month_or_day_raises_rather_than_being_misparsed(self, month, day):
        # "2000-01--1" is accepted by the parser's lexical cast as a plausible
        # instant rather than rejected, so a negative field has to be caught
        # before the string is built.
        with pytest.raises(ValueError, match="not a real date"):
            calendar_seconds(2000, month, day)


class TestDetectEpochLayout:
    def test_detects_julian_date(self):
        frame = pd.DataFrame({"JD of the moment of observation": [2451545.0]})

        assert detect_epoch_layout(frame).kind == "jd"

    def test_detects_julian_from_spelled_out_label(self):
        frame = pd.DataFrame({"Julian date": [2451545.0]})

        assert detect_epoch_layout(frame).kind == "jd"

    def test_detects_mjd(self):
        frame = pd.DataFrame({"MJD of observation": [58000.0]})

        assert detect_epoch_layout(frame).kind == "mjd"

    def test_mjd_is_detected_before_jd(self):
        # "mjd" contains "jd"; without an explicit exclusion the Julian branch
        # would claim it and misread the value by 2400000.5 days.
        frame = pd.DataFrame({"MJD of observation": [58000.0], "JD extra": [1.0]})

        assert detect_epoch_layout(frame).kind == "mjd"

    def test_detects_modified_julian_date_spelled_out(self):
        frame = pd.DataFrame({"Modified Julian date": [58000.0]})

        assert detect_epoch_layout(frame).kind == "mjd"

    def test_detects_iso(self):
        frame = pd.DataFrame({"iso_time": ["2012-06-30T11:11:37"]})

        assert detect_epoch_layout(frame).kind == "iso"

    def test_detects_calendar(self):
        frame = pd.DataFrame(
            {
                "Year of observation": [2012],
                "Month of observation": [6],
                "Day of observation": [30],
            }
        )

        layout = detect_epoch_layout(frame)

        assert layout.kind == "calendar"
        assert layout.is_calendar

    def test_calendar_excludes_ra_and_dec_components(self):
        frame = pd.DataFrame(
            {
                "Year of the moment of observation": [2012],
                "Month of the moment of observation": [6],
                "Day of the moment of observation": [30],
                "Hour of right ascension (alpha, h)": [22],
                "Minute of right ascension (alpha, m)": [19],
                "Second of right ascension (alpha, s)": [53.1],
                "Degree of declination (delta, deg)": [-11],
                "Minute of declination (delta, ')": [2],
                "Second of declination (delta, '')": [32.7],
                "Hour of the moment of observation": [11],
                "Minute of the moment of observation": [11],
                "Second of the moment of observation": [37.0],
            }
        )

        layout = detect_epoch_layout(frame)

        assert layout.columns["hour"] == "Hour of the moment of observation"
        assert layout.columns["minute"] == "Minute of the moment of observation"
        assert layout.columns["second"] == "Second of the moment of observation"

    def test_raises_when_no_epoch_is_present(self):
        frame = pd.DataFrame({"something": [1], "else": [2]})

        with pytest.raises(ValueError, match="Could not infer an observation epoch"):
            detect_epoch_layout(frame)

    def test_error_lists_available_columns(self):
        frame = pd.DataFrame({"alpha": [1]})

        with pytest.raises(ValueError, match="alpha"):
            detect_epoch_layout(frame)


class TestDeclaredEpochReference:
    """A numeric epoch column may count days from a reference it names itself.

    Reading such a value as an absolute Julian Date is a silent, enormous error:
    sm0077 stores Hipparcos epochs as days since JD 2440000.0, and 7968 read
    literally is July 4713 BC rather than 1990 - a 6700-year mistake that no
    downstream validation rejects.
    """

    def test_offset_is_detected_from_the_label(self):
        frame = pd.DataFrame({"Measurement epoch with respect to JD 2440000.0, epoch": [7967.96]})

        layout = detect_epoch_layout(frame)

        assert layout.kind == "jd"
        assert layout.offset_days == pytest.approx(2440000.0)

    def test_offset_is_applied_to_the_value(self):
        frame = pd.DataFrame({"Measurement epoch with respect to JD 2440000.0, epoch": [7967.96]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        # 2440000.0 + 7967.96 is 1990-07-15, not 4713 BC.
        expected = to_seconds_since_j2000(pd.DataFrame({"JD": [2447967.96]}))[EPOCH_COLUMN].iloc[0]
        assert seconds == pytest.approx(expected)

    def test_relative_to_without_the_literal_jd_token(self):
        # sm0085/sm0087 label the column this way; "jd" never appears, so the
        # Julian finder misses it and detection has to key on the declaration.
        frame = pd.DataFrame(
            {"Gaia-centric epoch in TCB relative to 2455197.5 (Epoch_TCB, days)": [1913.69]}
        )

        layout = detect_epoch_layout(frame)

        assert layout.kind == "jd"
        assert layout.offset_days == pytest.approx(2455197.5)

    def test_relative_to_value_lands_in_the_gaia_era(self):
        frame = pd.DataFrame(
            {"Gaia-centric epoch in TCB relative to 2455197.5 (Epoch_TCB, days)": [1913.69]}
        )

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        # 2455197.5 + 1913.69 is early 2016, the Gaia data-release era.
        years = 2000.0 + seconds / (365.25 * 86400.0)
        assert 2015.0 < years < 2017.0

    def test_offset_is_detected_on_mjd_columns_too(self):
        frame = pd.DataFrame({"Modified Julian date relative to 50000.0": [1234.5]})

        layout = detect_epoch_layout(frame)

        assert layout.kind == "mjd"
        assert layout.offset_days == pytest.approx(50000.0)

    def test_absolute_jd_label_is_not_treated_as_an_offset(self):
        # A label naming a real, absolute date while storing absolute dates must
        # not have its own number subtracted from it.
        frame = pd.DataFrame({"JD 2451545.0 epoch of observation": [2451545.0]})

        layout = detect_epoch_layout(frame)

        assert layout.offset_days == 0.0

    def test_labels_without_a_reference_carry_no_offset(self):
        frame = pd.DataFrame({"JD of the moment of observation": [2451545.0]})

        assert detect_epoch_layout(frame).offset_days == 0.0

    def test_calendar_layout_carries_no_offset(self):
        frame = pd.DataFrame(
            {
                "Year of observation": [2012],
                "Month of observation": [6],
                "Day of observation": [30],
            }
        )

        assert detect_epoch_layout(frame).offset_days == 0.0

    def test_offset_participates_in_equality(self):
        def layout(offset):
            return EpochLayout("jd", {"jd": "JD"}, offset_days=offset)

        assert layout(0.0) == layout(0.0)
        assert layout(0.0) != layout(2440000.0)


class TestToSecondsSinceJ2000:
    def test_jd_j2000_is_zero(self):
        frame = pd.DataFrame({"JD": [JD_J2000]})

        assert to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0] == pytest.approx(0.0)

    def test_jd_one_day_later(self):
        frame = pd.DataFrame({"JD": [JD_J2000 + 1.0]})

        assert to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0] == pytest.approx(SECONDS_PER_DAY)

    def test_mjd_j2000_is_zero(self):
        frame = pd.DataFrame({"MJD of observation": [JD_J2000 - MJD_OFFSET_DAYS]})

        assert to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0] == pytest.approx(0.0)

    def test_mjd_is_not_read_as_jd(self):
        # A Julian-Date misreading would give a value ~2.08e11 s away.
        frame = pd.DataFrame({"MJD of observation": [58000.0]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx((58000.0 + MJD_OFFSET_DAYS - JD_J2000) * SECONDS_PER_DAY)

    def test_iso_matches_tudat(self):
        frame = pd.DataFrame({"iso_time": ["2012-06-30T11:11:37"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx(tudat_seconds(2012, 6, 30, 11, 11, 37))

    def test_compact_iso_date_only(self):
        # A bare date has no time of day; DateTime requires one, so it is padded
        # with midnight rather than rejected.
        frame = pd.DataFrame({"iso_time": ["2000-01-01"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx(-SECONDS_PER_DAY / 2)

    def test_space_separated_time_is_accepted(self):
        frame = pd.DataFrame({"iso_time": ["2012-06-30 11:11:37"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx(tudat_seconds(2012, 6, 30, 11, 11, 37))

    def test_trailing_z_is_resolved_to_utc(self):
        # DateTime rejects the "Z" designator, so it is normalised first.
        frame = pd.DataFrame({"iso_time": ["2012-06-30T11:11:37Z"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx(tudat_seconds(2012, 6, 30, 11, 11, 37))

    def test_numeric_utc_offset_is_resolved_to_utc(self):
        # The offset states the clock explicitly, so a +02:00 instant is the same
        # epoch as the corresponding UTC instant.
        frame = pd.DataFrame({"iso_time": ["2012-06-30T13:11:37+02:00"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx(tudat_seconds(2012, 6, 30, 11, 11, 37))

    def test_unparseable_iso_becomes_nan(self):
        frame = pd.DataFrame({"iso_time": ["not a date"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert np.isnan(seconds)

    def test_impossible_iso_date_becomes_nan(self):
        # Consistency with the calendar path, which raises on such a date: both
        # end up as a logged NaN rather than aborting a whole dataset.
        frame = pd.DataFrame({"iso_time": ["2012-02-30T00:00:00"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert np.isnan(seconds)

    def test_null_iso_stays_nan(self):
        frame = pd.DataFrame({"iso_time": [None, "2000-01-01"]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].to_numpy()

        assert np.isnan(seconds[0])
        assert seconds[1] == pytest.approx(-SECONDS_PER_DAY / 2)

    def test_calendar_matches_tudat(self):
        frame = pd.DataFrame(
            {
                "Year of observation": [2015],
                "Month of observation": [6],
                "Day of observation": [15],
                "Hour of observation": [8],
                "Minute of observation": [30],
                "Second of observation": [15.0],
            }
        )

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0]

        assert seconds == pytest.approx(tudat_seconds(2015, 6, 15, 8, 30, 15))

    def test_nan_jd_stays_nan(self):
        frame = pd.DataFrame({"JD": [np.nan]})

        assert np.isnan(to_seconds_since_j2000(frame)[EPOCH_COLUMN].iloc[0])

    def test_does_not_mutate_by_default(self):
        frame = pd.DataFrame({"JD": [JD_J2000]})

        to_seconds_since_j2000(frame)

        assert EPOCH_COLUMN not in frame.columns

    def test_in_place_adds_the_column(self):
        frame = pd.DataFrame({"JD": [JD_J2000]})

        result = to_seconds_since_j2000(frame, in_place=True)

        assert result is frame
        assert EPOCH_COLUMN in frame.columns

    def test_custom_column_name(self):
        frame = pd.DataFrame({"JD": [JD_J2000]})

        result = to_seconds_since_j2000(frame, column="epoch_TDB")

        assert "epoch_TDB" in result.columns

    def test_accepts_an_explicit_layout(self):
        frame = pd.DataFrame({"weird name": [JD_J2000]})
        layout = detect_epoch_layout(pd.DataFrame({"JD": [JD_J2000]}))

        # Explicit layout pointing at the real column.
        layout = type(layout)("jd", {"jd": "weird name"})

        assert to_seconds_since_j2000(frame, layout)[EPOCH_COLUMN].iloc[0] == pytest.approx(0.0)

    def test_multiple_rows(self):
        frame = pd.DataFrame({"JD": [JD_J2000, JD_J2000 + 1.0, JD_J2000 + 2.0]})

        seconds = to_seconds_since_j2000(frame)[EPOCH_COLUMN].to_numpy()

        assert seconds == pytest.approx([0.0, SECONDS_PER_DAY, 2 * SECONDS_PER_DAY])


class TestRoundTrip:
    def test_zero_seconds_is_j2000_jd(self):
        assert seconds_since_j2000_to_julian_date(np.array([0.0]))[0] == pytest.approx(JD_J2000)

    def test_jd_to_seconds_to_jd(self):
        original = np.array([JD_J2000, 2456108.966395065, 2415020.5])

        seconds = (original - JD_J2000) * SECONDS_PER_DAY

        assert seconds_since_j2000_to_julian_date(seconds) == pytest.approx(original)

    def test_zero_seconds_is_the_j2000_epoch(self):
        assert seconds_since_j2000_to_datetime(0.0).isoformat() == "2000-01-01T12:00:00"

    def test_one_day(self):
        assert seconds_since_j2000_to_datetime(SECONDS_PER_DAY).isoformat() == (
            "2000-01-02T12:00:00"
        )

    def test_negative_seconds(self):
        assert seconds_since_j2000_to_datetime(-SECONDS_PER_DAY / 2).isoformat() == (
            "2000-01-01T00:00:00"
        )

    def test_historical_epoch_is_not_out_of_range(self):
        # The corpus reaches back to 1847; pandas timestamps stop at 1677.
        assert seconds_since_j2000_to_datetime(-4702017601.0).isoformat() == ("1850-12-31T23:59:59")

    def test_fractional_seconds_are_preserved(self):
        assert seconds_since_j2000_to_datetime(645466801.5).microsecond == 500000

    def test_round_trip_through_datetime(self):
        original = np.array([-4702017601.0, 0.0, 645466801.5])

        rendered = np.array(
            [datetime_to_seconds(seconds_since_j2000_to_datetime(value)) for value in original]
        )

        assert rendered == pytest.approx(original)


class TestLeapSecondConvention:
    """The convention must be TudatPy's uniform day, not astropy's UTC Julian Date."""

    def test_leap_second_day_matches_tudat(self):
        # 2012-06-30 carried a leap second.  A uniform 86400 s day puts
        # 11:11:37 at 40297/86400 through the day; astropy's UTC JD at 40297/86401.
        out = calendar_seconds(2012, 6, 30, 11, 11, 37.0)

        assert out == pytest.approx(tudat_seconds(2012, 6, 30, 11, 11, 37))

    def test_astropy_would_differ(self):
        from astropy.time import Time

        astropy_seconds = (Time("2012-06-30T11:11:37", scale="utc").jd - JD_J2000) * (
            SECONDS_PER_DAY
        )
        ours = calendar_seconds(2012, 6, 30, 11, 11, 37.0)

        # Documents why astropy is not the reference: ~0.47 s apart.
        assert abs(astropy_seconds - ours) > 0.4
