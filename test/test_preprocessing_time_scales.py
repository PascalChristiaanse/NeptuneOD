"""Tests for timescale conversion to TDB seconds since J2000."""

import numpy as np
import pytest
from tudatpy.astro import time_representation as time_rep

from orbitdet.preprocessing.time_scales import (
    ASTRONOMICAL_DAY_OFFSET,
    ASTRONOMICAL_DAY_SWITCH_EPOCH,
    LOCAL_MERIDIAN_TIMES,
    MERIDIAN_LONGITUDES,
    InputTimeScale,
    MeridianTime,
    delta_t_seconds,
    describe_timescale,
    longitude_to_time_offset,
    meridian_label_offset,
    resolve_timescale,
    to_tdb_seconds_since_j2000,
)

#: Epochs used across the tests, as raw calendar offsets (s since J2000).
EPOCH_J2000 = time_rep.iso_string_to_epoch("2000-01-01T12:00:00")
EPOCH_1900 = time_rep.iso_string_to_epoch("1900-01-01T00:00:00")
EPOCH_1924 = time_rep.iso_string_to_epoch("1924-12-31T00:00:00")
EPOCH_1925 = time_rep.iso_string_to_epoch("1925-01-01T00:00:00")
EPOCH_1963 = time_rep.iso_string_to_epoch("1963-03-05T00:00:00")

#: Tolerance for comparisons of UT1-based offsets, in seconds.
#:
#: UT1 -> TDB is epoch-dependent (it involves the Earth's rotation angle), so an
#: offset applied in UT1 is not exactly preserved in TDB.  The residual is under
#: a millisecond, far below the precision of historical photographic astrometry.
OFFSET_TOLERANCE = 1e-3

#: Tolerance for pre-1961 comparisons, in seconds.
#:
#: Before 1961 the UT1 conversion goes through the Delta T model, whose value
#: changes slightly between the shifted and unshifted epoch.  A few milliseconds
#: of variation is expected and physically meaningless at this precision.
HISTORICAL_TOLERANCE = 1e-2


class TestAstronomicalDayCorrection:
    """Before 1925 the astronomical day ran noon to noon."""

    def test_pre_1925_epoch_is_shifted_by_twelve_hours(self):
        out = to_tdb_seconds_since_j2000([EPOCH_1900], "TDB")[0]

        assert out - EPOCH_1900 == pytest.approx(ASTRONOMICAL_DAY_OFFSET, abs=1e-6)

    def test_last_day_of_1924_is_still_shifted(self):
        out = to_tdb_seconds_since_j2000([EPOCH_1924], "TDB")[0]

        assert out - EPOCH_1924 == pytest.approx(ASTRONOMICAL_DAY_OFFSET, abs=1e-6)

    def test_epoch_exactly_at_switchover_is_not_shifted(self):
        out = to_tdb_seconds_since_j2000([EPOCH_1925], "TDB")[0]

        assert out == pytest.approx(EPOCH_1925, abs=1e-6)

    def test_post_1925_epoch_is_not_shifted(self):
        out = to_tdb_seconds_since_j2000([EPOCH_1963], "TDB")[0]

        assert out == pytest.approx(EPOCH_1963, abs=1e-6)

    def test_shift_applies_regardless_of_timescale(self):
        # The astronomical day is a dating convention, not a timescale, so it
        # applies to UTC, GMT and mean times alike.  Comparing each scale against
        # itself with the shift disabled isolates the 12 h from the timescale
        # conversion.
        for scale in ("UTC", "GMT", "UT1", "Washington Mean Time"):
            shifted = to_tdb_seconds_since_j2000([EPOCH_1900], scale)[0]
            unshifted = to_tdb_seconds_since_j2000([EPOCH_1900], scale, allow_pre_1925_shift=False)[
                0
            ]
            assert shifted - unshifted == pytest.approx(
                ASTRONOMICAL_DAY_OFFSET, abs=HISTORICAL_TOLERANCE
            ), scale

    def test_shift_can_be_disabled(self):
        out = to_tdb_seconds_since_j2000([EPOCH_1900], "TDB", allow_pre_1925_shift=False)[0]

        assert out == pytest.approx(EPOCH_1900, abs=1e-9)

    def test_mixed_eras_shift_only_the_pre_1925_rows(self):
        out = to_tdb_seconds_since_j2000([EPOCH_1900, EPOCH_1963], "TDB")

        assert out[0] - EPOCH_1900 == pytest.approx(ASTRONOMICAL_DAY_OFFSET, abs=1e-6)
        assert out[1] == pytest.approx(EPOCH_1963, abs=1e-6)

    def test_nan_propagates_without_being_shifted(self):
        out = to_tdb_seconds_since_j2000([np.nan, EPOCH_1900], "TDB")

        assert np.isnan(out[0])
        assert out[1] - EPOCH_1900 == pytest.approx(ASTRONOMICAL_DAY_OFFSET, abs=1e-6)

    def test_switchover_constant_matches_1925_january_1(self):
        assert ASTRONOMICAL_DAY_SWITCH_EPOCH == pytest.approx(EPOCH_1925, abs=1.0)


class TestTimescaleResolution:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("UTC", InputTimeScale.UTC),
            ("UT1", InputTimeScale.UT1),
            ("UT", InputTimeScale.UT1),
            ("TT", InputTimeScale.TT),
            ("TDB", InputTimeScale.TDB),
            ("TCB", InputTimeScale.TCB),
            ("TCG", InputTimeScale.TCG),
            ("TAI", InputTimeScale.TAI),
            ("GMT", InputTimeScale.UT1),
            ("G.M.T.", InputTimeScale.UT1),
            ("Greenwich Mean Time", InputTimeScale.UT1),
        ],
    )
    def test_modern_scales(self, label, expected):
        assert resolve_timescale(label).scale is expected

    @pytest.mark.parametrize(
        ("label", "meridian"),
        [
            ("Washington Mean Time", MeridianTime.WASHINGTON),
            ("Washington  M. T.", MeridianTime.WASHINGTON),
            ("Washington M.T.", MeridianTime.WASHINGTON),
            ("washington mean time", MeridianTime.WASHINGTON),
            ("WMT", MeridianTime.WASHINGTON),
            ("Paris M. T.", MeridianTime.PARIS),
            ("Paris Mean Time", MeridianTime.PARIS),
            ("Pulkovo M.T.", MeridianTime.PULKOVO),
            ("Pulkovo Mean Time", MeridianTime.PULKOVO),
            ("M.Z.Strasb", MeridianTime.STRASBOURG),
            ("T.m.de Nice", MeridianTime.NICE),
            ("Mt. Hamilton M. T.", MeridianTime.MOUNT_HAMILTON),
            ("Leander McCormick mean time", MeridianTime.LEANDER_MCCORMICK),
        ],
    )
    def test_local_mean_times_resolve_to_a_meridian(self, label, meridian):
        resolution = resolve_timescale(label)

        assert resolution.scale is InputTimeScale.LOCAL_MEAN_TIME
        assert resolution.meridian is meridian

    @pytest.mark.parametrize(
        ("label", "hours"),
        [
            ("Pacific Standard Time", -8.0),
            ("Standard Pacific Time", -8.0),
            ("Pacific standard time", -8.0),
            ("P.S.T. (Pacific S. T.)", -8.0),
            ("Central Standard Time", -6.0),
            ("Central Stand. Time", -6.0),
            ("90th Meridian Time = UTC - 6 h", -6.0),
            ("90 degree time", -6.0),
        ],
    )
    def test_zone_times_resolve_to_an_offset(self, label, hours):
        resolution = resolve_timescale(label)

        assert resolution.scale is InputTimeScale.STANDARD_ZONE
        assert resolution.zone_offset_seconds == pytest.approx(hours * 3600.0)

    def test_gmt_equals_utc_label_resolves_to_utc(self):
        # The source explicitly equates them, which is more information than a
        # bare "GMT".
        assert resolve_timescale("GMT=UTC").scale is InputTimeScale.UTC

    def test_bare_mean_time_is_station_dependent(self):
        resolution = resolve_timescale("some observatory mean time")

        assert resolution.scale is InputTimeScale.LOCAL_MEAN_TIME
        assert resolution.meridian is None
        assert resolution.needs_meridian

    def test_sidereal_label_is_recognised(self):
        assert resolve_timescale("local sidereal time").scale is InputTimeScale.LOCAL_SIDEREAL_TIME

    @pytest.mark.parametrize("label", ["", "no information", "no inf", "banana", "???"])
    def test_unknown_labels_resolve_to_unknown(self, label):
        assert resolve_timescale(label).scale is InputTimeScale.UNKNOWN

    def test_unknown_label_warns(self, caplog):
        with caplog.at_level("WARNING"):
            resolve_timescale("banana")

        assert any("Unrecognised timescale label" in record.message for record in caplog.records)

    def test_unknown_label_falls_back_to_utc_without_raising(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "banana")

        assert np.isfinite(out[0])

    def test_resolution_preserves_the_raw_label(self):
        resolution = resolve_timescale("Washington  M. T.")

        assert resolution.raw_label == "Washington  M. T."


class TestMeridianOffsets:
    def test_washington_is_west_of_greenwich(self):
        assert meridian_label_offset(MeridianTime.WASHINGTON) < 0

    def test_paris_and_pulkovo_are_east_of_greenwich(self):
        assert meridian_label_offset(MeridianTime.PARIS) > 0
        assert meridian_label_offset(MeridianTime.PULKOVO) > 0

    def test_pulkovo_is_further_east_than_paris(self):
        assert meridian_label_offset(MeridianTime.PULKOVO) > meridian_label_offset(
            MeridianTime.PARIS
        )

    def test_greenwich_is_zero(self):
        assert meridian_label_offset(MeridianTime.GREENWICH) == 0.0

    def test_washington_offset_is_about_minus_five_hours(self):
        # 77.0667 W is 5h08m20s behind Greenwich.
        assert meridian_label_offset(MeridianTime.WASHINGTON) / 3600.0 == pytest.approx(
            -5.1378, abs=1e-3
        )

    def test_longitude_to_time_offset_full_circle(self):
        assert longitude_to_time_offset(2 * np.pi) == pytest.approx(86400.0)

    def test_longitude_to_time_offset_quarter_circle(self):
        assert longitude_to_time_offset(np.pi / 2) == pytest.approx(21600.0)

    def test_longitude_to_time_offset_is_signed(self):
        assert longitude_to_time_offset(-np.pi / 2) == pytest.approx(-21600.0)

    def test_every_named_meridian_has_a_longitude(self):
        for meridian in MeridianTime:
            assert meridian in MERIDIAN_LONGITUDES

    def test_local_meridian_times_map_covers_every_meridian(self):
        for meridian in MeridianTime:
            assert LOCAL_MERIDIAN_TIMES[meridian] is InputTimeScale.LOCAL_MEAN_TIME


class TestConversionCorrectness:
    """Compare against the timescale conversions performed by TudatPy directly."""

    def test_tdb_is_the_identity(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "TDB")[0]

        assert out == pytest.approx(EPOCH_J2000, abs=1e-9)

    def test_utc_matches_tudatpy_directly(self):
        expected = time_rep.default_time_scale_converter().convert_time(
            time_rep.TimeScales.utc_scale, time_rep.TimeScales.tdb_scale, EPOCH_J2000
        )

        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "UTC")[0]

        assert out == pytest.approx(expected, abs=1e-6)

    def test_tt_matches_tudatpy_directly(self):
        expected = time_rep.default_time_scale_converter().convert_time(
            time_rep.TimeScales.tt_scale, time_rep.TimeScales.tdb_scale, EPOCH_J2000
        )

        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "TT")[0]

        assert out == pytest.approx(expected, abs=1e-6)

    def test_gmt_matches_ut1(self):
        gmt = to_tdb_seconds_since_j2000([EPOCH_J2000], "GMT")[0]
        ut1 = to_tdb_seconds_since_j2000([EPOCH_J2000], "UT1")[0]

        assert gmt == pytest.approx(ut1, abs=1e-9)

    def test_tcb_matches_tudatpy_helper(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "TCB")[0]

        assert out == pytest.approx(time_rep.TCB_to_TDB(EPOCH_J2000), abs=1e-6)

    def test_tcg_chains_through_tt(self):
        # TCG -> TT -> TDB, matching the IAU 2006 relation then TudatPy.
        tt = time_rep.TCG_to_TT(EPOCH_J2000)
        expected = time_rep.default_time_scale_converter().convert_time(
            time_rep.TimeScales.tt_scale, time_rep.TimeScales.tdb_scale, tt
        )

        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "TCG")[0]

        assert out == pytest.approx(expected, abs=1e-6)

    def test_tcb_offset_is_about_minus_11_seconds_at_j2000(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "TCB")[0]

        assert out - EPOCH_J2000 == pytest.approx(-11.25, abs=0.1)

    def test_utc_tdb_offset_is_about_64_seconds_at_j2000(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "UTC")[0]

        assert out - EPOCH_J2000 == pytest.approx(64.18, abs=0.5)


class TestMeridianAndZoneApplication:
    """A local mean time must be shifted by its meridian before conversion."""

    def test_named_meridian_shift_is_applied(self):
        plain = to_tdb_seconds_since_j2000([EPOCH_J2000], "UT1")[0]
        washington = to_tdb_seconds_since_j2000([EPOCH_J2000], "Washington Mean Time")[0]

        assert washington - plain == pytest.approx(
            meridian_label_offset(MeridianTime.WASHINGTON), abs=OFFSET_TOLERANCE
        )

    def test_station_longitude_overrides_named_meridian(self):
        # Greenwich has a zero meridian offset, so the result must match UT1.
        plain = to_tdb_seconds_since_j2000([EPOCH_J2000], "UT1")[0]
        via_longitude = to_tdb_seconds_since_j2000(
            [EPOCH_J2000], "Washington Mean Time", station_longitude_radians=0.0
        )[0]

        assert via_longitude == pytest.approx(plain, abs=OFFSET_TOLERANCE)

    def test_station_longitude_of_30_degrees_east(self):
        plain = to_tdb_seconds_since_j2000([EPOCH_J2000], "UT1")[0]
        via_longitude = to_tdb_seconds_since_j2000(
            [EPOCH_J2000],
            "some observatory mean time",
            station_longitude_radians=np.deg2rad(30.0),
        )[0]

        assert via_longitude - plain == pytest.approx(7200.0, abs=OFFSET_TOLERANCE)

    def test_zone_offset_is_applied(self):
        plain = to_tdb_seconds_since_j2000([EPOCH_J2000], "UT1")[0]
        pacific = to_tdb_seconds_since_j2000([EPOCH_J2000], "Pacific Standard Time")[0]

        assert pacific - plain == pytest.approx(-8.0 * 3600.0, abs=OFFSET_TOLERANCE)

    def test_local_mean_time_without_meridian_or_station_raises(self):
        with pytest.raises(ValueError, match="local mean time"):
            to_tdb_seconds_since_j2000([EPOCH_J2000], "some observatory mean time")

    def test_meridian_shift_combines_with_astronomical_day_shift(self):
        # A 1900 Washington mean time needs both corrections.
        out = to_tdb_seconds_since_j2000([EPOCH_1900], "Washington Mean Time")[0]
        unshifted = to_tdb_seconds_since_j2000(
            [EPOCH_1900], "Washington Mean Time", allow_pre_1925_shift=False
        )[0]

        assert out - unshifted == pytest.approx(ASTRONOMICAL_DAY_OFFSET, abs=HISTORICAL_TOLERANCE)


class TestInputHandling:
    def test_accepts_a_list(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000, EPOCH_1963], "TDB")

        assert len(out) == 2

    def test_accepts_a_numpy_array(self):
        out = to_tdb_seconds_since_j2000(np.array([EPOCH_J2000]), "TDB")

        assert isinstance(out, np.ndarray)

    def test_accepts_a_pandas_series(self):
        import pandas as pd

        out = to_tdb_seconds_since_j2000(pd.Series([EPOCH_J2000]), "TDB")

        assert isinstance(out, np.ndarray)
        assert out[0] == pytest.approx(EPOCH_J2000, abs=1e-9)

    def test_returns_float_array_of_matching_length(self):
        out = to_tdb_seconds_since_j2000([EPOCH_J2000] * 5, "UTC")

        assert out.shape == (5,)
        assert out.dtype == np.float64

    def test_accepts_a_timescale_enum(self):
        from_enum = to_tdb_seconds_since_j2000([EPOCH_J2000], InputTimeScale.TDB)[0]
        from_label = to_tdb_seconds_since_j2000([EPOCH_J2000], "TDB")[0]

        assert from_enum == pytest.approx(from_label, abs=1e-12)

    def test_empty_input_returns_empty_array(self):
        out = to_tdb_seconds_since_j2000([], "UTC")

        assert out.shape == (0,)

    def test_all_nan_input_returns_all_nan(self):
        out = to_tdb_seconds_since_j2000([np.nan, np.nan], "UTC")

        assert np.isnan(out).all()


class TestDescribeTimescale:
    def test_mentions_the_astronomical_day(self):
        assert "astronomical day" in describe_timescale("Washington Mean Time")

    def test_mentions_the_meridian_for_local_mean_time(self):
        assert "washington" in describe_timescale("WMT").lower()

    def test_mentions_the_zone_offset(self):
        assert "-8.0 h" in describe_timescale("Pacific Standard Time")

    def test_handles_a_plain_scale(self):
        assert "utc" in describe_timescale("UTC").lower()


class TestDeltaT:
    """TudatPy cannot convert UT1 before 1961, so a Delta T model fills the gap."""

    def test_delta_t_matches_published_values(self):
        # Espenak & Meeus give these values for 1900 January 1 and 1950 January 1.
        epochs = np.array(
            [
                time_rep.iso_string_to_epoch("1900-01-01T00:00:00"),
                time_rep.iso_string_to_epoch("1950-01-01T00:00:00"),
            ]
        )

        out = delta_t_seconds(epochs)

        assert out[0] == pytest.approx(-2.7, abs=0.5)
        assert out[1] == pytest.approx(29.1, abs=0.5)

    def test_delta_t_increases_through_the_twentieth_century(self):
        epochs = np.array(
            [
                time_rep.iso_string_to_epoch(f"{year}-01-01T00:00:00")
                for year in (1900, 1920, 1940, 1950, 1960, 1970)
            ]
        )

        out = delta_t_seconds(epochs)

        assert np.all(np.diff(out) > 0)

    def test_delta_t_is_continuous_across_polynomial_boundaries(self):
        # The four polynomials must join smoothly or the conversion would step.
        for boundary in (1920, 1941, 1961):
            before = delta_t_seconds(
                np.array([time_rep.iso_string_to_epoch(f"{boundary}-01-01T00:00:00") - 3600.0])
            )[0]
            after = delta_t_seconds(
                np.array([time_rep.iso_string_to_epoch(f"{boundary}-01-01T00:00:00") + 3600.0])
            )[0]

            assert after - before == pytest.approx(0.0, abs=0.5), f"step at {boundary}"

    def test_delta_t_warns_outside_the_modelled_range(self, caplog):
        with caplog.at_level("WARNING"):
            delta_t_seconds(np.array([time_rep.iso_string_to_epoch("1850-01-01T00:00:00")]))

        assert any("outside the modelled" in record.message for record in caplog.records)

    def test_pre_1961_ut1_is_converted_not_rejected(self):
        # This used to raise RuntimeError; the Delta T fallback makes it work.
        out = to_tdb_seconds_since_j2000([EPOCH_1900], "UT1", allow_pre_1925_shift=False)

        assert np.isfinite(out[0])
        assert out[0] - EPOCH_1900 == pytest.approx(-2.7, abs=0.5)

    def test_pre_1961_gmt_matches_ut1(self):
        gmt = to_tdb_seconds_since_j2000([EPOCH_1900], "GMT", allow_pre_1925_shift=False)[0]
        ut1 = to_tdb_seconds_since_j2000([EPOCH_1900], "UT1", allow_pre_1925_shift=False)[0]

        assert gmt == pytest.approx(ut1, abs=1e-6)

    def test_post_1961_ut1_still_uses_tudatpy(self):
        # The fallback must not intercept epochs TudatPy handles natively.
        out = to_tdb_seconds_since_j2000([EPOCH_J2000], "UT1")[0]
        expected = time_rep.default_time_scale_converter().convert_time(
            time_rep.TimeScales.ut1_scale, time_rep.TimeScales.tdb_scale, EPOCH_J2000
        )

        assert out == pytest.approx(expected, abs=1e-6)

    def test_mixed_era_ut1_conversion_handles_both_paths(self):
        epochs = [EPOCH_1900, EPOCH_J2000]

        out = to_tdb_seconds_since_j2000(epochs, "UT1", allow_pre_1925_shift=False)

        assert np.isfinite(out).all()
        assert out[0] - EPOCH_1900 == pytest.approx(-2.7, abs=0.5)
        assert out[1] - EPOCH_J2000 == pytest.approx(63.83, abs=0.5)

    def test_historical_ut1_conversion_logs_the_fallback(self, caplog):
        with caplog.at_level("INFO"):
            to_tdb_seconds_since_j2000([EPOCH_1900], "UT1", allow_pre_1925_shift=False)

        assert any("Delta T model" in record.message for record in caplog.records)

    def test_washington_mean_time_converts_at_1900(self):
        # Combines three corrections: 12 h astronomical day, Washington meridian,
        # and the Delta T fallback.
        out = to_tdb_seconds_since_j2000([EPOCH_1900], "Washington Mean Time")[0]

        expected = (
            EPOCH_1900 + ASTRONOMICAL_DAY_OFFSET + meridian_label_offset(MeridianTime.WASHINGTON)
        )
        assert out - expected == pytest.approx(-2.7, abs=0.5)
