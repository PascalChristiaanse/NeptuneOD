"""Tests for the standard observation dataframe schema."""

import numpy as np
import pandas as pd
import pytest

from orbitdet.preprocessing import (
    ANGLE_COLUMNS,
    COLUMN_SPECS,
    NOT_GIVEN,
    NOT_GIVEN_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    STANDARD_DATAFRAME_COLUMNS,
    ObservationFrameError,
    apply_not_given,
    build_frame,
    empty_frame,
    is_not_given,
    verify,
)
from orbitdet.preprocessing.schema import (
    _RANGE_RULES,
    TWO_PI,
    CoordinateKind,
    DataSource,
    FrameCenter,
    ObservationType,
    ReceptorType,
    ReductionCatalogue,
    ReferenceFrame,
    describe,
)

#: A minimal valid record.  Every test starts from this and overrides one field,
#: so a failure points at exactly the field under test.
VALID_RECORD: dict[str, object] = {
    "observed_object": "Triton",
    "observatory_code": "673",
    "data_source": "nsdb",
    "source_set_id": "nm0001",
    "reference": "(2001) Communicated to NSDC by W. M. Owen Jr.",
    "observation_type": "absolute",
    "coordinate_kind": "ra_dec",
    "frame_center": "topocentric",
    "original_equinox": "J2000",
    "original_reference_frame": "astrometric",
    "original_reduction": "none",
    "receptor_type": "ccd",
    "epoch_tdb": 44492331.99169189,
    "first_value": 2.0,
    "second_value": -0.3,
}


def make_frame(**overrides: object) -> pd.DataFrame:
    """Build a one-row standard frame, overriding fields of ``VALID_RECORD``."""
    return build_frame([{**VALID_RECORD, **overrides}])


class TestSchemaDefinition:
    def test_column_names_are_unique(self):
        names = [spec.name for spec in STANDARD_DATAFRAME_COLUMNS]

        assert len(names) == len(set(names))

    def test_required_and_optional_partition_the_schema(self):
        all_names = {spec.name for spec in STANDARD_DATAFRAME_COLUMNS}

        assert set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS) == all_names
        assert not set(REQUIRED_COLUMNS) & set(OPTIONAL_COLUMNS)

    def test_column_specs_is_an_alias(self):
        assert COLUMN_SPECS is STANDARD_DATAFRAME_COLUMNS

    def test_angle_columns_are_declared_in_radians(self):
        by_name = {spec.name: spec for spec in STANDARD_DATAFRAME_COLUMNS}

        for column in ANGLE_COLUMNS:
            unit = by_name[column].unit
            # Angles are rad; proper motions are rad/s.  Both must be radian-based.
            assert unit is not None and unit.startswith("rad"), (
                f"{column} must be declared in radians, got {unit!r}"
            )

    def test_epoch_is_declared_in_seconds(self):
        by_name = {spec.name: spec for spec in STANDARD_DATAFRAME_COLUMNS}

        assert by_name["epoch_tdb"].unit == "s"
        assert "TDB" in by_name["epoch_tdb"].description

    def test_every_column_has_a_description(self):
        for spec in STANDARD_DATAFRAME_COLUMNS:
            assert spec.description.strip(), f"{spec.name} has no description"

    def test_describe_returns_one_row_per_column(self):
        table = describe()

        assert len(table) == len(STANDARD_DATAFRAME_COLUMNS)
        assert list(table.columns) == ["column", "dtype", "required", "unit", "description"]


class TestEmptyFrame:
    def test_has_every_column_in_canonical_order(self):
        frame = empty_frame()

        assert list(frame.columns) == [spec.name for spec in STANDARD_DATAFRAME_COLUMNS]

    def test_has_declared_dtypes(self):
        frame = empty_frame()

        for spec in STANDARD_DATAFRAME_COLUMNS:
            assert str(frame[spec.name].dtype) == spec.dtype, spec.name

    def test_is_valid(self):
        assert verify(empty_frame()) == []

    def test_can_be_rejected_when_empty_frames_are_disallowed(self):
        problems = verify(empty_frame(), strict=False, allow_empty=False)

        assert problems == ["dataframe contains no observations"]


class TestBuildFrame:
    def test_fills_omitted_columns_with_null(self):
        frame = build_frame([{"observed_object": "Triton"}])

        assert frame["observatory_code"].isna().all()
        assert frame["pm_first"].isna().all()

    def test_preserves_declared_dtypes_for_unpopulated_columns(self):
        frame = build_frame([{"observed_object": "Triton"}])

        assert str(frame["relative_to_object"].dtype) == "object"
        assert str(frame["pm_first"].dtype) == "float64"

    def test_returns_empty_frame_for_no_records(self):
        frame = build_frame([])

        assert frame.empty
        assert list(frame.columns) == [spec.name for spec in STANDARD_DATAFRAME_COLUMNS]

    def test_rejects_unknown_keys(self):
        with pytest.raises(KeyError, match="not standard observation columns"):
            build_frame([{**VALID_RECORD, "typo_column": 1}])

    def test_accepts_multiple_records(self):
        frame = build_frame([VALID_RECORD, {**VALID_RECORD, "epoch_tdb": 0.0}])

        assert len(frame) == 2
        assert verify(frame) == []


class TestVerifyAcceptsValidFrames:
    def test_minimal_valid_frame(self):
        assert verify(make_frame()) == []

    def test_relative_observation_with_relative_to_object(self):
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="cartesian_xy",
            relative_to_object="Neptune",
            first_value=1e-5,
            second_value=-2e-5,
        )

        assert verify(frame) == []

    def test_position_angle_separation(self):
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="position_angle_separation",
            relative_to_object="Neptune",
            first_value=1.2,
            second_value=3.5e-5,
        )

        assert verify(frame) == []

    def test_ra_in_negative_wrap_convention(self):
        frame = make_frame(first_value=-3.0)

        assert verify(frame) == []

    def test_ra_in_positive_wrap_convention(self):
        frame = make_frame(first_value=6.0)

        assert verify(frame) == []

    def test_declination_at_pole(self):
        frame = make_frame(second_value=np.pi / 2)

        assert verify(frame) == []

    def test_all_optional_columns_populated(self):
        frame = make_frame(
            relative_to_object="Neptune",
            first_unit="deg",
            second_unit="deg",
            catalogue_epoch=0.0,
            pm_first=1e-12,
            pm_second=-1e-12,
            first_uncertainty=1e-7,
            second_uncertainty=1e-7,
            scan_angle=0.5,
        )

        assert verify(frame) == []

    def test_scan_angle_is_optional(self):
        # Only scanning instruments record a scan angle, so a frame from every
        # other source must stay valid without one.
        assert "scan_angle" in OPTIONAL_COLUMNS
        assert verify(make_frame()) == []

    def test_scan_angle_may_be_null(self):
        frame = make_frame(scan_angle=np.nan)

        assert verify(frame) == []

    def test_scan_angle_rejects_a_sentinel(self):
        # The not-given marker means "this half was not measured", which is not a
        # statement a scan angle can make: a row either has one or does not.
        frame = make_frame(scan_angle=NOT_GIVEN)

        problems = verify(frame, strict=False)

        assert any("scan_angle" in p and "not-given" in p for p in problems)

    def test_spacecraft_frame_centre_is_accepted(self):
        # Voyager observes from the spacecraft itself, so the vocabulary has to
        # admit it as distinctly from a ground station.
        frame = make_frame(
            frame_center="spacecraft",
            observatory_code="Voyager 2",
            data_source="voyager",
            receptor_type="vidicon",
        )

        assert verify(frame) == []


class TestVerifyRejectsMissingColumns:
    def test_missing_required_column(self):
        frame = make_frame()
        del frame["reference"]

        problems = verify(frame, strict=False)

        assert any("missing required column" in p and "reference" in p for p in problems)

    def test_reports_all_missing_columns_at_once(self):
        frame = make_frame()
        del frame["reference"]
        del frame["receptor_type"]

        problems = verify(frame, strict=False)

        assert len(problems) == 1
        assert "reference" in problems[0] and "receptor_type" in problems[0]

    def test_absent_optional_column_is_fine(self):
        frame = make_frame()
        del frame["pm_first"]

        assert verify(frame) == []


class TestVerifyRejectsNulls:
    def test_null_in_required_column(self):
        frame = make_frame()
        frame.loc[0, "observed_object"] = None

        problems = verify(frame, strict=False)

        assert any("'observed_object' is required but has 1 null" in p for p in problems)

    def test_null_in_optional_column_is_fine(self):
        frame = make_frame()
        frame.loc[0, "relative_to_object"] = None

        assert verify(frame) == []


class TestVerifyRejectsBadDtypes:
    def test_text_in_numeric_column(self):
        frame = make_frame()
        frame["epoch_tdb"] = frame["epoch_tdb"].astype(object)
        frame.loc[0, "epoch_tdb"] = "not a number"

        problems = verify(frame, strict=False)

        assert any("'epoch_tdb' must be numeric" in p for p in problems)

    def test_number_in_text_column(self):
        frame = make_frame()
        frame["observed_object"] = [42]

        problems = verify(frame, strict=False)

        assert any("'observed_object' must be string-like" in p for p in problems)

    def test_all_null_column_is_accepted_regardless_of_dtype(self):
        # Pandas coerces an all-null object column to float64 on row assignment;
        # that must not be reported as a dtype violation.
        frame = make_frame()
        frame["relative_to_object"] = np.nan

        assert verify(frame) == []


class TestVerifyRejectsBadEnums:
    @pytest.mark.parametrize(
        ("column", "bad_value", "vocabulary"),
        [
            ("data_source", "NSDB", "DataSource"),
            ("observation_type", "Absolute", "ObservationType"),
            ("coordinate_kind", "radec", "CoordinateKind"),
            ("frame_center", "topocentre", "FrameCenter"),
            ("original_reference_frame", "Astrometric", "ReferenceFrame"),
            ("original_reduction", "UCAC2", "ReductionCatalogue"),
            ("receptor_type", "CCD", "ReceptorType"),
        ],
    )
    def test_value_outside_vocabulary(self, column, bad_value, vocabulary):
        frame = make_frame(**{column: bad_value})

        problems = verify(frame, strict=False)

        assert any(column in p and vocabulary in p for p in problems), problems

    def test_enum_error_lists_allowed_values(self):
        frame = make_frame(data_source="NSDB")

        problems = verify(frame, strict=False)

        assert any("allowed:" in p and "nsdb" in p for p in problems)

    def test_null_enum_value_is_not_a_vocabulary_violation(self):
        # Nulls in required columns are reported by the null check, not here.
        frame = make_frame()
        frame.loc[0, "data_source"] = None

        problems = verify(frame, strict=False)

        assert not any("outside DataSource" in p for p in problems)


class TestVerifyRejectsBadRanges:
    def test_degrees_instead_of_radians_is_caught(self):
        frame = make_frame(first_value=114.6, second_value=-17.9)

        problems = verify(frame, strict=False)

        assert any("first_value" in p and "radians, not degrees" in p for p in problems)
        assert any("second_value" in p and "radians, not degrees" in p for p in problems)

    def test_declination_beyond_pole(self):
        frame = make_frame(second_value=2.0)

        problems = verify(frame, strict=False)

        assert any("second_value" in p and "out of range" in p for p in problems)

    def test_negative_separation(self):
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="position_angle_separation",
            relative_to_object="Neptune",
            first_value=1.0,
            second_value=-1e-5,
        )

        problems = verify(frame, strict=False)

        assert any("second_value" in p and "out of range" in p for p in problems)

    def test_cartesian_xy_has_no_range_constraint(self):
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="cartesian_xy",
            relative_to_object="Neptune",
            first_value=1e-3,
            second_value=-1e-3,
        )

        assert verify(frame) == []

    def test_range_check_only_applies_to_matching_coordinate_kind(self):
        # A large first_value is fine for cartesian_xy even though it would be
        # out of range for ra_dec.
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="cartesian_xy",
            relative_to_object="Neptune",
            first_value=100.0,
            second_value=100.0,
        )

        assert verify(frame) == []

    def test_nan_values_are_skipped_by_range_check(self):
        frame = make_frame(first_value=np.nan)

        problems = verify(frame, strict=False)

        assert not any("out of range" in p for p in problems)


class TestNotGivenSentinel:
    """A source that records no measurement for one component of a pair.

    NSDB declares such a component with a sentinel in the column heading rather
    than omitting the row, so the observation is one-dimensional: a position
    angle with no separation, or a separation with no position angle.  The
    sentinel is kept (as ``NOT_GIVEN``) rather than turned into null, because the
    half that *was* measured is a real observation.
    """

    def test_not_given_is_excluded_from_range_checks(self):
        # The marker is deliberately out of range, so the range check must skip
        # it: otherwise every source that records a one-dimensional measurement
        # would be reported as holding a corrupt one.  This is the reason the
        # marker is not merely "a large number" - a separation's range is
        # [0, inf), which would happily admit 9999.
        assert _RANGE_RULES[CoordinateKind.POSITION_ANGLE_SEPARATION] == (
            ("first_value", -TWO_PI, TWO_PI),
            ("second_value", 0.0, np.inf),
        )
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="position_angle_separation",
            relative_to_object="Neptune",
            first_value=NOT_GIVEN,
            second_value=NOT_GIVEN,
        )

        assert verify(frame) == []

    def test_not_given_exceeds_the_real_values_in_the_corpus(self):
        # The corpus's real angles span [-0.39, 6.28] rad, i.e. at most one full
        # turn (2*pi = 6.283).  Every legitimate measurement is therefore orders
        # of magnitude below the marker, which is what makes it unambiguous.
        assert NOT_GIVEN > 100 * TWO_PI

    def test_is_not_given_detects_the_marker(self):
        values = np.array([1.0, NOT_GIVEN, np.nan, -2.0])

        assert is_not_given(values).tolist() == [False, True, False, False]

    def test_is_not_given_treats_null_as_ordinary(self):
        # Null and sentinel are different claims: only the sentinel says the
        # source considered the quantity and recorded nothing.
        assert is_not_given(np.array([np.nan])).tolist() == [False]

    def test_is_not_given_accepts_a_series(self):
        series = pd.Series([0.5, NOT_GIVEN])

        assert is_not_given(series).tolist() == [False, True]

    def test_apply_not_given_scales_real_values(self):
        out = apply_not_given(np.array([0.0, 90.0]), (999.0,), scale=np.pi / 180.0)

        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(np.pi / 2)

    def test_apply_not_given_marks_the_sentinel_without_scaling_it(self):
        out = apply_not_given(np.array([90.0, 999.0]), (999.0,), scale=np.pi / 180.0)

        assert out[1] == NOT_GIVEN

    def test_apply_not_given_accepts_higher_precision_sentinels(self):
        # NSDB declares 999 but stores 999.9 in some files and 999.99 in others.
        out = apply_not_given(np.array([999.0, 999.9, 999.99]), (999.0,))

        assert out.tolist() == [NOT_GIVEN, NOT_GIVEN, NOT_GIVEN]

    def test_apply_not_given_marks_nulls(self):
        out = apply_not_given(np.array([1.0, np.nan]), (999.0,))

        assert out[0] == 1.0
        assert out[1] == NOT_GIVEN

    def test_apply_not_given_applies_only_the_status_it_is_given(self):
        # The sentinel is scoped to its own column: the same number in a different
        # column is untouched.  A position angle declared with sentinel 999 leaves
        # 99.99 alone, because 99.99 is a real angle in degrees.
        out = apply_not_given(np.array([99.99, 999.0]), (999.0,))

        assert out[0] == pytest.approx(99.99)
        assert out[1] == NOT_GIVEN

    def test_apply_not_given_matches_a_sentinel_declared_per_column(self):
        # An arcminute field declares 99.  Its physical range stops at 60, so any
        # value reported near 99 came from the sentinel, not from a measurement.
        out = apply_not_given(np.array([59.9, 99.0]), (99.0,))

        assert out[0] == pytest.approx(59.9)
        assert out[1] == NOT_GIVEN

    def test_verify_accepts_the_marker_in_measurement_columns(self):
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="position_angle_separation",
            relative_to_object="Neptune",
            first_value=NOT_GIVEN,
            second_value=2e-5,
        )

        assert verify(frame) == []

    def test_verify_accepts_the_marker_on_either_component(self):
        frame = make_frame(
            observation_type="relative",
            coordinate_kind="position_angle_separation",
            relative_to_object="Neptune",
            first_value=1.0,
            second_value=NOT_GIVEN,
        )

        assert verify(frame) == []

    def test_verify_rejects_the_marker_in_a_forbidden_column(self):
        # A missing proper motion is a genuine gap, so it must stay null: writing
        # the sentinel there would claim the catalogue supplied a value.
        frame = make_frame()
        frame["pm_first"] = NOT_GIVEN

        problems = verify(frame, strict=False)

        assert any("pm_first" in p and "sentinel" in p for p in problems)

    def test_not_given_columns_lists_only_measurement_columns(self):
        assert set(NOT_GIVEN_COLUMNS) == {"first_value", "second_value"}

    def test_marker_is_a_plain_float(self):
        # Downstream code compares numerically, so the marker must not be a
        # NaN-with-payload or a subclass that changes arithmetic.
        assert type(NOT_GIVEN) is float
        assert np.isfinite(NOT_GIVEN)


class TestVerifyStrictMode:
    def test_raises_with_all_problems_listed(self):
        frame = make_frame(data_source="NSDB", first_value=114.6)

        with pytest.raises(ObservationFrameError) as excinfo:
            verify(frame)

        message = str(excinfo.value)
        assert "2 problem(s)" in message
        assert "data_source" in message
        assert "first_value" in message

    def test_non_strict_returns_problems(self):
        frame = make_frame(data_source="NSDB")

        problems = verify(frame, strict=False)

        assert isinstance(problems, list)
        assert len(problems) == 1

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
            verify({"observed_object": "Triton"})

    def test_error_is_a_value_error(self):
        assert issubclass(ObservationFrameError, ValueError)


class TestVocabularies:
    def test_observation_type_values(self):
        assert {m.value for m in ObservationType} == {"absolute", "relative"}

    def test_coordinate_kind_values(self):
        assert {m.value for m in CoordinateKind} == {
            "ra_dec",
            "position_angle_separation",
            "cartesian_xy",
        }

    def test_frame_center_values(self):
        assert {m.value for m in FrameCenter} == {
            "topocentric",
            "geocentric",
            "spacecraft",
            "barycentric",
            "heliocentric",
            "unknown",
        }

    def test_reference_frame_values(self):
        assert {m.value for m in ReferenceFrame} == {
            "astrometric",
            "apparent",
            "bcrs",
            "unknown",
        }

    def test_reduction_catalogue_values(self):
        assert {m.value for m in ReductionCatalogue} == {
            "none",
            "ucac2",
            "ucac4",
            "tycho2",
            "gaia_dr1",
            "gaia_dr2",
            "sao",
            "ast",
            "given_in_data",
            "unknown",
        }

    def test_data_source_values(self):
        assert {m.value for m in DataSource} == {
            "nsdb",
            "mast",
            "gaia_archive",
            "voyager",
            "manual",
            "simulated",
        }

    def test_receptor_type_values(self):
        assert {m.value for m in ReceptorType} == {
            "ccd",
            "photographic",
            "visual",
            "micrometric",
            "meridian_transit_circle",
            "vidicon",
            "unknown",
        }
