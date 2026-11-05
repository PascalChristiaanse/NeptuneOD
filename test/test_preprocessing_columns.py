"""Tests for ``orbitdet.preprocessing.columns``."""

import numpy as np
import pandas as pd
import pytest

from orbitdet.preprocessing.columns import (
    ARCSECOND_TO_RADIANS,
    SECONDS_OF_TIME_TO_RADIANS,
    apply_format_columns,
    find_column,
    find_column_substring,
    normalize_column_name,
    numeric_series,
    relative_position_scale,
)


class TestNormalizeColumnName:
    def test_lowercases_and_collapses_padding(self):
        assert (
            normalize_column_name("Hour   of right ascension (alpha, h)")
            == "hour of right ascension (alpha, h)"
        )

    def test_underscores_become_spaces(self):
        assert normalize_column_name("scan_angle") == "scan angle"

    def test_strips_surrounding_whitespace(self):
        assert normalize_column_name("  Dec  ") == "dec"

    def test_non_string_input_is_stringified(self):
        assert normalize_column_name(7) == "7"


class TestFindColumn:
    def test_matches_all_needles_as_whole_words(self):
        frame = pd.DataFrame({"Hour of right ascension (alpha, h)": [1]})

        assert find_column(frame, "hour", "ascension") == "Hour of right ascension (alpha, h)"

    def test_returns_none_when_a_needle_is_missing(self):
        frame = pd.DataFrame({"Hour of right ascension (alpha, h)": [1]})

        assert find_column(frame, "hour", "declination") is None

    def test_jd_needle_also_matches_julian(self):
        frame = pd.DataFrame({"Julian date of observation": [1.0]})

        assert find_column(frame, "jd") == "Julian date of observation"

    def test_jd_needle_matches_literal_jd(self):
        frame = pd.DataFrame({"JD": [1.0]})

        assert find_column(frame, "jd") == "JD"

    def test_excludes_disqualify_whole_words(self):
        frame = pd.DataFrame({"Second of declination": [1], "Second of observation": [2]})

        found = find_column(frame, "second", excludes=["declination"])

        assert found == "Second of observation"

    def test_needle_matches_plural_column(self):
        frame = pd.DataFrame({"Seconds of the observation": [1]})

        assert find_column(frame, "second") == "Seconds of the observation"

    def test_needle_matches_singular_column(self):
        frame = pd.DataFrame({"Second of the observation": [1]})

        assert find_column(frame, "second") == "Second of the observation"

    def test_exclusion_is_word_bounded(self):
        frame = pd.DataFrame({"arcsec": [1]})

        # "ra" must not match inside "arcsec".
        assert find_column(frame, "ra", excludes=["declination"]) is None

    def test_returns_first_match_in_column_order(self):
        frame = pd.DataFrame({"RA one": [1], "RA two": [2]})

        assert find_column(frame, "ra") == "RA one"

    def test_no_columns_returns_none(self):
        assert find_column(pd.DataFrame(), "ra") is None


class TestFindColumnSubstring:
    def test_matches_substrings(self):
        frame = pd.DataFrame({"Second of right ascension (alpha, s)": [1]})

        assert find_column_substring(frame, "right", "ascension") is not None

    def test_returns_none_when_a_substring_is_missing(self):
        frame = pd.DataFrame({"Second of right ascension": [1]})

        assert find_column_substring(frame, "right", "declination") is None

    def test_excludes_are_substring_matches(self):
        frame = pd.DataFrame({"Second of observation": [1], "Second of declination": [2]})

        found = find_column_substring(frame, "second", excludes=["declination"])

        assert found == "Second of observation"


class TestNumericSeries:
    def test_reads_a_numeric_column(self):
        frame = pd.DataFrame({"x": ["1.5", "2.5"]})

        assert numeric_series(frame, "x").tolist() == [1.5, 2.5]

    def test_unparseable_values_become_nan(self):
        frame = pd.DataFrame({"x": ["1.5", "oops"]})

        result = numeric_series(frame, "x")

        assert result[0] == 1.5
        assert np.isnan(result[1])

    def test_missing_column_yields_zeros_of_matching_length(self):
        frame = pd.DataFrame({"x": [1, 2, 3]})

        result = numeric_series(frame, None)

        assert result.tolist() == [0.0, 0.0, 0.0]
        assert result.index.equals(frame.index)

    def test_absent_component_columns_are_zero_not_an_error(self):
        # A source recording RA in degrees alone has no minute/second column.
        frame = pd.DataFrame({"ra": [10.0]})

        assert numeric_series(frame, None).iloc[0] == 0.0


class TestApplyFormatColumns:
    def test_overlays_1_based_indices(self):
        frame = pd.DataFrame([[1, 2, 3]])

        apply_format_columns(frame, {1: "jd", 2: "ra", 3: "dec"})

        assert list(frame.columns) == ["jd", "ra", "dec"]

    def test_accepts_string_keys(self):
        frame = pd.DataFrame([[1, 2]])

        apply_format_columns(frame, {"1": "jd", "2": "ra"})

        assert list(frame.columns) == ["jd", "ra"]

    def test_partial_map_leaves_other_columns_positional(self):
        frame = pd.DataFrame([[1, 2, 3]])

        apply_format_columns(frame, {1: "jd"})

        assert list(frame.columns) == ["jd", 1, 2]

    def test_out_of_range_index_is_skipped_not_raised(self):
        # Fixed-width NSDB files collapse blank fields, so a dataset can declare
        # more Format entries than the whitespace split produces.
        frame = pd.DataFrame([[1, 2]])

        apply_format_columns(frame, {1: "jd", 2: "ra", 3: "cam", 4: "notes"})

        assert list(frame.columns) == ["jd", "ra"]

    def test_empty_map_is_a_no_op(self):
        frame = pd.DataFrame([[1, 2]])

        apply_format_columns(frame, {})

        assert list(frame.columns) == [0, 1]

    def test_none_map_is_a_no_op(self):
        frame = pd.DataFrame([[1, 2]])

        apply_format_columns(frame, None)

        assert list(frame.columns) == [0, 1]

    def test_none_value_leaves_that_column_positional(self):
        frame = pd.DataFrame([[1, 2]])

        apply_format_columns(frame, {1: None, 2: "ra"})

        assert list(frame.columns) == [0, "ra"]

    def test_returns_the_same_object(self):
        frame = pd.DataFrame([[1]])

        assert apply_format_columns(frame, {1: "jd"}) is frame

    def test_zero_index_is_out_of_range_for_a_1_based_map(self):
        frame = pd.DataFrame([[1, 2]])

        apply_format_columns(frame, {0: "bogus", 1: "jd"})

        assert list(frame.columns) == ["jd", 1]

    def test_numeric_ordering_is_not_lexicographic(self):
        # 10 must not be applied before 2.
        frame = pd.DataFrame([[0] * 10])

        apply_format_columns(frame, {i: f"c{i}" for i in range(1, 11)})

        assert list(frame.columns) == [f"c{i}" for i in range(1, 11)]


class TestRelativePositionScale:
    def test_arcsec(self):
        assert relative_position_scale("X, arcsec") == pytest.approx(ARCSECOND_TO_RADIANS)

    def test_arcsecond_spelling(self):
        assert relative_position_scale("delta, arcsecond") == pytest.approx(ARCSECOND_TO_RADIANS)

    def test_seconds_of_time(self):
        assert relative_position_scale("Delta alpha, sec of time") == pytest.approx(
            SECONDS_OF_TIME_TO_RADIANS
        )

    def test_seconds_of_time_full_spelling(self):
        assert relative_position_scale("Delta alpha, seconds of time") == pytest.approx(
            SECONDS_OF_TIME_TO_RADIANS
        )

    def test_degrees(self):
        assert relative_position_scale("x, degree") == pytest.approx(np.pi / 180.0)

    def test_deg_abbreviation(self):
        assert relative_position_scale("x, deg") == pytest.approx(np.pi / 180.0)

    def test_arcsecond_relation_to_seconds_of_time(self):
        # A second of time is 15 arcsec.
        assert SECONDS_OF_TIME_TO_RADIANS == pytest.approx(15 * ARCSECOND_TO_RADIANS)

    def test_unknown_unit_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="Could not infer angular units"):
            relative_position_scale("X, pixels")
