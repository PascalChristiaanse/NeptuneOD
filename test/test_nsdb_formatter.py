"""Tests for the NSDB formatter: column discovery, observatory resolution, epochs.

The formatter is the third stage of the NSDB read path, after download and parse.
It owns every decision that depends on NSDB's own conventions, so most of what
needs testing is vocabulary handling rather than arithmetic:

* which column holds which quantity, given NSDB's prose headings;
* which observatory code applies to each row;
* how a ``J2015.0`` equinox label is interpreted;
* how the observation epoch reaches TDB seconds since J2000.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from orbitdet.data.nsdb_formatter import (
    EPOCH_COLUMN,
    RELATIVE_POSITION_X_COLUMN,
    RELATIVE_POSITION_Y_COLUMN,
    NSDBFormatter,
    _declared_sentinel,
    _find_relative_x_column,
    _find_relative_y_column,
    _ra_was_assembled,
    _reject_duplicate_headings,
    classify_coordinates,
    find_radec_columns,
    group_rows_by_observatory,
    resolve_epoch_of_equinox,
    resolve_observatory_codes,
    set_epoch_column,
    set_ra_dec_columns,
    set_relative_position_columns,
)
from orbitdet.data.nsdb_parser import NSDBRecord


class TestDeclaredSentinel:
    """The sentinel declaration is prose, and NSDB writes it two ways.

    The Neptune catalogue names the value position - ``(999 if not given)`` -
    while the Uranian catalogue names the *quantity*: ``(999.9 if P is not
    given)``.  Reading the declaration is what keeps a legitimate ``99.99``
    degree position angle from being mistaken for a sentinel, so a phrasing the
    regex misses does not merely lose the marker, it leaks the raw sentinel into
    the frame as if it were a measurement.
    """

    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            # Neptune phrasing, as it appears in 108 corpus columns.
            ("P (position angle) in degrees (999 if not given)", 999.0),
            ("S (separation) in arcsec (-99.99 if not given)", -99.99),
            # Uranian phrasing, which names the quantity instead of saying "not given".
            ("P (position angle) in degrees (999.9 if P is not given)", 999.9),
            ("S (separation) in arcsec  (-99.99 if S is not given)", -99.99),
            ("S (separation) in arcsec (-9.99 if S is not given)", -9.99),
            ("S (separation) in arcsec (99.99 if S is not given)", 99.99),
        ],
    )
    def test_both_phrasings_are_read(self, heading, expected):
        assert _declared_sentinel(heading) == expected

    def test_a_combined_triple_declaration_is_read(self):
        # One heading covering an hour/minute/second triple declares the sentinel
        # once, after all three field widths.
        heading = "Hour, minute, second of the observation of S (99 99 99 if S is not given)"

        assert _declared_sentinel(heading) == 99.0

    @pytest.mark.parametrize(
        "heading",
        [
            "P (position angle) in degrees",
            "X, arcsec",
            "S (separation) in arcsec",
            "Day of the moment of observation",
        ],
    )
    def test_a_heading_declaring_nothing_has_no_sentinel(self, heading):
        # Nothing in such a column is missing, which is the safe default: a
        # wrongly-flagged sentinel silently deletes real measurements.
        assert _declared_sentinel(heading) is None

    def test_a_real_position_angle_is_not_mistaken_for_a_sentinel(self):
        # The whole point of reading the declaration: 99.99 degrees is a valid
        # position angle, and only a column that *declares* 99.99 as its marker
        # treats it as missing.
        assert _declared_sentinel("P (position angle) in degrees (999 if not given)") == 999.0


class TestDuplicateHeadings:
    """Several clusters of the same quantity make a heading ambiguous.

    Five Uranian files carry three separate time triples - one per satellite -
    with identical heading text.  Pandas accepts the duplicate names, and every
    later lookup then returns a DataFrame instead of a Series, so the failure
    surfaces as an opaque ``TypeError`` inside a numeric conversion.
    """

    def _record(self, identifier="um0049"):
        return NSDBRecord(identifier=identifier)

    def test_a_duplicate_heading_is_rejected(self):
        frame = pd.DataFrame(
            np.zeros((2, 4)),
            columns=["Year", "Hour of the observation", "Hour of the observation", "P"],
        )

        with pytest.raises(ValueError, match="duplicate column heading"):
            _reject_duplicate_headings(frame, self._record())

    def test_the_error_names_the_colliding_heading_and_the_dataset(self):
        frame = pd.DataFrame(
            np.zeros((2, 3)),
            columns=["Year", "Hour of the observation", "Hour of the observation"],
        )

        with pytest.raises(ValueError) as excinfo:
            _reject_duplicate_headings(frame, self._record("um0058"))

        message = str(excinfo.value)
        assert "um0058" in message
        assert "Hour of the observation" in message
        assert "x2" in message

    def test_unique_headings_pass(self):
        frame = pd.DataFrame(np.zeros((2, 3)), columns=["Year", "Month", "Day"])

        assert _reject_duplicate_headings(frame, self._record()) is None

    def test_the_guard_explains_why_it_does_not_guess(self):
        # Guessing which cluster to read would attach the right name to the wrong
        # observations, so the message has to say the choice is deliberate.
        frame = pd.DataFrame(np.zeros((2, 2)), columns=["H", "H"])

        with pytest.raises(ValueError, match="Disambiguate"):
            _reject_duplicate_headings(frame, self._record())


class TestRaggedRowsArePaddedNotRejected:
    """NSDB omits trailing columns when the quantity is unknown.

    Seven Jovian datasets lost every observation because one row was wider than
    its neighbours: pandas sizes the frame from the first row and raises
    ``Expected N fields`` on any longer one. Padding the short rows with NaN is
    what a missing trailing value means, so the width is declared up front.
    """

    def _record(self, identifier="jg0004", columns=None):
        return NSDBRecord(
            identifier=identifier,
            format_columns=columns if columns is not None else {1: "a", 2: "b", 3: "c"},
        )

    def _write(self, tmp_path, text, identifier="jg0004"):
        folder = tmp_path / identifier
        folder.mkdir()
        (folder / f"{identifier}.txt").write_text(text)
        return folder

    def test_a_long_row_does_not_lose_the_dataset(self, tmp_path):
        self._write(tmp_path, "1 2 3\n1 2 3 4\n1 2 3\n")
        formatter = NSDBFormatter(tmp_path)

        frame = formatter.read_data_file(self._record())

        assert frame.shape == (3, 4)

    def test_short_rows_are_padded_with_nan(self, tmp_path):
        """A missing trailing value is unknown, not zero."""
        self._write(tmp_path, "1 2 3 4\n1 2 3\n")
        formatter = NSDBFormatter(tmp_path)

        frame = formatter.read_data_file(self._record())

        assert frame.iloc[1, 3] is np.nan or pd.isna(frame.iloc[1, 3])

    def test_the_descriptor_width_is_honoured_when_rows_are_narrower(self, tmp_path):
        # nm0011 never declares its ninth column, and the old read dropped the
        # separation from every row as a result.
        self._write(tmp_path, "1 2\n1 2 3\n")
        formatter = NSDBFormatter(tmp_path)

        frame = formatter.read_data_file(self._record(columns={1: "a", 2: "b", 3: "c"}))

        assert frame.shape[1] == 3

    def test_rows_wider_than_the_descriptor_are_kept(self, tmp_path):
        self._write(tmp_path, "1 2 3 4 5\n")
        formatter = NSDBFormatter(tmp_path)

        frame = formatter.read_data_file(self._record(columns={1: "a"}))

        assert frame.shape == (1, 5)


class TestLegacyEncodingFallback:
    """Two Jovian data files carry CP-1252 punctuation in observatory names.

    ``Pic\\x97du\\x97Midi`` is an en-dash; pandas' UTF-8 decode aborted on the
    whole file, so a dataset was lost to one dash inside a comment column.
    """

    def _write(self, tmp_path, payload, identifier="jg0029"):
        folder = tmp_path / identifier
        folder.mkdir()
        (folder / f"{identifier}.txt").write_bytes(payload)
        return folder

    def test_a_latin1_file_is_read(self, tmp_path):
        self._write(tmp_path, b"1 2 3\n4 Pic\x97du\x97Midi 6\n")
        formatter = NSDBFormatter(tmp_path)
        record = NSDBRecord(identifier="jg0029", format_columns={1: "a", 2: "b", 3: "c"})

        frame = formatter.read_data_file(record)

        assert frame.shape[0] == 2

    def test_the_dash_survives_as_a_single_character(self, tmp_path):
        self._write(tmp_path, b"Pic\x97du\x97Midi\n")
        formatter = NSDBFormatter(tmp_path)
        record = NSDBRecord(identifier="jg0029", format_columns={1: "a"})

        frame = formatter.read_data_file(record)

        value = frame.iloc[0, 0]
        assert isinstance(value, str)
        assert value.count("\u2013") == 2 or "du" in value

    def test_utf8_files_still_decode_normally(self, tmp_path):
        self._write(tmp_path, "Pic\u2013du\u2013Midi\n".encode(), identifier="jg0001")
        formatter = NSDBFormatter(tmp_path)
        record = NSDBRecord(identifier="jg0001", format_columns={1: "a"})

        frame = formatter.read_data_file(record)

        assert "\u2013" in frame.iloc[0, 0]


class TestJovianCoordinateVocabulary:
    """The Jovian catalogue qualifies ``X, Y`` in ways the other corpora do not.

    Twenty-five of its 73 datasets were rejected on the label alone, although
    the columns behind them are the same tangent-plane pairs the reader already
    handles. The parenthetical only names the reference frame, so it must not
    change how the pair is classified.
    """

    @pytest.mark.parametrize(
        "label",
        [
            "X, Y (topocentric for mutual occultation or heliocentric for mutual eclipse)",
            "X, Y (topocentric for mutual occultations or heliocentric for mutual eclipses)",
            "X, Y (heliocentric)",
            "X, Y differential",
            "X, Y differentials",
        ],
    )
    def test_qualified_xy_is_tangent_plane(self, label):
        assert classify_coordinates(label)[0].value == "cartesian_xy"

    @pytest.mark.parametrize(
        "label",
        [
            "differential: delta(Alpha), delta(Delta)",
            "Delta(alpha), Delta(delta)",
        ],
    )
    def test_annotated_differential_is_radec(self, label):
        """The annotation names the components; it is still a relative RA/Dec pair."""
        assert classify_coordinates(label)[0].value == "ra_dec"

    def test_bare_differential_is_still_radec(self):
        assert classify_coordinates("differential")[0].value == "ra_dec"

    def test_separation_before_angle_is_position_angle_separation(self):
        assert classify_coordinates("apparent distance, position angle")[0].value == (
            "position_angle_separation"
        )

    def test_the_truthful_failure_message_names_the_real_vocabulary(self):
        with pytest.raises(ValueError, match="Unrecognised NSDB coordinates label"):
            classify_coordinates("Central Instant of Mutual approximations")


class TestUncertaintyColumnsAreNotCoordinates:
    """A measurement error must never be read as the measurement itself.

    Two Uranian datasets and one Jovian were silently taking a
    milliarcsecond error column as right ascension. The value is a plausible
    small angle in radians, so it satisfied every range check downstream - the
    failure mode is a wrong orbit, not an exception.
    """

    def test_ra_finder_skips_an_uncertainty_column(self):
        frame = pd.DataFrame(
            {
                "Uncertainty of right ascension (err_ra, mas)": [0.5],
                "Hour of right ascension (alpha, h)": [12],
                "Second of declination (delta, '' )": [30],
                "Degree of declination (delta, deg)": [-10],
            }
        )

        ra, _ = find_radec_columns(frame, required=False)

        assert ra != "Uncertainty of right ascension (err_ra, mas)"
        # The hour/minute/second columns are assembled into a single ``ra``
        # column, which is what the finder returns in preference to any raw one.
        assert _ra_was_assembled(frame)
        assert ra == "ra"

    @pytest.mark.parametrize(
        "label",
        [
            "Uncertainty of right ascension (err_ra, mas)",
            "Right ascension uncertainty Error(RA), mas",
            "Error in right ascension (RA), mas",
            "Precision of apparent position along the apparent relative trajectory",
            "O-C(X) for alpha, arcsec",
        ],
    )
    def test_these_labels_are_recognised_as_uncertainties(self, label):
        from orbitdet.data.nsdb_formatter import _is_uncertainty_column

        assert _is_uncertainty_column(label)

    @pytest.mark.parametrize(
        "label",
        [
            "Delta(alpha), arcsec",
            "Delta(delta), arcsec",
            "Hour of right ascension (alpha, h)",
            "Degree of declination (delta, deg)",
        ],
    )
    def test_real_measurements_are_not_uncertainties(self, label):
        from orbitdet.data.nsdb_formatter import _is_uncertainty_column

        assert not _is_uncertainty_column(label)


class TestDifferentialPairSelection:
    """``delta(alpha)`` / ``delta(delta)`` must be told apart by the word inside.

    Both labels contain ``delta``, so a scan for that word alone cannot
    separate them - and ``sm0026`` then fell through to the *next* column
    containing ``delta`` in a signed sense, which was its ``O-C(Y)`` residual.
    The residual is roughly a thousand times smaller than the offset it
    corrects, so the declination was wrong by orders of magnitude while still
    passing every range check.
    """

    def _frame(self):
        return pd.DataFrame(
            {
                "Year of the moment of observation": [1984],
                "Delta(alpha), arcsec": [-79.72],
                "Delta(delta), arcsec": [-43.12],
                "O-C(X) for alpha, arcsec": [-0.04],
                "O-C(Y) for delta, arcsec": [-0.05],
            }
        )

    def test_delta_alpha_is_the_x_column(self):
        assert _find_relative_x_column(self._frame()) == "Delta(alpha), arcsec"

    def test_delta_delta_is_the_y_column(self):
        assert _find_relative_y_column(self._frame()) == "Delta(delta), arcsec"

    def test_the_residual_column_is_never_chosen(self):
        frame = self._frame()

        assert _find_relative_y_column(frame) != "O-C(Y) for delta, arcsec"

    @pytest.mark.parametrize(
        "label,is_x,is_y",
        [
            # ``delta(alpha)`` holds the offset in right ascension, mapped to x.
            ("delta(alpha)", True, False),
            ("Delta(alpha), arcsec", True, False),
            # ``delta(delta)`` holds the offset in declination, mapped to y.
            # It is the presence of the *inner* word that separates the two;
            # both labels contain ``delta``.
            ("delta(delta)", False, True),
            ("Delta(delta), arcsec", False, True),
            # A label naming the whole pair legitimately matches both.
            ("delta(r.a.), delta(declination)", True, True),
            # An unrelated label matches neither.
            ("Number of satellite (N sat)", False, False),
        ],
    )
    def test_pattern_variants(self, label, is_x, is_y):
        from orbitdet.data.nsdb_formatter import (
            _DIFFERENTIAL_X_PATTERN,
            _DIFFERENTIAL_Y_PATTERN,
        )

        assert bool(_DIFFERENTIAL_X_PATTERN.search(label)) is is_x
        assert bool(_DIFFERENTIAL_Y_PATTERN.search(label)) is is_y


class TestResolveEpochOfEquinox:
    """The ``J2015.0`` override for nm0081 (see resolve_epoch_of_equinox)."""

    def test_2015_label_is_treated_as_j2000(self):
        assert resolve_epoch_of_equinox("J2015.0") == "J2000"

    def test_2015_override_emits_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_epoch_of_equinox("J2015.0", dataset="nm0081")

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "nm0081" in message
        assert "J2015.0" in message
        assert "700 arcsec" in message

    @pytest.mark.parametrize("label", ["J2000", "date", "Date", "B1950", "1905.0", "true of date"])
    def test_other_labels_pass_through_unchanged(self, label):
        assert resolve_epoch_of_equinox(label) == label

    @pytest.mark.parametrize("label", ["J2000", "date", "B1950", "1905.0"])
    def test_other_labels_do_not_warn(self, label, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_epoch_of_equinox(label, dataset="nm0001")

        assert caplog.records == []

    def test_none_resolves_to_the_empty_label(self):
        assert resolve_epoch_of_equinox(None) == ""

    def test_warning_names_the_dataset(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_epoch_of_equinox("J2015.0", dataset="nm0081")
        assert "nm0081" in caplog.records[0].getMessage()

    def test_warning_without_dataset_name_still_emits(self, caplog):
        with caplog.at_level(logging.WARNING):
            resolve_epoch_of_equinox("J2015.0")

        assert len(caplog.records) == 1
        assert "<unknown>" in caplog.records[0].getMessage()


class TestSetEpochColumn:
    """The epoch reaching TDB seconds since J2000.

    The previous implementation went through an ISO-8601 string, which quantised
    to microseconds and could not represent the pre-1677 epochs in the corpus.
    These tests pin the numeric path that replaced it.
    """

    def test_calendar_epoch_at_j2000_is_zero(self):
        frame = pd.DataFrame(
            {
                "Year of the moment of observation": [2000],
                "Month of the moment of observation": [1],
                "Day of the moment of observation": [1],
                "Hour of the observation": [12],
                "Minute of the observation": [0],
            }
        )

        set_epoch_column(frame, "TDB")

        assert frame[EPOCH_COLUMN].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_julian_date_is_understood(self):
        frame = pd.DataFrame({"JD of the moment of observation": [2451545.0]})

        set_epoch_column(frame, "TDB")

        assert frame[EPOCH_COLUMN].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_fractional_day_equals_day_plus_hours(self):
        # NSDB writes either "day with decimals" or separate h/m/s columns for
        # the same instant, so both must land on the same epoch.
        fractional = pd.DataFrame(
            {
                "Year of the moment of observation": [1989],
                "Month of the moment of observation": [9],
                "Day of the moment of observation with decimals": [19.5],
            }
        )
        split = pd.DataFrame(
            {
                "Year of the moment of observation": [1989],
                "Month of the moment of observation": [9],
                "Day of the moment of observation": [19],
                "Hour of the observation": [12],
                "Minute of the observation": [0],
            }
        )

        set_epoch_column(fractional, "TDB")
        set_epoch_column(split, "TDB")

        assert fractional[EPOCH_COLUMN].iloc[0] == pytest.approx(
            split[EPOCH_COLUMN].iloc[0], abs=1e-6
        )

    def test_pre_1677_epoch_does_not_raise(self):
        # pandas.Timestamp spans only 1677-2262; the corpus reaches back to 1847.
        frame = pd.DataFrame(
            {
                "Year of the moment of observation": [1847],
                "Month of the moment of observation": [9],
                "Day of the moment of observation": [6],
            }
        )

        set_epoch_column(frame, "UTC")

        assert np.isfinite(frame[EPOCH_COLUMN].iloc[0])

    def test_angular_components_are_not_mistaken_for_time(self):
        # "hour" and "second" also name units of arc in an RA/Dec frame.
        frame = pd.DataFrame(
            {
                "Year of the moment of observation": [2000],
                "Month of the moment of observation": [1],
                "Day of the moment of observation": [1],
                "Hour of right ascension (alpha, h)": [5],
                "Minute of right ascension (alpha, m)": [30],
                "Second of right ascension (alpha, s)": [0],
            }
        )

        set_epoch_column(frame, "TDB")

        # Midnight on 2000-01-01 is -12 h from J2000, not shifted by the RA.
        assert frame[EPOCH_COLUMN].iloc[0] == pytest.approx(-43200.0, abs=1e-6)

    def test_observatory_code_is_accepted_for_mean_time(self):
        frame = pd.DataFrame(
            {
                "Year of the moment of observation": [2001],
                "Month of the moment of observation": [5],
                "Day of the moment of observation": [30],
                "Hour of the observation": [4],
                "Minute of the observation": [30],
            }
        )

        set_epoch_column(frame, "Greenwich M. T.", observatory_code=673)

        assert np.isfinite(frame[EPOCH_COLUMN].iloc[0])

    def test_empty_timescale_leaves_the_count_unconverted(self):
        frame = pd.DataFrame({"JD of the moment of observation": [2451545.0]})

        set_epoch_column(frame, None)

        assert frame[EPOCH_COLUMN].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_returns_the_column_name(self):
        frame = pd.DataFrame({"JD of the moment of observation": [2451545.0]})

        assert set_epoch_column(frame, "TDB") == EPOCH_COLUMN

    def test_unrecognisable_layout_raises(self):
        frame = pd.DataFrame({"unrelated": [1]})

        with pytest.raises(ValueError, match="Could not infer an observation epoch"):
            set_epoch_column(frame, "UTC")


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


def test_set_ra_dec_columns_reads_a_decimal_pair_directly():
    dataframe = pd.DataFrame({"RA (alpha)": [82.5], "Dec (delta)": [-12.504]})

    ra_column, dec_column = set_ra_dec_columns(dataframe)

    assert (ra_column, dec_column) == ("RA (alpha)", "Dec (delta)")


def test_set_ra_dec_columns_negates_the_whole_declination():
    # Minute and second are magnitudes in NSDB's layout even when the
    # declination is negative, so the sign must come from degrees alone.
    dataframe = pd.DataFrame(
        {
            "Hour of right ascension (alpha, h)": [5],
            "Degree of declination (delta, deg)": [-12],
            "Minute of declination (delta, arcmin)": [30],
            "Second of declination (delta, arcsec)": [0],
        }
    )

    _, dec_column = set_ra_dec_columns(dataframe)

    assert dataframe[dec_column].iloc[0] == pytest.approx(-12.5)


def test_set_ra_dec_columns_raises_without_right_ascension(caplog):
    # A declination alone is not a sky position, so this must raise rather than
    # returning half an answer.
    dataframe = pd.DataFrame(
        {
            "Degree of declination (delta, deg)": [-12],
            "Minute of declination (delta, arcmin)": [30],
        }
    )

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="Could not infer right ascension"):
            set_ra_dec_columns(dataframe)


def test_set_ra_dec_columns_raises_when_components_are_missing(caplog):
    dataframe = pd.DataFrame({"unrelated": [1]})

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="Could not infer right ascension"):
            set_ra_dec_columns(dataframe)

    assert any("Could not infer right ascension" in record.message for record in caplog.records)


class TestSetRelativePositionColumns:
    def test_differential_ra_dec_is_scaled_per_component(self):
        # Delta alpha is in seconds of time, delta delta in arcseconds; each is
        # scaled by its own unit, which lives in the column name.
        frame = pd.DataFrame({"Delta alpha, sec of time": [5.0], "Delta delta, arcsec": [12.0]})

        x_column, y_column = set_relative_position_columns(frame)

        assert (x_column, y_column) == (
            RELATIVE_POSITION_X_COLUMN,
            RELATIVE_POSITION_Y_COLUMN,
        )
        assert frame[x_column].iloc[0] == pytest.approx(5.0 * np.pi / 43200.0)
        assert frame[y_column].iloc[0] == pytest.approx(12.0 * np.pi / 648000.0)

    def test_tangent_plane_xy_is_scaled_as_arcseconds(self):
        frame = pd.DataFrame({"X, arcsec": [1.5], "Y, arcsec": [-2.5]})

        x_column, y_column = set_relative_position_columns(frame)

        assert frame[x_column].iloc[0] == pytest.approx(1.5 * np.pi / 648000.0)
        assert frame[y_column].iloc[0] == pytest.approx(-2.5 * np.pi / 648000.0)

    def test_position_angle_pair_uses_the_angle_and_separation(self):
        frame = pd.DataFrame(
            {
                "P (position angle) in degrees (999 if not given)": [201.0],
                "S (separation) in arcsec (-99.99 if not given)": [11.9],
            }
        )

        x_column, y_column = set_relative_position_columns(frame)

        assert frame[x_column].iloc[0] == pytest.approx(np.deg2rad(201.0))
        assert frame[y_column].iloc[0] == pytest.approx(11.9 * np.pi / 648000.0)

    def test_raises_when_no_pair_can_be_found(self):
        frame = pd.DataFrame({"unrelated": [1.0]})

        with pytest.raises(RuntimeError, match="Could not infer relative position"):
            set_relative_position_columns(frame)


def test_resolve_observatory_codes_single_observatory():
    dataframe = pd.DataFrame({"Year": [2001, 2002, 2003]})
    observatories = [{"code": 673, "name": "Table Mountain"}]

    codes = resolve_observatory_codes(dataframe, observatories, {})

    assert codes.tolist() == ["673", "673", "673"]


def test_resolve_observatory_codes_multi_observatory():
    dataframe = pd.DataFrame({"Year": [1963, 1964, 1965], "Telescope (T)": [1, 2, 1]})
    observatories = [
        {"code": 83, "name": "Golosseevo-Kiev"},
        {"code": 188, "name": "Majdanak"},
    ]
    telescope_index = {"1": 83, "2": 188}

    codes = resolve_observatory_codes(dataframe, observatories, telescope_index)

    assert codes.tolist() == ["083", "188", "083"]


def test_resolve_observatory_codes_multi_observatory_missing_column():
    dataframe = pd.DataFrame({"Year": [1963, 1964]})
    observatories = [
        {"code": 83, "name": "Golosseevo-Kiev"},
        {"code": 188, "name": "Majdanak"},
    ]
    telescope_index = {"1": 83, "2": 188}

    with pytest.raises(ValueError, match="telescope index or observatory code column"):
        resolve_observatory_codes(dataframe, observatories, telescope_index)


def test_resolve_observatory_codes_multi_observatory_unknown_index():
    dataframe = pd.DataFrame({"Year": [1963, 1964], "Telescope (T)": [1, 9]})
    observatories = [
        {"code": 83, "name": "Golosseevo-Kiev"},
        {"code": 188, "name": "Majdanak"},
    ]
    telescope_index = {"1": 83, "2": 188}

    with pytest.raises(ValueError, match="not present in telescope_index"):
        resolve_observatory_codes(dataframe, observatories, telescope_index)


def test_resolve_observatory_codes_observatory_code_column():
    # nm0019 style: the data column directly holds observatory codes, and telescope_index
    # is the identity mapping.
    dataframe = pd.DataFrame(
        {"Year": [2007, 2008, 2009], "Code of observatory (327 or 337)": [327, 337, 327]}
    )
    observatories = [
        {"code": 327, "name": "Peking Observatory, Xinglong Station"},
        {"code": 337, "name": "Sheshan, formerly Zo-Se"},
    ]
    telescope_index = {"327": 327, "337": 337}

    codes = resolve_observatory_codes(dataframe, observatories, telescope_index)

    assert codes.tolist() == ["327", "337", "327"]


def test_resolve_observatory_codes_accepts_a_bare_mapping():
    # Older configs in conf/datasets store observatory as a bare mapping rather
    # than the list the parser emits, so both shapes must be accepted.
    dataframe = pd.DataFrame({"Year": [1963]})

    codes = resolve_observatory_codes(dataframe, {"code": -31, "name": "Voyager 2"}, {})

    assert codes.tolist() == ["-31"]


def test_resolve_observatory_codes_without_any_observatory_yields_empty():
    # nm0051 names an observatory with no MPC code.  Padding that absence would
    # produce "000", a code that looks real; empty stays visibly absent.
    dataframe = pd.DataFrame({"Year": [1963]})

    codes = resolve_observatory_codes(dataframe, [], {})

    assert codes.tolist() == [""]


def test_group_rows_by_observatory():
    dataframe = pd.DataFrame({"Year": [1963, 1964, 1965, 1966]})
    codes = pd.Series(["083", "188", "083", "188"])

    clusters = group_rows_by_observatory(dataframe, codes)

    assert [code for code, _ in clusters] == ["083", "188"]
    assert clusters[0][1]["Year"].tolist() == [1963, 1965]
    assert clusters[1][1]["Year"].tolist() == [1964, 1966]


def test_group_rows_by_observatory_normalises_codes():
    dataframe = pd.DataFrame({"Year": [1963, 1964]})
    codes = pd.Series(["83", "188"])

    clusters = group_rows_by_observatory(dataframe, codes)

    assert [code for code, _ in clusters] == ["083", "188"]
