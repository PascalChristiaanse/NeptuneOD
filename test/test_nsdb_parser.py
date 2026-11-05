"""Tests for :mod:`orbitdet.data.nsdb_parser`.

Two layers are tested. The first uses small hand-written snippets to pin the
grammar down exhaustively. The second runs the whole parser over every
descriptor in ``data/raw``, because the real corpus is the only place the
heading and field-shape variations actually show up — several of the bugs this
module fixes were invisible in synthetic input.
"""

import re
from pathlib import Path

import pytest

from orbitdet.data.nsdb_parser import (
    NSDBParser,
    NSDBRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
REAL_DESCRIPTORS = sorted(DATA_RAW.glob("nm*/nm*.html"))


class FakeResponse:
    """Minimal stand-in for a requests response."""

    def __init__(self, text: str, status_code: int = 200):
        self.content = text.encode()
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def descriptor(body: str, title: str = "nm0001") -> str:
    """Wrap a ``<pre>`` body in a minimal descriptor document."""
    return f"<HTML><HEAD><TITLE>{title}</TITLE></HEAD><PRE>\n{body}\n</PRE></BODY></HTML>"


SIMPLE_BODY = """Contents.
      planet: 8 - Neptune
  satellites: N1-Triton    :3
total number: 3
        type: absolute
       dates: 2001-2001
 observatory: 673 -Table Mountain Observatory, Wrightwood

Reference.
   (2001) Communicated to NSDC by W. M. Owen Jr.,

Informations.
         relative to: absolute
     reference frame: astrometric
     centre of frame: topocentre
    epoch of equinox: J2000
          time scale: UTC
           reduction: no information
         coordinates: absolute
    diff. refraction: no information
            receptor: CCD
           telescope: Reflector, D = 0.61 m, f/16
           observers: Owen Jr.W.M.
 data included in standard data file: no

Comments.
                      no information

Format.
  1. Number of satellite (N sat)
  2. Year   of the moment of observation
  3. Day    of the moment of observation with decimals

----------------------------------------------------
 N  Year      UT         alpha         delta
sat.  month day
       with decimals     h  m  s      deg   '  ''
----------------------------------------------------
"""


@pytest.fixture
def parser() -> NSDBParser:
    return NSDBParser()


# ---------------------------------------------------------------------------
# Heading recognition
# ---------------------------------------------------------------------------


class TestNormaliseHeading:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("Contents.", "contents"),
            ("contents.", "contents"),
            ("CONTENTS.", "contents"),
            ("Informations.", "informations"),
            ("Information.", "informations"),
            ("References.", "reference"),
            ("Reference.", "reference"),
            ("Reference", "reference"),
            ("References", "reference"),
            ("Comments.", "comments"),
            ("Format.", "format"),
            ("   Format.   ", "format"),
        ],
    )
    def test_recognised_headings(self, parser, line, expected):
        assert parser.normalise_heading(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "no. 404, Berkeley",
            "Berkeley: The University Press",
            "This is just prose.",
            "Moskva",
            "observatory: 083 - Golosseevo-Kiev",
            "1874MNRAS__35___49D",
            "Table",
        ],
    )
    def test_non_headings(self, parser, line):
        """Prose and data lines must not be mistaken for headings.

        ``no. 404, Berkeley`` is a real bibliography line that contains a period
        and would be a heading if the token count were not checked.
        """
        assert parser.normalise_heading(line) is None


class TestSplitSections:
    def test_splits_all_sections(self, parser):
        sections = parser.split_sections(SIMPLE_BODY)

        assert set(sections) == {"contents", "reference", "informations", "comments", "format"}
        assert "planet: 8 - Neptune" in sections["contents"]
        assert "reference frame: astrometric" in sections["informations"]

    def test_singular_and_plural_reference_agree(self, parser):
        singular = "Contents.\nx: 1\nReference.\nbib\nInformations.\ny: 2\nFormat.\n1. a\n"
        plural = "Contents.\nx: 1\nReferences.\nbib\nInformations.\ny: 2\nFormat.\n1. a\n"

        a = parser.split_sections(singular)
        b = parser.split_sections(plural)

        assert a["reference"] == b["reference"] == "bib"

    def test_bare_reference_heading(self, parser):
        """``Reference`` with no period occurs in 5 real datasets."""
        sections = parser.split_sections("Contents.\nx: 1\nReference\nbib\nFormat.\n1. a\n")

        assert sections["reference"] == "bib"

    def test_unrecognised_heading_does_not_end_the_section(self, parser):
        """A stray heading-like line must not truncate the current section.

        ``nm0036`` publishes its metadata with no ``Informations.`` heading. If
        an unknown heading ended the section, the rest of that block would be
        lost rather than recoverable.
        """
        sections = parser.split_sections(
            "Contents.\nplanet: 8 - Neptune\nUnknown.\nobservatory: 786 - USNO\nFormat.\n1. a\n"
        )

        assert "observatory: 786 - USNO" in sections["contents"]

    def test_handles_crlf(self, parser):
        """8 real descriptors use CRLF, which would otherwise leave ``\\r``."""
        sections = parser.split_sections("Contents.\r\nplanet: 8 - Neptune\r\nFormat.\r\n1. a\r\n")

        assert sections["contents"] == "planet: 8 - Neptune"

    def test_missing_sections_are_absent(self, parser):
        sections = parser.split_sections("Contents.\nx: 1\nFormat.\n1. a\n")

        assert "informations" not in sections
        assert "comments" not in sections


# ---------------------------------------------------------------------------
# The key/value grammar
# ---------------------------------------------------------------------------


class TestParseKeyValueBlock:
    def test_single_line(self, parser):
        assert parser.parse_key_value_block("observatory: 083\n") == {"observatory": "083"}

    def test_multiple_lines(self, parser):
        result = parser.parse_key_value_block("observatory: 083\ntype: absolute\n")

        assert result == {"observatory": "083", "type": "absolute"}

    def test_value_containing_a_colon(self, parser):
        """Everything after the first colon belongs to the value."""
        result = parser.parse_key_value_block("source: https://nsdb.imcce.fr/obspos\n")

        assert result == {"source": "https://nsdb.imcce.fr/obspos"}

    def test_multi_line_value_becomes_a_list(self, parser):
        result = parser.parse_key_value_block(
            "observatory: 083 - Golosseevo-Kiev\n"
            "             188 - Majdanak\n"
            "             095 - Simeiz\n"
        )

        assert result == {
            "observatory": ["083 - Golosseevo-Kiev", "188 - Majdanak", "095 - Simeiz"]
        }

    def test_dash_prefixed_continuation_is_not_a_key(self, parser):
        """The key regex excludes ``-``, which is what makes this safe.

        ``188 - Z6: The 60-cm Zeiss Reflector`` contains a colon, so a naive
        ``line.split(':')`` would promote it to a key called ``188 - Z6``.
        """
        result = parser.parse_key_value_block(
            "telescope: 083 - DL: The Tepfer Double Long-Focus Astrograph\n"
            "           188 - Z6: The 60-cm Zeiss Reflector\n"
        )

        assert list(result) == ["telescope"]
        assert len(result["telescope"]) == 2

    def test_blank_line_ends_the_value(self, parser):
        """A blank line closes the field, which stops prose being absorbed.

        Without this, ``nm0020``'s bibliography (which follows its observatory
        line with no heading) was swallowed into the observatory list.
        """
        result = parser.parse_key_value_block(
            "observatory: 999 - Bordeaux-Floirac\n"
            "\n"
            "  Soulie G. (1975)\n"
            "  Astronomy and Astrophysics\n"
            "receptor: photographic\n"
        )

        assert result["observatory"] == "999 - Bordeaux-Floirac"
        assert result["receptor"] == "photographic"

    def test_leading_blank_lines_ignored(self, parser):
        assert parser.parse_key_value_block("\n\nobservatory: 083\n") == {"observatory": "083"}

    def test_duplicate_key_raises(self, parser):
        with pytest.raises(ValueError, match="Duplicate key"):
            parser.parse_key_value_block("observatory: 083\nobservatory: 188\n")

    def test_empty_input(self, parser):
        assert parser.parse_key_value_block("") == {}

    def test_key_without_value(self, parser):
        assert parser.parse_key_value_block("observatory:\n") == {"observatory": ""}


# ---------------------------------------------------------------------------
# Format section
# ---------------------------------------------------------------------------


class TestParseFormatBlock:
    def test_numbered_entries(self, parser):
        result = parser.parse_format_block("  1. Year of observation\n  2. Month\n")

        assert result == {1: "Year of observation", 2: "Month"}

    def test_stops_at_the_separator_rule(self, parser):
        """Everything after the rule is a worked example, not a declaration.

        ``nm0076`` publishes a sample row starting with ``1903``; parsing it as
        a declaration produced a bogus column 272 that propagated into the
        dataframe column names.
        """
        result = parser.parse_format_block(
            "  1. Year\n"
            "  2. Month\n"
            "\n"
            "-----------------------------------------\n"
            "Year        90deg.time  P\n"
            "   month               deg.     S\n"
            "       day  h  m  s           arcsec\n"
            "-----------------------------------------\n"
            "1903  2 16  8 45 43  272.99   -99.99\n"
        )

        assert result == {1: "Year", 2: "Month"}

    def test_stops_at_short_rule(self, parser):
        result = parser.parse_format_block("1. a\n----\n1903  2 16 272.99\n")

        assert result == {1: "a"}

    def test_two_digit_indices(self, parser):
        result = parser.parse_format_block("  9. Ninth\n 10. Tenth\n 20. Twentieth\n")

        assert result == {9: "Ninth", 10: "Tenth", 20: "Twentieth"}

    def test_blank_lines_skipped(self, parser):
        assert parser.parse_format_block("\n1. a\n\n2. b\n") == {1: "a", 2: "b"}

    def test_empty(self, parser):
        assert parser.parse_format_block("") == {}


# ---------------------------------------------------------------------------
# Domain field parsing
# ---------------------------------------------------------------------------


class TestParsePlanet:
    def test_number_and_name(self, parser):
        assert parser.parse_planet("8 - Neptune") == {"number": 8, "name": "Neptune"}

    def test_extra_spacing(self, parser):
        assert parser.parse_planet("8 - Neptune  ") == {"number": 8, "name": "Neptune"}

    def test_name_only(self, parser):
        result = parser.parse_planet("Neptune")

        assert result["name"] == "Neptune"
        assert result["number"] is None

    def test_missing(self, parser):
        assert parser.parse_planet(None) is None


class TestParseSatellites:
    def test_designation_and_count(self, parser):
        result = parser.parse_satellites("N1-Triton    :3")

        assert result == {"Triton": {"designation": "N1-Triton", "number_of_observations": 3}}

    def test_spaces_around_the_dash(self, parser):
        result = parser.parse_satellites("N1 - Triton : 10")

        assert result["Triton"]["number_of_observations"] == 10

    def test_without_count(self, parser):
        result = parser.parse_satellites("N1-Triton")

        assert result["Triton"]["number_of_observations"] is None

    def test_several_satellites_are_merged(self, parser):
        result = parser.parse_satellites(["N1-Triton : 5", "N2-Nereid : 7"])

        assert set(result) == {"Triton", "Nereid"}

    def test_missing(self, parser):
        assert parser.parse_satellites(None) == {}


class TestParseDates:
    def test_range(self, parser):
        assert parser.parse_dates("1993-1998") == {"start_year": 1993, "end_year": 1998}

    def test_single_year(self, parser):
        assert parser.parse_dates("1874") == {"start_year": 1874, "end_year": 1874}

    def test_same_start_and_end(self, parser):
        assert parser.parse_dates("2001-2001") == {"start_year": 2001, "end_year": 2001}

    def test_spaces_around_the_dash(self, parser):
        assert parser.parse_dates("1993 - 1998") == {"start_year": 1993, "end_year": 1998}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1898-1899, 1901-1902", {"start_year": 1898, "end_year": 1902}),
            ("1912-1913, 1918-1919", {"start_year": 1912, "end_year": 1919}),
            ("1849-1852, 1856", {"start_year": 1849, "end_year": 1856}),
            ("1899-1900, 1902, 1907", {"start_year": 1899, "end_year": 1907}),
        ],
    )
    def test_discontinuous_ranges(self, parser, text, expected):
        """Several ranges must not fuse into a nonsense year.

        Stripping non-digits from ``1898-1899, 1901-1902`` produced the single
        number ``1898199919011902``, which then became ``end_year: 18991901``.
        """
        assert parser.parse_dates(text) == expected

    def test_missing(self, parser):
        assert parser.parse_dates(None) is None
        assert parser.parse_dates("") is None


class TestParseObservatoryEntries:
    def test_standard_form(self, parser):
        observatories, unmatched = parser.parse_observatory_entries("083 - Golosseevo-Kiev")

        assert observatories == [{"code": 83, "name": "Golosseevo-Kiev"}]
        assert unmatched == []

    def test_missing_space_after_dash(self, parser):
        """``nm0001`` has ``673 -Table Mountain Observatory``."""
        observatories, _ = parser.parse_observatory_entries("673 -Table Mountain Observatory")

        assert observatories == [{"code": 673, "name": "Table Mountain Observatory"}]

    def test_no_dash(self, parser):
        """``nm0018`` has ``999 Bordeaux-Floirac``."""
        observatories, _ = parser.parse_observatory_entries("999 Bordeaux-Floirac")

        assert observatories == [{"code": 999, "name": "Bordeaux-Floirac"}]

    def test_letter_prefixed_code(self, parser):
        """``nm0028`` has ``H80 - Halsted observatory``; MPC issues such codes."""
        observatories, _ = parser.parse_observatory_entries("H80 - Halsted observatory")

        assert observatories == [{"code": "H80", "name": "Halsted observatory"}]

    def test_name_only(self, parser):
        """``nm0051`` names its observatory with no code at all."""
        observatories, _ = parser.parse_observatory_entries("Cambridge")

        assert observatories == [{"code": None, "name": "Cambridge"}]

    def test_several_entries(self, parser):
        observatories, _ = parser.parse_observatory_entries(
            ["083 - Golosseevo-Kiev", "188 - Majdanak"]
        )

        assert [o["code"] for o in observatories] == [83, 188]

    def test_bibliography_is_not_an_observatory(self, parser):
        """A citation line must not become a bogus entry.

        A four-digit prefix like ``1874`` is not a valid MPC code, so bounding
        the code to three digits rejects it. Before that bound, 49 datasets
        produced 395 bogus entries this way.
        """
        observatories, unmatched = parser.parse_observatory_entries(
            ["787 - U.S. Naval Obs., Washington", "1874MNRAS__35___49D", "University Press"]
        )

        assert [o["code"] for o in observatories] == [787, None]
        assert unmatched == ["1874MNRAS__35___49D"]


class TestParseTelescopes:
    def test_free_text_stays_a_string(self, parser):
        result = parser.parse_telescopes("Reflector, D = 0.61 m, f/16")

        assert result == "Reflector, D = 0.61 m, f/16"

    @pytest.mark.parametrize("text", ["26-inch", "40-inch refractor", "1-m reflectors", "1.6 m"])
    def test_dimensions_are_not_telescope_entries(self, parser, text):
        """A hyphen inside a dimension must not be read as a separator.

        ``26-inch`` splits on its dash into index 26 with an empty name if the
        pattern does not require whitespace on both sides.
        """
        assert parser.parse_telescopes(text) == text

    def test_observatory_code_prefix(self, parser):
        result = parser.parse_telescopes(
            ["083 - DL: The Tepfer Double Long-Focus Astrograph", "188 - Z6: The Zeiss Reflector"]
        )

        assert isinstance(result, list)
        assert result[0] == {
            "code": 83,
            "name": "DL",
            "description": "The Tepfer Double Long-Focus Astrograph",
            "index": 83,
        }

    def test_embedded_observatory_code(self, parser):
        """``nm0019`` embeds the code in the description."""
        result = parser.parse_telescopes(
            ["1 - Reflector, D=156 cm (at the Sheshan Station, code 337),"]
        )

        assert isinstance(result, list)
        assert result[0]["index"] == 1
        assert result[0]["code"] == 337

    def test_missing(self, parser):
        assert parser.parse_telescopes(None) is None


class TestParseTelescopeIndex:
    def test_index_to_name_mapping(self, parser):
        telescopes = [
            {"code": 83, "name": "DL"},
            {"code": 188, "name": "Z6"},
        ]
        result = parser.parse_telescope_index({19: "Telescope (T): 1 - DL, 2 - Z6"}, telescopes)

        assert result == {"1": 83, "2": 188}

    def test_direct_observatory_code_column(self, parser):
        """The column already holds codes, so the mapping is the identity."""
        telescopes = [{"code": 327, "name": "X"}, {"code": 337, "name": "Y"}]
        result = parser.parse_telescope_index({10: "Code of observatory (327 or 337)."}, telescopes)

        assert result == {"327": 327, "337": 337}

    def test_no_suitable_column(self, parser):
        assert parser.parse_telescope_index({1: "Year"}, [{"code": 83, "name": "DL"}]) == {}

    def test_free_text_telescope_yields_nothing(self, parser):
        assert parser.parse_telescope_index({1: "Year"}, "Reflector, D=0.61 m") == {}


class TestTelescopeCodeResolution:
    """A Format column may name instruments in a shorthand the field does not use.

    ``sm0074`` writes ``1 - DLFA`` where the telescope field says ``083 - DLFA
    (Double Long Focus Astrograph), D/f=0.4/5.5m``, so an exact-name lookup found
    nothing and the dataset was rejected outright.
    """

    def test_unique_prefix_resolves(self, parser):
        telescopes = [
            {"code": 83, "name": "DLFA (Double Long Focus Astrograph), D/f=0.4/5.5m"},
            {"code": 186, "name": "DAZ (Wide Angle Astrograph), D/f=0.4/3.0m"},
        ]
        result = parser.parse_telescope_index(
            {16: "Telescope (tel: 1 - DLFA, 2 - DAZ)"}, telescopes
        )

        assert result == {"1": 83, "2": 186}

    def test_exact_match_still_wins(self, parser):
        """A dataset that resolves today must keep its mapping."""
        telescopes = [
            {"code": 83, "name": "DL"},
            {"code": 84, "name": "DL extra"},
        ]
        result = parser.parse_telescope_index({1: "Telescope: 1 - DL"}, telescopes)

        assert result == {"1": 83}

    def test_ambiguous_prefix_yields_none(self, parser):
        """Pointing at the wrong observatory is worse than reporting a gap."""
        telescopes = [
            {"code": 83, "name": "DL Alpha"},
            {"code": 84, "name": "DL Beta"},
        ]
        result = parser.parse_telescope_index({1: "Telescope: 1 - DL"}, telescopes)

        assert result == {"1": None}

    def test_prefix_must_break_on_a_word_boundary(self, parser):
        telescopes = [{"code": 83, "name": "DLFA something"}]

        result = parser.parse_telescope_index({1: "Telescope: 1 - DL"}, telescopes)

        assert result == {"1": None}

    def test_trailing_parenthesis_is_stripped(self, parser):
        """The last entry is followed by the closing bracket of the label."""
        telescopes = [
            {"code": 186, "name": "DAZ (Wide Angle Astrograph), D/f=0.4/3.0m"},
            {"code": 188, "name": "Z600 (Zeiss-600), D=0.6, Cassegrain, F=7.5m"},
        ]
        result = parser.parse_telescope_index(
            {16: "Telescope (tel: 3 - DAZ, 4 - Z600)"}, telescopes
        )

        assert result == {"3": 186, "4": 188}

    def test_unrelated_acronym_stays_unresolved(self, parser):
        """``sm0084`` writes ``P26``/``PNA``/``ZDA`` for names sharing no prefix."""
        telescopes = [
            {"code": 1, "name": "Pulkovo 26-inch refractor (D=0.65m, F=10.413m),"},
            {"code": 2, "name": "Pulkovo normal astrograph (D=0.33m, F=3.463m),"},
            {"code": 3, "name": "Abastumani Zeiss Double Astrograph (D=0.4m, F=3.024m)"},
        ]
        result = parser.parse_telescope_index(
            {5: "Telescope (1 - P26, 2 - PNA, 3 - ZDA)"}, telescopes
        )

        assert result == {"1": None, "2": None, "3": None}


class TestTelescopeContinuationLines:
    """A continuation line repeats no code, so it inherits the preceding one.

    ``sm0079`` lists ``083 - DL: ...`` and then ``- DW: ...``, both instruments
    at code 083.  Leaving the second code ``None`` rejected the dataset.
    """

    def test_continuation_inherits_preceding_code(self, parser):
        result = parser.parse_telescopes(
            [
                "083 - DL: The Tepfer Double Long-Focus Astrograph (D=40cm, F=550cm)",
                "- DW: The Zeiss Double Wide-Angle Astrograph (D=40cm, F=220cm)",
                "186 - DZ: The Zeiss Double Wide-Angle Astrograph (D=40cm, F=300cm)",
                "188 - Z6: The 60-cm Zeiss Reflector (D=60cm, Cassegrain F=750cm)",
            ]
        )

        assert isinstance(result, list)
        assert [t["code"] for t in result] == [83, 83, 186, 188]

    def test_continuation_before_any_code_stays_unresolved(self, parser):
        """With no preceding code to inherit there is nothing to resolve to."""
        result = parser.parse_telescopes(["- DW: The Zeiss Double Wide-Angle Astrograph"])

        assert result == "- DW: The Zeiss Double Wide-Angle Astrograph"


class TestFuseObservables:
    def test_fuses_three_parts(self, parser):
        assert (
            parser.fuse_observables("absolute", "absolute", "CCD") == "absolute_absolute_CCD_nsdb"
        )

    def test_spaces_become_underscores(self, parser):
        result = parser.fuse_observables("relative", "position angle and separation", "micrometric")

        assert result == "relative_position_angle_and_separation_micrometric_nsdb"

    def test_source_typos_are_preserved(self, parser):
        """The registry keys contain NSDB's own mistakes, so they must survive."""
        assert parser.fuse_observables("relative", "a", "photograpihc") == (
            "relative_a_photograpihc_nsdb"
        )

    def test_missing_receptor(self, parser):
        assert parser.fuse_observables("absolute", "absolute", None) is None

    def test_missing_type(self, parser):
        assert parser.fuse_observables(None, "absolute", "CCD") is None


class TestParseInformations:
    def test_field_names_are_mapped(self, parser):
        block = {
            "reference frame": "astrometric",
            "epoch of equinox": "J2000",
            "time scale": "UTC",
            "diff. refraction": "no information",
            "data included in standard data file": "no",
        }
        result = parser.parse_informations(block)

        assert result["reference_frame"] == "astrometric"
        assert result["epoch_of_equinox"] == "J2000"
        assert result["time_scale"] == "UTC"
        assert result["diff_refraction"] == "no information"
        assert result["data_included_in_standard_data_file"] == "no"

    def test_both_centre_spellings_are_emitted(self, parser):
        result = parser.parse_informations({"centre of frame": "topocentre"})

        assert result["centre_of_frame"] == "topocentre"
        assert result["center_of_frame"] == "topocentre"

    def test_relative_to(self, parser):
        result = parser.parse_informations({"relative to": "N1-Triton  :174"})

        assert result["relative_to"] == {
            "designation": "N1-Triton",
            "number_of_observations": 174,
        }

    def test_relative_to_absolute(self, parser):
        result = parser.parse_informations({"relative to": "absolute"})

        assert result["relative_to"]["designation"] == "absolute"
        assert result["relative_to"]["number_of_observations"] is None

    def test_multi_line_value_is_joined(self, parser):
        result = parser.parse_informations({"observers": ["Hall A.", "Barnard E. E."]})

        assert result["observers"] == "Hall A. Barnard E. E."

    def test_unrecognised_keys_are_dropped(self, parser):
        result = parser.parse_informations({"some new field": "value"})

        assert result == {}


# ---------------------------------------------------------------------------
# End-to-end parsing of a synthetic descriptor
# ---------------------------------------------------------------------------


class TestParseSynthetic:
    def test_basic_fields(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert record.identifier == "nm0001"
        assert record.planet == {"number": 8, "name": "Neptune"}
        assert record.total_number == 3
        assert record.observables == "absolute_absolute_CCD_nsdb"
        assert record.dates == {"start_year": 2001, "end_year": 2001}

    def test_observatory(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert record.observatories == [
            {"code": 673, "name": "Table Mountain Observatory, Wrightwood"}
        ]
        assert record.primary_observatory_code == 673

    def test_informations(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert record.parameters["reference_frame"] == "astrometric"
        assert record.parameters["epoch_of_equinox"] == "J2000"
        assert record.parameters["receptor"] == "CCD"

    def test_telescope_is_free_text(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert record.telescopes == "Reflector, D = 0.61 m, f/16"

    def test_reference_and_comments(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert record.reference is not None and "Communicated" in record.reference
        assert record.comments == "no information"

    def test_format_columns_exclude_the_sample_row(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert sorted(record.format_columns) == [1, 2, 3]

    def test_identifier_falls_back_to_title(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY, title="nm9999"))

        assert record.identifier == "nm9999"

    def test_explicit_identifier_wins_over_title(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY, title="um0036"), "nm0036")

        assert record.identifier == "nm0036"

    def test_raw_sections_are_retained(self, parser):
        record = parser.parse(descriptor(SIMPLE_BODY), "nm0001")

        assert "planet: 8 - Neptune" in record.sections["contents"]


class TestParseErrors:
    def test_missing_contents_section(self, parser):
        with pytest.raises(ValueError, match="no Contents"):
            parser.parse(descriptor("Format.\n1. a\n"), "nm0001")

    def test_missing_format_section(self, parser):
        with pytest.raises(ValueError, match="no Format"):
            parser.parse(descriptor("Contents.\nplanet: 8 - Neptune\n"), "nm0001")

    def test_format_with_no_columns(self, parser):
        body = "Contents.\nplanet: 8 - Neptune\nInformations.\nreference frame: x\nFormat.\nprose\n"
        with pytest.raises(ValueError, match="no numbered columns"):
            parser.parse(descriptor(body), "nm0001")

    def test_empty_pre_block(self, parser):
        with pytest.raises(ValueError, match="empty"):
            parser.parse("<HTML><PRE></PRE></HTML>", "nm0001")


class TestMissingInformationsRecovery:
    NO_INFORMATIONS_HEADING = """Contents.
      planet: 8 - Neptune
  satellites: N1-Triton : 129
total number: 129
        type: relative
       dates: 1899-1900
 observatory: 786 - U.S. Naval Obs., Washington

References.
   See T. J. J. (1900)
   Researches on the Orbit of the Satellite of Neptune

         relative to: planet
     reference frame: apparent
     centre of frame: topocentre
    epoch of equinox: date
          time scale: Washington Mean Time
           reduction: no information
         coordinates: position angle and separation
        diff. refraction: not information
            receptor: micrometric
           telescope: 26-inch
           observers: See
 data included in standard data file: no

Comments.

Format.
  1. Year of the moment of observation
  2. Month of the moment of observation
  3. Day of the moment of observation

------------------------------------
Year        W.M.T.     P
  month               deg.      S
      day   h  m              arcsec
------------------------------------
"""

    def test_metadata_is_recovered(self, parser):
        """``nm0036`` has no ``Informations.`` heading; its fields follow it.

        The block is found by vocabulary, so the astrometric metadata is not
        lost just because the heading is absent.
        """
        record = parser.parse(descriptor(self.NO_INFORMATIONS_HEADING), "nm0036")

        assert record.parameters["reference_frame"] == "apparent"
        assert record.parameters["epoch_of_equinox"] == "date"
        assert record.parameters["time_scale"] == "Washington Mean Time"

    def test_bibliography_is_not_mistaken_for_metadata(self, parser):
        record = parser.parse(descriptor(self.NO_INFORMATIONS_HEADING), "nm0036")

        assert record.reference is not None
        assert "See T. J. J." in record.reference

    def test_type_is_still_fused(self, parser):
        record = parser.parse(descriptor(self.NO_INFORMATIONS_HEADING), "nm0036")

        assert record.observables == ("relative_position_angle_and_separation_micrometric_nsdb")


# ---------------------------------------------------------------------------
# Config output
# ---------------------------------------------------------------------------


class TestToConfig:
    def test_identifier_and_format_columns(self, parser):
        config = parser.parse(descriptor(SIMPLE_BODY), "nm0001").to_config()

        assert config["identifier"] == "nm0001"
        assert config["format_columns"] == {
            "1": "Number of satellite (N sat)",
            "2": "Year of the moment of observation",
            "3": "Day of the moment of observation with decimals",
        }

    def test_fused_type_key(self, parser):
        config = parser.parse(descriptor(SIMPLE_BODY), "nm0001").to_config()

        assert config["type"] == "absolute_absolute_CCD_nsdb"

    def test_both_centre_spellings(self, parser):
        config = parser.parse(descriptor(SIMPLE_BODY), "nm0001").to_config()

        assert config["centre_of_frame"] == "topocentre"
        assert config["center_of_frame"] == "topocentre"

    def test_data_file_is_absolute(self, parser, tmp_path):
        config = parser.parse(descriptor(SIMPLE_BODY), "nm0001").to_config(tmp_path / "x.txt")

        assert Path(config["file"]).is_absolute()

    def test_data_file_omitted_when_not_given(self, parser):
        config = parser.parse(descriptor(SIMPLE_BODY), "nm0001").to_config()

        assert "file" not in config

    def test_key_order_is_stable(self, parser):
        config = parser.parse(descriptor(SIMPLE_BODY), "nm0001").to_config()

        assert list(config)[:2] == ["identifier", "format_columns"]


# ---------------------------------------------------------------------------
# The real corpus
# ---------------------------------------------------------------------------


pytestmark_skip_without_corpus = pytest.mark.skipif(
    not REAL_DESCRIPTORS,
    reason="data/raw has no NSDB descriptors; run scripts/generate_collection.py first",
)


@pytestmark_skip_without_corpus
class TestRealCorpus:
    """Run the parser over every descriptor in ``data/raw``.

    These are the tests that matter most: the heading variants, the
    bibliography that masquerades as an observatory, and the sample rows in the
    Format section are all invisible in hand-written input.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def records(cls) -> dict[str, NSDBRecord]:
        parser = NSDBParser()
        return {path.stem: parser.parse_file(path) for path in REAL_DESCRIPTORS}

    def test_every_descriptor_parses(self, records):
        assert len(records) == len(REAL_DESCRIPTORS)
        assert len(records) >= 80

    def test_no_format_index_gaps(self, records):
        """Declared column indices must be exactly 1..N.

        A gap means the separator rule failed and a sample data row was read as
        a declaration. Five datasets used to fail this with indices 15, 35, 65,
        197 and 272.
        """
        for name, record in records.items():
            indices = sorted(record.format_columns)
            assert indices == list(range(1, len(indices) + 1)), f"{name} has {indices}"

    def test_no_unparsed_observatory_lines(self, records):
        """Every observatory line must be understood.

        A non-empty list here means prose reached the field, which is how the
        bibliography leak would show up.
        """
        offenders = {
            name: record.unparsed_observatory_lines[:2]
            for name, record in records.items()
            if record.unparsed_observatory_lines
        }
        assert offenders == {}

    def test_observatory_entries_are_not_prose(self, records):
        """No bibliography text may appear in an observatory entry.

        Before the three-digit code bound, 49 datasets produced 395 bogus
        entries such as ``{'code': 18743549, 'name': ''}``.
        """
        total = 0
        for name, record in records.items():
            for observatory in record.observatories:
                total += 1
                entry = f"{observatory['name']} {observatory['code']}"
                assert observatory["name"] is not None, name
                assert not any(
                    marker in entry for marker in ("MNRAS", "Press", "bulletin", "University")
                ), f"{name}: {observatory}"
        # A handful of datasets legitimately list several instruments.
        assert 80 <= total <= 90

    def test_every_record_has_a_fused_type(self, records):
        missing = [name for name, record in records.items() if not record.observables]
        assert missing == []

    def test_every_record_has_a_reference_frame(self, records):
        """All 83 datasets describe their frame somewhere, including nm0036."""
        missing = [
            name for name, record in records.items() if not record.parameters.get("reference_frame")
        ]
        assert missing == []

    def test_reference_section_is_found_almost_everywhere(self, records):
        """The plural ``References.`` heading must be recognised.

        Only 31 of 83 were found before the heading aliases were added; the
        three that remain genuinely have no reference block.
        """
        found = [name for name, record in records.items() if record.reference]
        assert len(found) >= 78

    def test_multi_observatory_datasets_are_detected(self, records):
        multi = {name for name, record in records.items() if len(record.observatories) > 1}
        assert multi == {"nm0019", "nm0083"}

    def test_telescope_index_exists_only_where_expected(self, records):
        indexed = {name for name, record in records.items() if record.telescope_index}
        assert indexed == {"nm0019", "nm0083"}

    def test_structured_telescopes_only_where_expected(self, records):
        """Only the two multi-telescope datasets have a structured field."""
        structured = {
            name for name, record in records.items() if isinstance(record.telescopes, list)
        }
        assert structured == {"nm0019", "nm0083"}

    def test_discontinuous_date_ranges_are_sane(self, records):
        """No year may be a fused run of digits."""
        for name, record in records.items():
            if record.dates is None:
                continue
            for key, year in record.dates.items():
                assert 1600 <= year <= 2100, f"{name}.{key} = {year}"

    @pytest.mark.parametrize("name", ["nm0034", "nm0045", "nm0053", "nm0080"])
    def test_known_discontinuous_dates(self, records, name):
        assert records[name].dates is not None
        assert records[name].dates["start_year"] < records[name].dates["end_year"]

    def test_nm0036_metadata_is_recovered(self, records):
        """It is the only dataset with no ``Informations.`` heading."""
        record = records["nm0036"]

        assert record.parameters["reference_frame"] == "apparent"
        assert record.parameters["epoch_of_equinox"] == "date"

    def test_config_round_trips_through_yaml(self, records):
        import yaml

        for name in ("nm0001", "nm0019", "nm0083", "nm0036"):
            config = records[name].to_config()
            restored = yaml.safe_load(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

            assert restored["identifier"] == config["identifier"]
            assert restored["type"] == config["type"]
            assert restored["format_columns"] == config["format_columns"]

    def test_primary_observatory_code_is_usable(self, records):
        """The factories read this value, so it must be a real MPC code.

        ``nm0051`` is the one exception: it names ``Cambridge`` with no code at
        all, which the parser records as ``None`` rather than inventing one.
        """
        for name, record in records.items():
            code = record.primary_observatory_code
            if name == "nm0051":
                assert code is None
                continue
            assert code is not None, f"{name} has no observatory at all"
            if isinstance(code, int):
                assert 0 <= code <= 999, f"{name}: {code}"
            else:
                assert re.fullmatch(r"[A-Z]\d{2}", code), f"{name}: {code!r}"
