"""Tests for :func:`orbitdet.utility.parse_duration_to_seconds`.

The interpolator validation uses this to accept human-readable buffers such as
``"6 months"`` in the Hydra config, so the parsing must be strict (unknown
units and typos raise) and deterministic (month/year use mean Gregorian
lengths, independent of the calendar date).
"""

import pytest

from orbitdet.utility import parse_duration_to_seconds

SECONDS_PER_DAY = 86400.0
SECONDS_PER_MONTH = 2629746.0
SECONDS_PER_YEAR = 31556952.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3600 s", 3600.0),
        ("3600s", 3600.0),
        ("60 min", 3600.0),
        ("1 h", 3600.0),
        ("12 h", 43200.0),
        ("1 day", SECONDS_PER_DAY),
        ("180 days", 180 * SECONDS_PER_DAY),
        ("2 weeks", 2 * 604800.0),
        ("6 months", 6 * SECONDS_PER_MONTH),
        ("6month", 6 * SECONDS_PER_MONTH),
        ("1 year", SECONDS_PER_YEAR),
        ("1.5 years", 1.5 * SECONDS_PER_YEAR),
        ("1 year 6 months", SECONDS_PER_YEAR + 6 * SECONDS_PER_MONTH),
        ("0 s", 0.0),
    ],
)
def test_parse_duration_strings(value, expected):
    assert parse_duration_to_seconds(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [3600, 3600.0, 0])
def test_parse_duration_numbers_are_seconds(value):
    assert parse_duration_to_seconds(value) == float(value)


@pytest.mark.parametrize("value", ["", "   ", "abc", "6 fortnights", "6 months extra"])
def test_parse_duration_rejects_bad_strings(value):
    with pytest.raises(ValueError):
        parse_duration_to_seconds(value)


def test_parse_duration_rejects_boolean():
    with pytest.raises(TypeError):
        parse_duration_to_seconds(True)


def test_parse_duration_rejects_unsupported_type():
    with pytest.raises(TypeError):
        parse_duration_to_seconds(None)
