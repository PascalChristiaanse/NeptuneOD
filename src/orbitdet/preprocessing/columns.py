"""Column-name inference and the format-column overlay.

Observation sources name their columns in prose, and they all do it differently.
An NSDB "Format" block may call a column ``Hour   of right ascension (alpha, h)``
while a Voyager table calls the same quantity ``RA_h``; a Gaia scan column may be
``scanAngle``.  Before any astrometry can happen, the column holding a given
quantity has to be *found*, and this module is the only place that search lives.

Two jobs, both pure - no files, no config, no Tudat:

Finding columns
    :func:`normalize_column_name` reduces a header to a comparable token
    sequence, and :func:`find_column` / :func:`find_column_substring` search for
    one or more of those tokens.  Both take an ``excludes`` list, because the
    NSDB vocabulary is ambiguous: ``second`` is both a unit of time and a unit of
    arc, and ``hour`` is both an observation time and a right-ascension
    component.  Exclusions are how the caller states which reading it wants,
    rather than the finder guessing.

Applying the source's own column map
    :func:`apply_format_columns` overlays a source-declared index-to-name mapping
    onto a headerless frame.  NSDB numbers its Format entries from 1, and several
    datasets declare a different number of entries than the whitespace split
    produces - because the file is fixed-width and blank fields collapse - so the
    overlay is bounds-checked and partial rather than positional and total.

Why this is a separate module
    These functions were previously private to ``observations/nsdb_helpers.py``
    and duplicated verbatim in three factories, even though a non-NSDB modality
    (Voyager) depends on them.  Column inference is a source-agnostic
    preprocessing concern, so it lives here and the factories import it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_column_name",
    "find_column",
    "find_column_substring",
    "numeric_series",
    "apply_format_columns",
    "relative_position_scale",
    "ARCSECOND_TO_RADIANS",
    "SECONDS_OF_TIME_TO_RADIANS",
]


#: One arcsecond in radians.
ARCSECOND_TO_RADIANS = np.pi / 648000.0

#: One second of time, expressed as an angle, in radians.  A second of time is
#: 1/240 of a degree, so 15 arcsec - the conversion for differential right
#: ascension, which is measured in time units.
SECONDS_OF_TIME_TO_RADIANS = np.pi / 43200.0


def normalize_column_name(column_name: object) -> str:
    """Reduce a column header to a lower-case, single-spaced token sequence.

    Underscores become spaces so that ``RA_h`` and ``RA h`` compare equal, and
    runs of whitespace collapse so that the alignment padding NSDB uses inside
    its Format descriptions does not affect matching.

    Args:
        column_name: The raw column label.

    Returns:
        The normalised label.

    Examples:
        >>> normalize_column_name("Hour   of right ascension (alpha, h)")
        'hour of right ascension (alpha, h)'
        >>> normalize_column_name("scan_angle")
        'scan angle'
    """
    return " ".join(str(column_name).strip().lower().replace("_", " ").split())


def _matches_exclusion(normalized: str, excludes: Iterable[str]) -> bool:
    """Whether ``normalized`` contains any excluded word.

    An optional trailing ``s`` is allowed on the exclusion, so that excluding
    ``second`` also excludes a column labelled ``seconds``.  The word boundary
    matters: excluding ``ra`` must not reject a column merely containing those
    letters.
    """
    return any(re.search(rf"\b{re.escape(ex)}s?\b", normalized) for ex in excludes)


def _matches_needle(normalized: str, needle: str) -> bool:
    """Whether ``normalized`` contains ``needle`` as a word, singular or plural.

    The vocabulary is genuinely mixed - ``Hour of the observation`` sits beside
    ``Seconds of the observation`` - so accepting both forms is what keeps a
    component from being dropped silently.  Word boundaries are still enforced,
    so a search for ``ra`` does not match ``arcsec``.
    """
    return bool(re.search(rf"\b{re.escape(needle)}s?\b", normalized))


def find_column(
    dataframe: pd.DataFrame, *needles: str, excludes: list[str] | None = None
) -> str | None:
    """Find the first column whose label contains all ``needles`` as words.

    A needle matches whether it appears in singular or plural form, so a search
    for ``"second"`` also accepts a column labelled ``Seconds of the
    observation``.  This matters because the NSDB vocabulary mixes the two
    freely - ``Hour of the observation`` alongside ``Seconds of the
    observation`` - and a strict-singular match silently drops the component,
    losing the sub-minute part of the epoch rather than failing.

    Args:
        dataframe: The frame whose columns are searched.
        *needles: Tokens that must all appear.  The special token ``"jd"`` also
            matches the word ``julian``, because sources write both ``JD`` and
            ``Julian date`` for the same quantity.
        excludes: Tokens that disqualify a column.  Plurals are accepted here
            too, so excluding ``second`` also rejects ``seconds``.

    Returns:
        The matching column label as it appears in the frame, or ``None``.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"JD of the moment of observation": [1.0]})
        >>> find_column(frame, "jd")
        'JD of the moment of observation'
        >>> find_column(pd.DataFrame({"Seconds of the observation": [1]}), "second")
        'Seconds of the observation'
    """
    excludes = excludes or []
    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)

        if _matches_exclusion(normalized, excludes):
            continue

        matched_all = True
        for needle in needles:
            if needle == "jd":
                # Sources write both "JD" and "Julian date"; neither is a
                # substring of the other, so both are accepted here.
                if not ("jd" in normalized or "julian" in normalized):
                    matched_all = False
                    break
            elif not _matches_needle(normalized, needle):
                matched_all = False
                break

        if matched_all:
            return str(column_name)

    return None


def find_column_substring(
    dataframe: pd.DataFrame, *needles: str, excludes: list[str] | None = None
) -> str | None:
    """Find the first column containing all ``needles`` as substrings.

    Weaker than :func:`find_column`, and used where the label is a compound
    phrase rather than a set of words - ``"right ascension"`` appears inside
    ``Second of right ascension (alpha, s)``, which whole-word matching on both
    tokens would also accept, but ``"ra"`` appears inside ``arcsec`` where
    whole-word matching would not.

    Args:
        dataframe: The frame whose columns are searched.
        *needles: Substrings that must all appear.
        excludes: Substrings that disqualify a column.

    Returns:
        The matching column label, or ``None``.
    """
    excludes = excludes or []
    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)

        if any(ex in normalized for ex in excludes):
            continue

        if all(needle in normalized for needle in needles):
            return str(column_name)

    return None


def numeric_series(dataframe: pd.DataFrame, column_name: str | None) -> pd.Series:
    """Return a column as floats, or zeros when the column is absent.

    Absent-component columns are normal, not exceptional: a source that records
    right ascension as degrees alone has no minute or second column, and treating
    those as zero is what makes the compound conversion uniform.

    Args:
        dataframe: The frame holding the column.
        column_name: The column to read, or ``None``.

    Returns:
        The column as ``float64``, or a zero series of the same length.
    """
    if column_name is None:
        return pd.Series(0.0, index=dataframe.index, dtype="float64")
    return pd.to_numeric(dataframe[column_name], errors="coerce")


def apply_format_columns(
    dataframe: pd.DataFrame,
    format_columns: Mapping[Any, str] | None,
) -> pd.DataFrame:
    """Overlay a source-declared index-to-name map onto a headerless frame.

    NSDB numbers its Format entries from 1, and a dataset's declared count can
    differ from the number of whitespace-separated fields actually produced,
    because the data files are fixed-width and blank fields collapse.  The
    overlay is therefore **partial and bounds-checked**: an entry whose index
    falls outside the frame's column range is skipped rather than raising, and
    columns the map does not mention keep their positional integer label.

    Args:
        dataframe: The frame read without a header.
        format_columns: Mapping from 1-based column index to description.  Keys
            may be ``int`` or a string spelling of one, and ``None`` is treated
            as an empty map.

    Returns:
        The same frame, with columns renamed in place.
    """
    if not format_columns:
        return dataframe

    fmt = dict(format_columns)
    column_names = list(dataframe.columns)

    def _sort_key(key: Any) -> tuple[int, Any]:
        try:
            return (0, int(key))
        except (TypeError, ValueError):
            return (1, str(key))

    for index in sorted(fmt.keys(), key=_sort_key):
        name = fmt.get(index, fmt.get(str(index), None))
        try:
            # NSDB formats are 1-based, so the first declared column is index 1.
            position = int(index) - 1
        except (TypeError, ValueError):
            position = None

        if position is not None and 0 <= position < len(column_names):
            if name is not None:
                column_names[position] = name

    dataframe.columns = column_names
    return dataframe


def relative_position_scale(column_name: str) -> float:
    """Infer the angular scale of a relative-offset column from its label.

    Relative offsets reach the pipeline in whichever unit the source happened to
    use, and the unit travels in the *column name* rather than a separate field.
    The three that occur are seconds of time (differential right ascension),
    arcseconds (differential declination and X/Y plates), and degrees.

    Args:
        column_name: The column label, e.g. ``"X, arcsec"``.

    Returns:
        Radians per unit of that column.

    Raises:
        ValueError: If no unit can be inferred from the label.  Raised rather
            than defaulting, because a silent wrong unit is a large astrometric
            error rather than a small one.

    Examples:
        >>> import numpy as np
        >>> bool(np.isclose(relative_position_scale("X, arcsec"), np.pi / 648000.0))
        True
    """
    normalized = normalize_column_name(column_name)

    if "sec of time" in normalized or "seconds of time" in normalized:
        return SECONDS_OF_TIME_TO_RADIANS

    if "arcsec" in normalized or "arc second" in normalized or "arcsecond" in normalized:
        return ARCSECOND_TO_RADIANS

    if "degree" in normalized or re.search(r"\bdeg\b", normalized):
        return np.pi / 180.0

    raise ValueError(f"Could not infer angular units from column '{column_name}'.")
