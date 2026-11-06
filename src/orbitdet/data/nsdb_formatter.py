"""Turn a parsed NSDB record plus its raw data file into the standard frame.

This module sits immediately after :mod:`orbitdet.data.nsdb_parser`.  The parser
answers *what the descriptor says*; the formatter answers *what the observations
are*, in the column layout the rest of the pipeline speaks.

Why this is not in ``preprocessing``
------------------------------------
``preprocessing`` holds computations - precession, timescale conversion,
aberration, catalogue ties, the gnomonic inverse.  Every one of those is pure
and source-agnostic: it takes numbers and returns numbers, and it would behave
identically for a Gaia table or a Voyager table.

Formatting is the opposite.  Deciding that ``Hour of right ascension (alpha, h)``
is a right-ascension component, that ``X, Y`` means tangent-plane coordinates and
``differential`` does not, that ``photograpihc`` is a misspelling of
``photographic``, or that a bare ``Cambridge`` observatory line carries no code -
all of that is knowledge about *this source*.  It belongs next to the NSDB reader,
not in a library of astrometry.

So the division is:

    nsdb_parser   ->  descriptor HTML   ->  NSDBRecord
    nsdb_formatter->  NSDBRecord + data ->  StandardObservationFrame   [this file]
    preprocessing ->  standard frame    ->  corrected standard frame   [computation]

The formatter calls into ``preprocessing`` for the two things that genuinely are
computation - the epoch conversion and the angular-unit scaling - and does
everything else itself.

The messy parts of the job
--------------------------
Three NSDB habits make formatting more than a rename:

**Column names are prose.**  Twenty datasets write the year as ``Year of the
moment of observation``, others as ``Year of observation``; a right-ascension
second is ``Second of right ascension (alpha, s)``, ``Seconds of right
ascension with decimals (alpha, s)``, or ``Second of right ascension (alpha, s)
with decimals``.  The vocabulary is searched, not indexed, and every search
needs exclusions because ``second`` is a unit of both time and arc.

**Coordinate kinds are encoded in two fields at once.**  ``coordinates: absolute``
with RA/Dec columns is a different measurement from ``coordinates: X, Y``, and
``coordinates: differential`` is different again - yet all three are "relative"
in the ``type`` field.  :func:`classify_coordinates` reconciles them against the
columns that are actually present.

**The declared type string is a historic artefact.**  ``observables`` fuses
category, coordinates and receptor into one registered key such as
``relative_position_angle_and_separation_photograpihc_nsdb`` - typo preserved -
and it is what the dataset-factory registry dispatches on.  The formatter keeps
emitting it unchanged, but *also* produces the normalised schema enums, so a
future dispatcher can key on those instead.

Units
-----
The standard frame is in **radians and TDB seconds since J2000**.  NSDB is in
degrees, hours, arcseconds and any of twenty timescales, so the formatter
converts - and records the source unit in ``first_unit`` / ``second_unit`` so the
conversion stays auditable.
"""

from __future__ import annotations

import io
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from orbitdet.preprocessing.catalogues import normalize_catalogue
from orbitdet.preprocessing.columns import (
    apply_format_columns,
    find_column,
    find_column_substring,
    normalize_column_name,
    numeric_series,
    relative_position_scale,
)
from orbitdet.preprocessing.observatories import normalize_observatory_code
from orbitdet.preprocessing.schema import (
    NOT_GIVEN,
    CoordinateKind,
    DataSource,
    FrameCenter,
    ObservationType,
    ReceptorType,
    ReferenceFrame,
    build_frame,
)
from orbitdet.preprocessing.time_formatting import to_seconds_since_j2000
from orbitdet.preprocessing.time_scales import to_tdb_seconds_since_j2000

from .nsdb_parser import NSDBRecord

logger = logging.getLogger(__name__)

__all__ = [
    "EPOCH_COLUMN",
    "RELATIVE_POSITION_X_COLUMN",
    "RELATIVE_POSITION_Y_COLUMN",
    "NSDBFormatter",
    "classify_coordinates",
    "find_observatory_selector_column",
    "find_radec_columns",
    "find_relative_columns",
    "group_rows_by_observatory",
    "normalize_frame_center",
    "normalize_observation_type",
    "normalize_receptor",
    "normalize_reference_frame",
    "resolve_epoch_of_equinox",
    "resolve_observatory_codes",
    "set_epoch_column",
    "set_ra_dec_columns",
    "set_relative_position_columns",
]


#: Column words marking a *measurement uncertainty* rather than a measurement.
#: These reach the coordinate finders as plausible candidates - ``Uncertainty of
#: right ascension (err_ra, mas)`` contains both ``ra`` and ``right ascension`` -
#: and would otherwise be read as the coordinate itself.  Two Uranian datasets
#: (um0124, um0125) were silently supplying milliarcsecond errors as right
#: ascension, a value that looks plausible in radians and so passed every check.
_UNCERTAINTY_EXCLUDES = [
    "uncertain",
    "uncertainty",
    "error",
    "err",
    "sigma",
    "precision",
    "accuracy",
    "residual",
    "o-c",
]

#: Columns that name a right-ascension or declination *component* rather than a
#: quantity of their own.
_RA_COMPONENT_EXCLUDES = [
    "declination",
    "delta",
    "hour",
    "minute",
    "second",
    "parallax",
    "arcsec",
    *_UNCERTAINTY_EXCLUDES,
]

_DEC_COMPONENT_EXCLUDES = [
    "ra",
    "right",
    "ascension",
    "alpha",
    "hour",
    "minute",
    "second",
    "parallax",
    "arcsec",
    *_UNCERTAINTY_EXCLUDES,
]

#: Columns to ignore when looking for an RA/Dec component, because they are
#: observation-time components or measurement noise columns.
_TIME_COMPONENT_EXCLUDES = ["observation time", "utc", "parallax", "arcsec"]

#: Substring marking the Gaia DR1 catalogue reference epoch.  See
#: :func:`resolve_epoch_of_equinox`.
_GAIA_DR1_EPOCH_MARKER = "2015"

#: Coordinate labels, mapped onto :class:`CoordinateKind` plus the *shape* of the
#: pair.  ``"radec"`` means the two values are an (RA, Dec) or (dRA, dDec) pair;
#: ``"xy"`` means tangent-plane coordinates; ``"pa_sep"`` means position angle
#: and separation.  The distinction matters because ``differential`` names a
#: relative measurement in RA/Dec, not tangent-plane X/Y.
#:
#: Order is load-bearing.  The ``X, Y`` pattern deliberately accepts a trailing
#: qualifier, because the Jovian catalogue spells out the reference frame in the
#: label - ``X, Y (topocentric for mutual occultation or heliocentric for mutual
#: eclipse)`` - and that parenthetical does not change what the pair *is*.
_COORDINATE_LABELS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^absolute"), "radec"),
    (re.compile(r"right ascension.*declination"), "radec"),
    # ``differential`` in both its bare and annotated forms (``differential:
    # delta(Alpha), delta(Delta)``); the annotation only names the components.
    (re.compile(r"^differential\b"), "radec"),
    (re.compile(r"delta\s*\(\s*alpha"), "radec"),
    (re.compile(r"^x\s*,\s*y\b"), "xy"),
    # Separation first, angle second: ``apparent distance, position angle``.
    (re.compile(r"(separation|distance).*position angle"), "pa_sep"),
    (re.compile(r"position angle.*(separation|distance)"), "pa_sep"),
    (re.compile(r"^position angle"), "pa_sep"),
)

#: Receptor labels, mapped onto :class:`ReceptorType`.  The misspelling
#: ``photograpihc`` occurs twice in the NSDB corpus and is preserved in the fused
#: type string, but normalised here.
_RECEPTOR_LABELS: tuple[tuple[re.Pattern[str], ReceptorType], ...] = (
    (re.compile(r"^ccd"), ReceptorType.CCD),
    (re.compile(r"photo"), ReceptorType.PHOTOGRAPHIC),
    (re.compile(r"^visual"), ReceptorType.VISUAL),
    (re.compile(r"micrometric"), ReceptorType.MICROMETRIC),
    (re.compile(r"meridian"), ReceptorType.MERIDIAN_TRANSIT_CIRCLE),
    (re.compile(r"vidicon"), ReceptorType.VIDICON),
)

#: Frame-centre labels.  NSDB writes the British spelling ``topocentre``.
_FRAME_CENTER_LABELS: tuple[tuple[re.Pattern[str], FrameCenter], ...] = (
    (re.compile(r"topocent"), FrameCenter.TOPOCENTRIC),
    (re.compile(r"geocent"), FrameCenter.GEOCENTRIC),
    (re.compile(r"barycent"), FrameCenter.BARYCENTRIC),
    (re.compile(r"heliocent"), FrameCenter.HELIOCENTRIC),
    (re.compile(r"spacecraft|space craft|satellite"), FrameCenter.SPACECRAFT),
)

#: Pattern matching NSDB's own sentinel declaration in a column heading, e.g.
#: ``"P (position angle) in degrees (999 if not given)"``.  The declared number is
#: the authoritative sentinel for that column, so it is read rather than assumed
#: from a fixed list.  Across the Neptune corpus the declarations are ``999``
#: (position angle in degrees, 50 columns), ``99`` (position angle in arcminutes,
#: 7 columns) and ``-99.99`` (separation in arcseconds, 51 columns).
#:
#: Two phrasings occur.  The Neptune catalogue writes ``(999 if not given)``;
#: the Uranian catalogue names the quantity instead, ``(999.9 if P is not
#: given)``.  Both are matched, and the trailing ``(?:\s+\d+)*`` covers a
#: combined declaration such as ``(99 99 99 if S is not given)``, where one
#: heading covers an hour/minute/second triple.
_NOT_GIVEN_DECLARATION = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*(?:\s+\d+)*\s+if\s+"
    r"(?:not\s+given|(?:\w+\s+)*is\s+not\s+given)\s*\)",
    re.I,
)


#: A declared sentinel is an integer, but NSDB writes it at whatever precision
#: the column happens to carry: it declares ``999`` and stores ``999``, ``999.9``
#: or ``999.99`` depending on the file.  A stored value therefore counts as the
#: sentinel when it is within one unit of the declaration.  That window cannot
#: swallow real data, because every declaration sits far outside its column's
#: physical range: a position angle is at most 360 as against a declared 999, an
#: arcminute field is at most 60 as against a declared 99, and a separation is at
#: most a few tens of arcseconds as against a declared -99.99.
_SENTINEL_TOLERANCE = 1.0


def _declared_sentinel(column_name: str) -> float | None:
    """Return the sentinel NSDB declares for a column, if any.

    Args:
        column_name: The raw NSDB heading, which carries the declaration.

    Returns:
        The declared sentinel, or ``None`` when the column declares no sentinel -
        in which case nothing in that column is treated as missing.
    """
    match = _NOT_GIVEN_DECLARATION.search(str(column_name))
    if match is None:
        return None
    return float(match.group(1))


def _flag_not_given(values: np.ndarray, column_name: str) -> np.ndarray:
    """Replace a column's declared sentinel with the schema's marker.

    The sentinel is read from the column's own heading, so a value is only
    treated as missing in the column that declared it.  That distinction matters
    because the declarations are ordinary numbers in the units of their column:
    ``99.99`` degrees is a legitimate position angle, and ``99`` is only a
    sentinel in an arcminute field.

    Args:
        values: Raw values for one measured quantity.
        column_name: The NSDB heading the values came from.

    Returns:
        The values with any declared sentinel replaced by
        :data:`orbitdet.preprocessing.schema.NOT_GIVEN`, and nulls likewise
        marked not-given.  A null and a sentinel mean the same thing here - the
        source recorded no measurement - and the marker keeps the row, because
        the *other* component of the same observation may well have been measured.
    """
    out = np.asarray(values, dtype="float64").copy()
    missing = ~np.isfinite(out)

    declared = _declared_sentinel(column_name)
    if declared is not None:
        missing |= np.isclose(out, declared, rtol=0.0, atol=_SENTINEL_TOLERANCE)

    out[missing] = NOT_GIVEN
    return out


def resolve_epoch_of_equinox(epoch_of_equinox: object, *, dataset: str = "") -> str:
    """Return the equinox to precess from, overriding the misleading ``J2015.0``.

    ``nm0081`` (Wang et al. 2017, Gaia DR1) declares
    ``epoch of equinox: J2015.0``, but 2015.0 is Gaia DR1's *catalogue
    reference epoch* - the mean epoch of the observations that went into the
    catalogue - not the equinox its coordinates are referred to.  The frame
    itself is ICRS/J2000-aligned.

    Taking the label at face value precesses the data by ~15 years of
    precession, about **700 arcsec**, which is the dominant error in the
    dataset: measured against SPICE ``nep097``, nm0081 agrees with the Triton
    ephemeris to ~3 arcsec as given, and ~700 arcsec once precessed.  Its Gaia
    DR2 siblings (nm0082/nm0084/nm0086) are all labelled ``J2000`` despite DR2's
    catalogue epoch being 2015.5, which shows the field is populated
    inconsistently across NSDB.

    A ``2015`` label is therefore treated as J2000 and a warning is logged.  This
    is a deliberate, reversible override: delete :data:`_GAIA_DR1_EPOCH_MARKER`
    and the check below once the underlying data is corrected upstream.

    Args:
        epoch_of_equinox: The raw label from the descriptor.
        dataset: Dataset identifier, used only to make the warning actionable.

    Returns:
        ``"J2000"`` for a 2015 label, otherwise the label unchanged.

    Examples:
        >>> resolve_epoch_of_equinox("J2015.0")
        'J2000'
        >>> resolve_epoch_of_equinox("date")
        'date'
        >>> resolve_epoch_of_equinox("J2000")
        'J2000'
    """
    label = "" if epoch_of_equinox is None else str(epoch_of_equinox)
    if _GAIA_DR1_EPOCH_MARKER in label:
        logger.warning(
            "Dataset %s declares 'epoch of equinox: %s', which is the Gaia DR1 "
            "catalogue reference epoch rather than an equinox; its coordinates "
            "are already J2000/ICRS-aligned. Treating it as J2000. Precessing it "
            "instead would shift the data by ~700 arcsec.",
            dataset or "<unknown>",
            label,
        )
        return "J2000"
    return label


def classify_coordinates(
    coordinates: object,
    dataframe: pd.DataFrame | None = None,
) -> tuple[CoordinateKind, str]:
    """Reconcile the descriptor's coordinate label with the columns present.

    The descriptor alone is not enough.  ``coordinates: absolute`` describes an
    RA/Dec pair, but so does ``coordinates: right ascension, declination``, and
    ``coordinates: differential`` means a *relative* RA/Dec pair - while
    ``coordinates: X, Y`` is a tangent-plane pair that the gnomonic inversion
    must handle instead.  Three different ``CoordinateKind`` values can hide
    behind one label, so the columns are consulted as a cross-check.

    Args:
        coordinates: The ``coordinates`` field from the descriptor.
        dataframe: The formatted data columns, if available.  Used to confirm the
            label against what is actually present.

    Returns:
        Tuple ``(coordinate_kind, shape)`` where shape is ``"radec"``, ``"xy"``
        or ``"pa_sep"``.

    Raises:
        ValueError: If the label names no known coordinate kind.

    Examples:
        >>> classify_coordinates("X, Y")[0].value
        'cartesian_xy'
        >>> classify_coordinates("position angle and separation")[0].value
        'position_angle_separation'
    """
    text = "" if coordinates is None else " ".join(str(coordinates).strip().lower().split())

    for pattern, shape in _COORDINATE_LABELS:
        if pattern.search(text):
            break
    else:
        raise ValueError(
            f"Unrecognised NSDB coordinates label {coordinates!r}. Expected one of "
            f"'absolute', 'right ascension, declination', 'differential', 'X, Y', "
            f"or 'position angle and separation'."
        )

    kind = {
        "radec": CoordinateKind.RA_DEC,
        "xy": CoordinateKind.CARTESIAN_XY,
        "pa_sep": CoordinateKind.POSITION_ANGLE_SEPARATION,
    }[shape]

    # A label of "absolute" that turns out to sit beside X/Y columns is really a
    # tangent-plane pair; trust the columns over the label.
    if shape == "radec" and dataframe is not None:
        ra_column, dec_column = find_radec_columns(dataframe, required=False)
        if ra_column is None or dec_column is None:
            x_column = _find_relative_x_column(dataframe)
            y_column = _find_relative_y_column(dataframe)
            if x_column is not None and y_column is not None:
                logger.info(
                    "Descriptor labels the coordinates %r but the data carries X/Y "
                    "columns; treating them as tangent-plane coordinates.",
                    coordinates,
                )
                return CoordinateKind.CARTESIAN_XY, "xy"

    return kind, shape


def normalize_receptor(receptor: object) -> ReceptorType:
    """Map a free-text receptor label onto :class:`ReceptorType`.

    Args:
        receptor: The ``receptor`` field from the descriptor.

    Returns:
        The matching receptor, or :attr:`ReceptorType.UNKNOWN` for a label the
        vocabulary does not cover.  ``"no inf"`` occurs once in the corpus.

    Examples:
        >>> normalize_receptor("CCD").value
        'ccd'
        >>> normalize_receptor("photograpihc").value
        'photographic'
        >>> normalize_receptor("no inf").value
        'unknown'
    """
    text = "" if receptor is None else str(receptor).strip().lower()
    if "no inf" in text or not text:
        return ReceptorType.UNKNOWN
    for pattern, receptor_type in _RECEPTOR_LABELS:
        if pattern.search(text):
            return receptor_type
    logger.warning("Unrecognised NSDB receptor label %r; recording UNKNOWN.", receptor)
    return ReceptorType.UNKNOWN


def normalize_frame_center(center_of_frame: object) -> FrameCenter:
    """Map an NSDB frame-centre label onto :class:`FrameCenter`.

    Args:
        center_of_frame: The ``centre of frame`` field from the descriptor.

    Returns:
        The matching centre, or :attr:`FrameCenter.UNKNOWN`.

    Examples:
        >>> normalize_frame_center("topocentre").value
        'topocentric'
        >>> normalize_frame_center("geocentric").value
        'geocentric'
    """
    text = "" if center_of_frame is None else str(center_of_frame).strip().lower()
    for pattern, center in _FRAME_CENTER_LABELS:
        if pattern.search(text):
            return center
    if text:
        logger.warning("Unrecognised NSDB frame centre %r; recording UNKNOWN.", center_of_frame)
    return FrameCenter.UNKNOWN


def normalize_reference_frame(reference_frame: object) -> ReferenceFrame:
    """Map an NSDB reference-frame label onto :class:`ReferenceFrame`.

    Args:
        reference_frame: The ``reference frame`` field from the descriptor.

    Returns:
        The matching kind, or :attr:`ReferenceFrame.UNKNOWN`.

    Examples:
        >>> normalize_reference_frame("astrometric").value
        'astrometric'
        >>> normalize_reference_frame("apparent").value
        'apparent'
        >>> normalize_reference_frame("BCRS").value
        'bcrs'
    """
    text = "" if reference_frame is None else str(reference_frame).strip().lower()
    if "astrometric" in text:
        return ReferenceFrame.ASTROMETRIC
    if "apparent" in text:
        return ReferenceFrame.APPARENT
    if "bcr" in text:
        return ReferenceFrame.BCRS
    if text:
        logger.warning("Unrecognised NSDB reference frame %r; recording UNKNOWN.", reference_frame)
    return ReferenceFrame.UNKNOWN


def normalize_observation_type(observation_type: object) -> ObservationType:
    """Map the descriptor's ``type`` field onto :class:`ObservationType`.

    Args:
        observation_type: The ``type`` field, e.g. ``"absolute"``.

    Returns:
        :attr:`ObservationType.RELATIVE` for anything that is not explicitly
        absolute, because a relative measurement is the safer default: it does
        not assert that the coordinate is a sky direction.

    Examples:
        >>> normalize_observation_type("absolute").value
        'absolute'
        >>> normalize_observation_type("relative").value
        'relative'
    """
    text = "" if observation_type is None else str(observation_type).strip().lower()
    return ObservationType.ABSOLUTE if text == "absolute" else ObservationType.RELATIVE


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------


def set_ra_dec_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """Infer the right-ascension and declination columns, converting them to degrees.

    Handles both layouts: a decimal-degree pair, and the sexagesimal ``h m s`` /
    ``d m s`` components the historical datasets use.  When components are found
    they are assembled into new ``ra`` / ``dec`` columns, so the returned names
    may not have existed beforehand.

    The unit of the result is **degrees**, matching NSDB's own convention at this
    stage; conversion to radians happens when the standard frame is built.

    Args:
        dataframe: The frame, with source column names applied.

    Returns:
        ``(ra_column, dec_column)``.

    Raises:
        RuntimeError: If either column cannot be located.
    """
    ra_column, dec_column = find_radec_columns(dataframe, required=True)
    # required=True has already raised if either was missing.
    assert ra_column is not None and dec_column is not None
    return ra_column, dec_column


def find_radec_columns(
    dataframe: pd.DataFrame, *, required: bool = True
) -> tuple[str | None, str | None]:
    """Locate the right-ascension and declination columns.

    Handles both layouts: a single decimal-degree pair, and the sexagesimal
    ``h m s`` / ``d m s`` components.  The components are assembled into two new
    ``ra`` / ``dec`` columns **in degrees**, which is why the caller must not
    assume the returned columns already existed.

    Args:
        dataframe: The frame, with NSDB column names applied.
        required: Whether to raise when either column cannot be found.

    Returns:
        Tuple ``(ra_column, dec_column)``, either of which may be ``None`` when
        ``required`` is false.

    Raises:
        RuntimeError: If ``required`` and a column cannot be located.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({"Hour of right ascension (alpha, h)": [22],
        ...                       "Degree of declination (delta, deg)": [-10]})
        >>> find_radec_columns(frame)
        ('ra', 'dec')
    """
    ra_column = (
        find_column(dataframe, "ra", excludes=_RA_COMPONENT_EXCLUDES)
        or find_column(dataframe, "alpha", excludes=_RA_COMPONENT_EXCLUDES)
        or find_column(dataframe, "right", "ascension", excludes=_RA_COMPONENT_EXCLUDES)
    )
    dec_column = (
        find_column(dataframe, "dec", excludes=_DEC_COMPONENT_EXCLUDES)
        or find_column(dataframe, "delta", excludes=_DEC_COMPONENT_EXCLUDES)
        or find_column(dataframe, "declination", excludes=_DEC_COMPONENT_EXCLUDES)
    )
    if ra_column is not None and dec_column is not None:
        return ra_column, dec_column

    # Only one of the two was found as a whole quantity, so the frame is the
    # sexagesimal layout and the other has to be assembled.  Assembly takes
    # precedence even over a column that *was* found, because a component search
    # can land on a component: ``Degree of declination (delta, deg)`` contains
    # the word ``delta``, so the whole-quantity search above accepts it and would
    # otherwise return a declination with its arcminutes and arcseconds silently
    # dropped - 12d 30' 15" collapsing to 12d.
    ra_column = _assemble_ra_components(dataframe) or ra_column
    dec_column = _assemble_dec_components(dataframe) or dec_column

    if ra_column is None or dec_column is None:
        if not required:
            return ra_column, dec_column
        message = (
            "Could not infer right ascension and/or declination columns. Expected "
            "RA/Alpha/Right Ascension and Dec/Delta/Declination columns, either in "
            "decimal degrees or as hour/minute/second and degree/minute/second."
        )
        logger.error(message)
        raise RuntimeError(message)

    return ra_column, dec_column


def _assemble_ra_components(dataframe: pd.DataFrame) -> str | None:
    """Combine hour/minute/second right-ascension columns into degrees.

    Right ascension is measured in *hours*, so the assembled value is multiplied
    by 15 to reach degrees.  Returns ``None`` when no component is present.
    """
    hour = _find_ra_component(dataframe, "hour")
    minute = _find_ra_component(dataframe, "minute")
    second = _find_ra_component(dataframe, "second")
    if hour is None and minute is None and second is None:
        return None

    hours = (
        numeric_series(dataframe, hour)
        + numeric_series(dataframe, minute) / 60.0
        + numeric_series(dataframe, second) / 3600.0
    )
    dataframe["ra"] = hours * 15.0
    return "ra"


def _assemble_dec_components(dataframe: pd.DataFrame) -> str | None:
    """Combine degree/minute/second declination columns into degrees.

    The sign is taken from the degree component alone, because minute and second
    are magnitudes in NSDB's layout even when the declination is negative.
    Returns ``None`` when no component is present.
    """
    degree = _find_dec_component(dataframe, "degree") or _find_dec_component(dataframe, "deg")
    minute = _find_dec_component(dataframe, "minute")
    second = _find_dec_component(dataframe, "second")
    if degree is None and minute is None and second is None:
        return None

    degrees = numeric_series(dataframe, degree)
    sign = np.sign(degrees).replace(0, 1)
    dataframe["dec"] = sign * (
        np.abs(degrees)
        + np.abs(numeric_series(dataframe, minute)) / 60.0
        + np.abs(numeric_series(dataframe, second)) / 3600.0
    )
    return "dec"


def _find_ra_component(dataframe: pd.DataFrame, component: str) -> str | None:
    """Find one right-ascension component column, tolerating the spelling variants."""
    excludes = _TIME_COMPONENT_EXCLUDES + _RA_COMPONENT_EXCLUDES
    return (
        find_column_substring(dataframe, "ra", component, excludes=excludes)
        or find_column_substring(dataframe, "alpha", component, excludes=excludes)
        or find_column_substring(
            dataframe, "right", "ascension", component, excludes=_TIME_COMPONENT_EXCLUDES
        )
    )


def _find_dec_component(dataframe: pd.DataFrame, component: str) -> str | None:
    """Find one declination component column, tolerating the spelling variants."""
    return (
        find_column_substring(dataframe, "dec", component, excludes=_DEC_COMPONENT_EXCLUDES)
        or find_column_substring(dataframe, "delta", component, excludes=_DEC_COMPONENT_EXCLUDES)
        or find_column_substring(dataframe, "declination", component)
    )


#: A relative RA/Dec pair written as ``delta(alpha)`` / ``delta(delta)``, as the
#: Saturnian and Jovian catalogues write it.  The parenthesised word names which
#: component the offset belongs to, so the pair must be matched on that word and
#: not on the presence of ``delta``, which both labels contain.
_DIFFERENTIAL_X_PATTERN = re.compile(r"delta\s*[\(\s]\s*(?:alpha|ra\b|r\.a\.)", re.IGNORECASE)
_DIFFERENTIAL_Y_PATTERN = re.compile(r"delta\s*[\(\s]\s*(?:delta|dec\b|declination)", re.IGNORECASE)


def _is_uncertainty_column(column_name: object) -> bool:
    """Whether a column label names a measurement uncertainty, not a measurement.

    Used by the relative-coordinate finders, which scan for a bare ``P``, ``S``,
    ``X`` or ``Y`` token.  An error column such as ``Precision of apparent
    position along the apparent relative trajectory`` contains none of those, but
    ``sigma of X in arcseconds`` does, so the check has to be explicit.
    """
    normalized = normalize_column_name(column_name)
    return any(re.search(rf"\b{re.escape(word)}\b", normalized) for word in _UNCERTAINTY_EXCLUDES)


def _find_relative_x_column(dataframe: pd.DataFrame) -> str | None:
    """Find the first-coordinate column of a relative RA/Dec or X/Y pair."""
    # An explicit ``delta(alpha)`` pair is checked first, because the generic
    # scan below would otherwise accept ``delta(delta)`` as the x column - both
    # labels contain ``delta``.
    for column_name in dataframe.columns:
        if _is_uncertainty_column(column_name):
            continue
        if _DIFFERENTIAL_X_PATTERN.search(str(column_name)):
            return str(column_name)

    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)
        if _is_uncertainty_column(column_name):
            continue
        if (
            "delta alpha" in normalized
            or "right ascension" in normalized
            or re.search(r"\balpha\b", normalized)
            or re.search(r"\bx\b", normalized)
        ):
            return str(column_name)
    return None


def _find_relative_y_column(dataframe: pd.DataFrame) -> str | None:
    """Find the second-coordinate column of a relative RA/Dec or X/Y pair."""
    for column_name in dataframe.columns:
        if _is_uncertainty_column(column_name):
            continue
        if _DIFFERENTIAL_Y_PATTERN.search(str(column_name)):
            return str(column_name)

    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)
        if _is_uncertainty_column(column_name):
            continue
        if (
            "delta delta" in normalized
            or "declination" in normalized
            or re.search(r"\bdec\b", normalized)
            or re.search(r"\by\b", normalized)
        ):
            return str(column_name)
    return None


def _find_position_angle_column(dataframe: pd.DataFrame) -> str | None:
    """Find the position-angle column of a position-angle/separation pair.

    NSDB writes the position angle as ``P``, ``P (position angle)``, ``Position
    angle, degrees`` or ``Position angle, Distance``.  The bare ``P`` form is
    matched on a word boundary so that it cannot capture an unrelated column
    whose label merely contains that letter.
    """
    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)
        if "position angle" in normalized:
            return str(column_name)
    for column_name in dataframe.columns:
        if re.search(r"\bp\b", normalize_column_name(column_name)):
            return str(column_name)
    return None


def _find_separation_column(dataframe: pd.DataFrame) -> str | None:
    """Find the separation column of a position-angle/separation pair.

    NSDB writes it as ``S``, ``S (separation)``, ``separation``, ``distance`` or
    ``Distance``.  Separated from the angle finder because the two are genuinely
    different quantities: the angle is a direction in radians, the separation a
    distance on the sky.
    """
    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)
        if "separation" in normalized or "distance" in normalized:
            return str(column_name)
    for column_name in dataframe.columns:
        if re.search(r"\bs\b", normalize_column_name(column_name)):
            return str(column_name)
    return None


def find_relative_columns(
    dataframe: pd.DataFrame,
    coordinate_kind: CoordinateKind = CoordinateKind.CARTESIAN_XY,
) -> tuple[str, str]:
    """Locate the two columns of a relative measurement pair.

    The kind decides which vocabulary is searched, because the two relative
    families are not interchangeable: a tangent-plane pair is written ``X`` / ``Y``
    (or ``delta alpha`` / ``delta delta``), while a position-angle pair is
    written ``P`` / ``S``.  Searching one vocabulary for the other silently
    matches nothing, which is exactly the failure mode this parameter removes.

    The unit is carried in the column *name* rather than a separate field, so the
    returned names are also what determines the angular scale applied later.

    Args:
        dataframe: The frame, with NSDB column names applied.
        coordinate_kind: Which pair to look for.

    Returns:
        Tuple ``(first_column, second_column)``.

    Raises:
        RuntimeError: If either column cannot be located.
    """
    if coordinate_kind is CoordinateKind.POSITION_ANGLE_SEPARATION:
        first = _find_position_angle_column(dataframe)
        second = _find_separation_column(dataframe)
        expected = "a position angle (P) and a separation (S)"
    else:
        first = _find_relative_x_column(dataframe)
        second = _find_relative_y_column(dataframe)
        expected = "delta alpha / delta delta or X / Y offsets"

    if first is None or second is None:
        message = (
            f"Could not infer relative position component columns for "
            f"{coordinate_kind.value}: expected {expected}. Available columns: "
            f"{sorted(map(str, dataframe.columns))}"
        )
        logger.error(message)
        raise RuntimeError(message)
    return first, second


def find_observatory_selector_column(dataframe: pd.DataFrame) -> str | None:
    """Find the column that selects the observatory per row, if there is one.

    Only two datasets need this: ``nm0083`` has a telescope-index column whose
    values map through ``telescope_index`` to observatory codes, and ``nm0019``
    has a column holding observatory codes directly.

    Args:
        dataframe: The frame, with NSDB column names applied.

    Returns:
        The column label, or ``None`` for a single-observatory dataset.
    """
    for column_name in dataframe.columns:
        normalized = normalize_column_name(column_name)
        if "telescope" in normalized or "observatory" in normalized:
            return str(column_name)
    return None


def resolve_observatory_codes(
    dataframe: pd.DataFrame,
    observatories: list,
    telescope_index: dict,
) -> pd.Series:
    """Resolve the observatory code for each row of a dataset.

    For a single-observatory dataset every row gets the one code.  For the two
    multi-observatory datasets a selector column decides, mapped through
    ``telescope_index``.

    Args:
        dataframe: The frame, with NSDB column names applied.
        observatories: The dataset's observatory entries, each with a ``code``.
            The first is used when there is no selector column.
        telescope_index: Mapping from per-row selector value to observatory code.
            Empty for a single-observatory dataset.

    Returns:
        One normalised observatory code per row, aligned with ``dataframe.index``.

    Raises:
        ValueError: If a selector column is required but absent, or a row
            references an index that ``telescope_index`` does not define.
    """
    if not telescope_index:
        code = _first_observatory_code(observatories)
        if code in (None, ""):
            # A handful of datasets name an observatory without an MPC code
            # (nm0051).  Padding that absence would produce "000", a code that
            # looks real; an empty string stays visibly absent.
            return pd.Series([""] * len(dataframe), index=dataframe.index)
        return pd.Series([normalize_observatory_code(code)] * len(dataframe), index=dataframe.index)

    selector = find_observatory_selector_column(dataframe)
    if selector is None:
        raise ValueError(
            "Multi-observatory dataset requires a telescope index or observatory code "
            "column, but none was found."
        )

    index_to_code = {str(key): value for key, value in telescope_index.items()}
    codes = dataframe[selector].map(lambda value: index_to_code.get(str(value).strip()))
    if codes.isna().any():
        raise ValueError(
            "Some observation rows reference a telescope index or observatory code not "
            "present in telescope_index."
        )
    return codes.map(normalize_observatory_code)


def group_rows_by_observatory(
    dataframe: pd.DataFrame,
    observatory_codes: pd.Series,
) -> list[tuple[str, pd.DataFrame]]:
    """Split a frame by observatory, in order of first appearance.

    Each cluster becomes its own observation set, because a Tudat observation set
    is built for one receiver.

    Args:
        dataframe: The frame to split.
        observatory_codes: One code per row, aligned with ``dataframe.index``.

    Returns:
        ``(observatory_code, rows)`` pairs, one per distinct observatory.
    """
    clusters: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for code, cluster in dataframe.groupby(observatory_codes, sort=False):
        if code in seen:
            continue
        seen.add(code)
        clusters.append((normalize_observatory_code(code), cluster))
    return clusters


def _first_observatory_code(observatories: object) -> object:
    """Read the code of a dataset's first observatory entry.

    Accepts the list-of-mappings form the parser emits and the bare mapping form
    that older configs in ``conf/datasets`` still carry, so a caller never has to
    know which convention its config happens to use.
    """
    if observatories is None:
        return ""
    if isinstance(observatories, dict) or hasattr(observatories, "get"):
        return observatories.get("code", "")
    try:
        first = next(iter(observatories))
    except (TypeError, StopIteration):
        return ""
    if isinstance(first, dict) or hasattr(first, "get"):
        return first.get("code", "")
    return ""


def set_epoch_column(
    dataframe: pd.DataFrame,
    time_scale: str | None = None,
    observatory_code: int | str | None = None,
    *,
    column: str = "epoch_TDB",
    in_place: bool = True,
) -> str:
    """Add the observation epoch in TDB seconds since J2000 to a frame.

    This is the single time entry point for the NSDB path.  It replaces the
    previous two-step arrangement, in which one helper produced an ISO-8601
    *string* and a second parsed that string and applied the timescale.  Fusing
    the two through a string was the source of three separate defects:

    * precision, because ``Timestamp`` is microsecond-resolution and the round
      trip through text quantised the epoch;
    * range, because pandas timestamps span only 1677-2262 while the corpus
      reaches back to 1847;
    * coverage, because the old converter recognised four timescale labels
      (``UT1``/``UTC``/``TT``/``TDB``) and raised on every other one - including
      the local mean times that 48 of the 82 committed configs declare.

    Formatting the epoch and applying its clock are now two calls to
    ``preprocessing``, and the intermediate representation is a number.

    Args:
        dataframe: The frame, with source column names applied.
        time_scale: The dataset's ``time_scale`` label.  ``None`` or empty means
            the epochs are already in the target scale and need no clock
            conversion, only formatting.
        observatory_code: MPC code of the observing station, needed only by the
            local mean-time labels, whose offset from UT1 depends on longitude.
        column: Name of the column to write.
        in_place: If ``True`` (default) add the column to ``dataframe``.

    Returns:
        The name of the epoch column.

    Raises:
        ValueError: If the frame has no recognisable epoch columns, or the
            timescale is a local mean time and neither an observatory code nor a
            named meridian is available.
    """
    # Formatting to a raw count first, then applying the clock, keeps a wrong
    # intermediate value out of the destination column: the seconds are staged
    # in a scratch column and only the converted result is published.
    staged = to_seconds_since_j2000(dataframe, in_place=False, column=_STAGED_EPOCH_COLUMN)
    epochs = staged[_STAGED_EPOCH_COLUMN].to_numpy("float64")

    if time_scale:
        epochs = to_tdb_seconds_since_j2000(epochs, time_scale, observatory_code=observatory_code)

    if in_place:
        dataframe[column] = epochs
    else:
        staged[column] = epochs
    return column


#: Column holding the observation epoch in TDB seconds since J2000, matching
#: ``tudatpy.astro.time_reference_epochs`` and the standard schema's
#: ``epoch_tdb``.
EPOCH_COLUMN = "epoch_TDB"

#: Scratch column used while converting epochs, so that a raw count is never
#: published under :data:`EPOCH_COLUMN` before the clock has been applied.
_STAGED_EPOCH_COLUMN = "_epoch_before_timescale"

#: Column names the relative-position helpers write.
RELATIVE_POSITION_X_COLUMN = "relative_position_x"
RELATIVE_POSITION_Y_COLUMN = "relative_position_y"


def set_relative_position_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """Infer a relative-offset pair and convert it to radians in place.

    Handles every relative layout the corpus uses: differential right ascension
    and declination (``Delta alpha`` / ``Delta delta``), tangent-plane offsets
    (``X`` / ``Y``), and the position-angle/separation pair.  The unit lives in
    the column name, so each component is scaled by its own.

    Args:
        dataframe: The frame, with source column names applied.

    Returns:
        ``(x_column, y_column)``, the names of the two new columns.

    Raises:
        RuntimeError: If the pair cannot be located.
    """
    kind = (
        CoordinateKind.POSITION_ANGLE_SEPARATION
        if find_column(dataframe, "position angle") is not None
        else CoordinateKind.CARTESIAN_XY
    )
    first, second = find_relative_columns(dataframe, kind)

    dataframe[RELATIVE_POSITION_X_COLUMN] = _scaled_measurement(dataframe, first)
    dataframe[RELATIVE_POSITION_Y_COLUMN] = _scaled_measurement(dataframe, second)
    return RELATIVE_POSITION_X_COLUMN, RELATIVE_POSITION_Y_COLUMN


# ---------------------------------------------------------------------------
# Row-level assembly
# ---------------------------------------------------------------------------


def _reject_duplicate_headings(dataframe: pd.DataFrame, record: NSDBRecord) -> None:
    """Fail loudly when a dataset's declared headings collide.

    A handful of datasets publish several *clusters* of the same quantity - five
    Uranian files carry three separate ``hour/minute/second`` triples, one per
    satellite - and copy the same heading text onto each cluster.  Pandas accepts
    the duplicate names, but every later lookup then returns a *DataFrame*
    instead of a Series, so the failure surfaces far away as an opaque
    ``TypeError: arg must be a list, tuple, 1-d array, or Series`` inside a
    numeric conversion.

    Raising here is deliberate, and better than disambiguating automatically.
    With several time triples in one file, picking one would attach the *right*
    column name to the *wrong* observation cluster, so the epochs would come out
    plausible and wrong.  An explicit error names the colliding headings and
    leaves the choice to whoever knows the dataset.

    Args:
        dataframe: The frame as read, with declared names applied.
        record: The parsed descriptor, used to name the dataset in the message.

    Raises:
        ValueError: If any heading appears more than once.
    """
    counts = Counter(str(name) for name in dataframe.columns)
    duplicates = {name: count for name, count in counts.items() if count > 1}
    if not duplicates:
        return

    described = ", ".join(f"{name!r} x{count}" for name, count in sorted(duplicates.items()))
    raise ValueError(
        f"Dataset {record.identifier}: the Format block declares duplicate column "
        f"heading(s): {described}. The file holds several clusters of the same quantity "
        f"and the headings do not distinguish them, so there is no way to tell which "
        f"cluster a later lookup should read. Disambiguate the headings (or add a "
        f"per-cluster selector) before this dataset can be formatted."
    )


class NSDBFormatter:
    """Assemble parsed NSDB records and their data files into standard frames.

    The formatter is the third stage of the NSDB read path, after download and
    parse.  It owns every decision that depends on NSDB's own conventions -
    column vocabulary, coordinate kind, catalogue label, observatory code - and
    delegates the two genuine computations to ``preprocessing``: epoch
    conversion and angular-unit scaling.

    It holds no state beyond configuration, so one instance serves every dataset
    in a collection.

    Attributes:
        data_folder: Root of the downloaded NSDB tree, used to locate each
            dataset's ``.txt`` file when the record does not carry one.

    Examples:
        Assemble a frame from an already-parsed record:

        >>> from orbitdet.data.nsdb_parser import NSDBParser   # doctest: +SKIP
        >>> record = NSDBParser().parse_file("data/raw/nm0001/nm0001.html")
        >>> frame = NSDBFormatter().format(record)             # doctest: +SKIP
    """

    def __init__(self, data_folder: str | Path = "data/raw") -> None:
        self.data_folder = Path(data_folder)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def data_file_for(self, record: NSDBRecord) -> Path:
        """Locate the data file belonging to a record.

        Args:
            record: The parsed descriptor.

        Returns:
            Path to the ``.txt`` file.  Not checked for existence here; that is
            the caller's job, so the error names the operation that needed it.
        """
        return self.data_folder / record.identifier / f"{record.identifier}.txt"

    def read_data_file(self, record: NSDBRecord, path: str | Path | None = None) -> pd.DataFrame:
        """Read a dataset's whitespace-separated data file with NSDB column names applied.

        Args:
            record: The parsed descriptor, supplying the Format mapping.
            path: Explicit path to the data file.  Defaults to
                :meth:`data_file_for`.

        Returns:
            The raw frame, with the descriptor's declared column names overlaid
            where the declared count matches what the file actually contains.

        Raises:
            FileNotFoundError: If the data file does not exist.
        """
        data_path = Path(path) if path is not None else self.data_file_for(record)
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Data file for {record.identifier} not found at {data_path}. "
                f"Download it first, or pass an explicit path."
            )

        # Read the bytes here rather than letting pandas open the path, so that a
        # legacy single-byte encoding can be handled.  Two Jovian datasets carry
        # CP-1252 punctuation in observatory names - ``Pic\x97du\x97Midi`` - and
        # pandas' default UTF-8 decode aborted on the whole file, losing a
        # dataset to one en-dash in a comment column.  UTF-8 is tried first
        # because it is the strict superset; ``latin-1`` is the fallback because
        # it maps every byte, so it can never raise.
        raw = data_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            logger.info(
                "Data file for %s is not valid UTF-8; decoding as latin-1.",
                record.identifier,
            )
            text = raw.decode("latin-1")

        # Declare the frame's width explicitly rather than letting pandas infer
        # it from the first row.  NSDB files have trailing columns that are
        # omitted when the quantity is unknown, so a long row followed by a short
        # one raises ``Expected N fields`` and loses the entire dataset - for
        # seven Jovian sets, over a handful of rows each.  Given ``names``, pandas
        # pads the short rows with NaN instead, which is exactly what a missing
        # trailing value means.  The width is the largest of the Format section's
        # indices and the longest row actually present, because a few files carry
        # rows wider than their own descriptor says - ``nm0011`` never declares
        # its ninth column and was losing the separation of every observation.
        declared = max(record.format_columns, default=0)
        observed = max((len(line.split()) for line in text.splitlines()), default=0)
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=r"\s+",
            header=None,
            comment="#",
            engine="python",
            names=range(max(declared, observed)),
        )
        apply_format_columns(dataframe, record.format_columns)
        _reject_duplicate_headings(dataframe, record)
        return dataframe

    def format(
        self,
        record: NSDBRecord,
        data: pd.DataFrame | None = None,
        *,
        time_scale: str | None = None,
        data_file: str | Path | None = None,
    ) -> pd.DataFrame:
        """Assemble a standard observation frame for one NSDB dataset.

        Args:
            record: The parsed descriptor.
            data: The raw frame, already column-named.  When omitted the file is
                read via :meth:`read_data_file`.
            time_scale: Override the descriptor's ``time scale``.  Provided
                because the descriptor's label is sometimes wrong and the
                correct value is known at the call site.
            data_file: Override the path recorded in the standard frame's
                provenance, for when the data came from elsewhere.

        Returns:
            A frame satisfying :mod:`orbitdet.preprocessing.schema`.

        Raises:
            ValueError: If the descriptor lacks information the schema requires,
                or its coordinate label is unrecognised.
        """
        if data is None:
            data = self.read_data_file(record)

        parameters = record.parameters or {}
        coordinate_kind, shape = classify_coordinates(parameters.get("coordinates"), data)

        epoch_tdb = self._epoch_tdb(record, data, time_scale)
        first_value, second_value, first_unit, second_unit = self._measurement(
            data, coordinate_kind, shape
        )

        # NSDB marks a quantity it did not record with a sentinel declared in the
        # column heading, and marks the two halves of a relative measurement
        # independently - so many rows carry a position angle with no separation,
        # or the reverse.  Such a row is still a real observation of the half that
        # *was* measured, so it is kept and the missing half is marked with
        # NOT_GIVEN rather than nulled or dropped.  34 datasets store the position
        # angle and the separation on separate lines this way, which is why
        # treating the pair as indivisible would discard most of the corpus.
        observatory_codes = self.resolve_observatory_codes(record, len(epoch_tdb))

        frame_center_label = parameters.get("centre_of_frame") or parameters.get("center_of_frame")
        frame_center = normalize_frame_center(frame_center_label)

        records = self._build_rows(
            record,
            epoch_tdb=epoch_tdb,
            first_value=first_value,
            second_value=second_value,
            observatory_codes=observatory_codes,
            first_unit=first_unit,
            second_unit=second_unit,
            coordinate_kind=coordinate_kind,
            frame_center=frame_center,
            parameters=parameters,
        )
        return build_frame(records)

    # ------------------------------------------------------------------
    # Measurement assembly
    # ------------------------------------------------------------------

    def _epoch_tdb(
        self,
        record: NSDBRecord,
        data: pd.DataFrame,
        time_scale: str | None,
    ) -> np.ndarray:
        """Convert the dataset's observation epochs to TDB seconds since J2000.

        Two steps, deliberately separate: :mod:`time_formatting` decides the
        instant from whatever column layout the file uses, then
        :mod:`time_scales` applies the clock.  Fusing them is what previously
        forced every epoch through an ISO string and capped precision.
        """
        scale = time_scale or (record.parameters or {}).get("time_scale") or "UTC"
        with_epoch = to_seconds_since_j2000(data)
        return to_tdb_seconds_since_j2000(
            with_epoch["epoch_j2000"].to_numpy("float64"),
            scale,
            observatory_code=record.primary_observatory_code,
        )

    def _measurement(
        self,
        data: pd.DataFrame,
        coordinate_kind: CoordinateKind,
        shape: str,
    ) -> tuple[np.ndarray, np.ndarray, str, str]:
        """Extract the two measurement columns in radians, with their source units.

        The unit strings are provenance, not conversions: the values are already
        radians, and the recorded unit says what they were in the file.

        Returns:
            Tuple ``(first_value, second_value, first_unit, second_unit)``.
        """
        if coordinate_kind is CoordinateKind.RA_DEC:
            ra_column, dec_column = find_radec_columns(data)
            # NSDB records RA/Dec in degrees; the schema is radians.
            first_unit = "hour" if ra_column in {"ra"} and _ra_was_assembled(data) else "deg"
            return (
                np.deg2rad(numeric_series(data, ra_column).to_numpy("float64")),
                np.deg2rad(numeric_series(data, dec_column).to_numpy("float64")),
                first_unit,
                "deg",
            )

        if coordinate_kind is CoordinateKind.CARTESIAN_XY:
            x_column, y_column = find_relative_columns(data, coordinate_kind)
            return (
                _scaled_measurement(data, x_column),
                _scaled_measurement(data, y_column),
                _unit_label(x_column),
                _unit_label(y_column),
            )

        # Position angle and separation: the angle is a direction, the separation
        # a distance on the sky.  Both are scaled by their own column's unit.
        angle_column, separation_column = find_relative_columns(data, coordinate_kind)
        return (
            _scaled_measurement(data, angle_column),
            _scaled_measurement(data, separation_column),
            _unit_label(angle_column),
            _unit_label(separation_column),
        )

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _build_rows(
        self,
        record: NSDBRecord,
        *,
        epoch_tdb: np.ndarray,
        first_value: np.ndarray,
        second_value: np.ndarray,
        observatory_codes: list[str],
        first_unit: str,
        second_unit: str,
        coordinate_kind: CoordinateKind,
        frame_center: FrameCenter,
        parameters: dict,
    ) -> list[dict[str, object]]:
        """Build one schema-keyed mapping per observation row."""
        observation_type = normalize_observation_type(record.contents.get("type"))
        relative_to = _relative_to_object(record)
        reference = _coalesce(record.reference)
        reduction = normalize_catalogue(parameters.get("reduction"))

        rows: list[dict[str, object]] = []
        for index in range(len(epoch_tdb)):
            rows.append(
                {
                    "observed_object": _observed_object(record),
                    "observatory_code": observatory_codes[index],
                    "data_source": DataSource.NSDB.value,
                    "source_set_id": record.identifier,
                    "reference": reference,
                    "observation_type": observation_type.value,
                    "coordinate_kind": coordinate_kind.value,
                    "frame_center": frame_center.value,
                    "original_equinox": resolve_epoch_of_equinox(
                        parameters.get("epoch_of_equinox"), dataset=record.identifier
                    ),
                    "original_reference_frame": normalize_reference_frame(
                        parameters.get("reference_frame")
                    ).value,
                    "original_reduction": reduction.value,
                    "receptor_type": normalize_receptor(parameters.get("receptor")).value,
                    "epoch_tdb": epoch_tdb[index],
                    "first_value": first_value[index],
                    "second_value": second_value[index],
                    "relative_to_object": relative_to,
                    "first_unit": first_unit,
                    "second_unit": second_unit,
                }
            )
        return rows

    def resolve_observatory_codes(self, record: NSDBRecord, count: int) -> list[str]:
        """Resolve the observatory code for each row of a dataset.

        For a single-observatory dataset every row gets the one code.  For the
        two multi-observatory datasets the selector column decides, mapped
        through ``telescope_index``; those rows additionally re-read their data
        file, because the mapping needs the column values.

        Args:
            record: The parsed descriptor.
            count: Number of observation rows.

        Returns:
            One normalised observatory code per row.
        """
        primary = record.primary_observatory_code
        if not record.telescope_index:
            code = normalize_observatory_code(primary) if primary is not None else ""
            return [code] * count

        data = self.read_data_file(record)
        codes = resolve_observatory_codes(data, record.observatories, record.telescope_index)
        return [normalize_observatory_code(value) for value in codes.tolist()]

    def group_by_observatory(
        self,
        dataframe: pd.DataFrame,
    ) -> list[tuple[str, pd.DataFrame]]:
        """Split a standard frame by observatory, in order of first appearance.

        Args:
            dataframe: A standard frame produced by :meth:`format`.

        Returns:
            ``(observatory_code, rows)`` pairs, one per distinct observatory.
            A single-observatory dataset yields one pair.
        """
        return group_rows_by_observatory(dataframe, dataframe["observatory_code"])


def _scaled_measurement(dataframe: pd.DataFrame, column_name: str) -> np.ndarray:
    """Read one measured column as radians, marking values that were not given.

    Combining the read, the unit conversion and the sentinel handling in one
    place is deliberate: they are the same operation on the same column, and
    separating them previously let a raw ``999`` reach the frame as though it
    were a position angle.

    Args:
        dataframe: The raw NSDB data frame.
        column_name: The column to read, whose heading carries both the unit and
            any sentinel declaration.

    Returns:
        Values in radians, with :data:`orbitdet.preprocessing.schema.NOT_GIVEN`
        where the column declares that nothing was recorded.  The marker is
        applied *after* the scaling, since scaling would otherwise move it off
        its own value and destroy the one number a consumer recognises.
    """
    marked = _flag_not_given(
        numeric_series(dataframe, column_name).to_numpy("float64"), column_name
    )
    scaled = marked * relative_position_scale(column_name)
    scaled[marked == NOT_GIVEN] = NOT_GIVEN
    return scaled


def _unit_label(column_name: str) -> str:
    """Describe a column's angular unit, for the provenance columns."""
    normalized = normalize_column_name(column_name)
    if "sec of time" in normalized or "seconds of time" in normalized:
        return "sec_of_time"
    if "arcsec" in normalized or "arc second" in normalized or "arcsecond" in normalized:
        return "arcsec"
    if "degree" in normalized or re.search(r"\bdeg\b", normalized):
        return "deg"
    return "unknown"


def _ra_was_assembled(dataframe: pd.DataFrame) -> bool:
    """Whether ``ra`` was built from hour/minute/second components by this module.

    The assembled column is already in degrees and carries no unit in its name,
    so :func:`_unit_label` would report ``"unknown"``.  The components are the
    signal: they had to have been in hours, so the provenance is ``"hour"``.
    """
    return (
        find_column_substring(dataframe, "ascension", "hour", excludes=_TIME_COMPONENT_EXCLUDES)
        is not None
    )


def _observed_object(record: NSDBRecord) -> str:
    """Name the body observed, preferring the satellite over the planet.

    A dataset that tracks Triton is *about* Triton; the planet is the reference
    body, recorded separately in ``relative_to_object``.  Falling back to the
    planet covers datasets whose ``satellites`` block is empty.
    """
    if record.satellites:
        return next(iter(record.satellites))
    if record.planet:
        return str(record.planet.get("name", ""))
    return ""


def _relative_to_object(record: NSDBRecord) -> str | None:
    """Resolve the body a relative measurement is referred to, or ``None``.

    NSDB writes this four ways: the literal ``"absolute"`` (meaning there is no
    reference body), the literal ``"planet"`` (meaning whatever ``planet.name``
    says), a planet name, or a designation such as ``"N1-Triton"``.  The two
    relative factories previously parsed this differently and one of them raised
    ``IndexError`` on the literal ``"absolute"``, which every absolute dataset
    contains.

    Resolving ``"planet"`` here rather than downstream matters: it is the only
    place that knows the descriptor's ``planet.name``, so leaving it unresolved
    would put a body name of ``"planet"`` into the frame.  Every consumer that
    looks the reference body up in the system of bodies would then fail.
    """
    designation = str(((record.parameters or {}).get("relative_to") or {}).get("designation") or "")
    text = designation.strip()
    if not text or text.lower() == "absolute":
        return None
    if text.lower() == "planet":
        name = str((record.planet or {}).get("name", "")).strip()
        return name or None
    if " - " in text:
        return text.split(" - ")[1].strip()
    if "-" in text:
        return text.split("-")[-1].strip()
    return text


def _coalesce(value: object) -> str:
    """Render an optional text field as a string, never ``None``.

    The schema requires ``reference`` to be non-null, and NSDB leaves it out for
    a few datasets, so absence becomes an empty string rather than a null.
    """
    return "" if value is None else str(value)
