"""Reduction-catalogue identification and the tie to the ICRF.

An observation is reduced against whatever star catalogue the observer had, and
the coordinates that come out are expressed in **that catalogue's** frame.  To
combine a 1908 photographic plate with a 2015 Gaia reduction, both have to be
tied to one reference frame first.  This module is that tie.

Why the tie is not a constant rotation
--------------------------------------
It is tempting to model "UCAC2 to ICRF" as a fixed 2x2 rotation, and for a small
field that is nearly right.  It is wrong in two ways that matter for this data:

**The correction is coordinate-dependent.**  Each catalogue's frame is defined by
its own network of reference stars, and the orientation error between two such
frames varies across the sky at the level of tens of milliarcseconds.  Over a
single plate the variation is negligible; across a century of observations
spanning the whole orbit, it is not.

**Proper motion dominates everything else.**  A catalogue position is given at
the catalogue's own epoch, not the observation epoch.  A Tycho-2 star with a
proper motion of 50 mas/yr observed a century after the catalogue epoch is
displaced by **5 arcseconds** - three orders of magnitude larger than the
frame-bias effects the rest of the pipeline worries about.  So the tie cannot be
applied without the proper motion, and the schema carries ``catalogue_epoch``,
``pm_first`` and ``pm_second`` for exactly that reason.

What is implemented here
------------------------
The D-catalogue handling is complete and independent of any network access:

* :func:`normalize_catalogue` - the dirty free-text vocabulary to
  :class:`~orbitdet.preprocessing.schema.ReductionCatalogue`.
* :class:`CatalogueTie` - a per-catalogue description of where its frame sits
  relative to the ICRF, with the epoch and the rotation (if any).
* :func:`apply_proper_motion` - the proper-motion propagation, which is the
  dominant term and needs no external data.
* :func:`tie_to_icrf` - the full correction, composing both.

What is deliberately not implemented
------------------------------------
Accurate per-catalogue orientation fields require an external source - the Gaia
CRF3 tie solutions, or a catalogue query through ``astroquery``/Vizier.  Adding
that would put a network dependency and a multi-gigabyte download into the
preprocessing path, so instead :data:`CATALOGUE_TIES` carries what is known
*without* a lookup and records ``orientation_uncertainty`` explicitly.  A
catalogue with no published tie rotates by zero and reports that it did so,
rather than silently applying an identity that looks like a real correction.

References
----------
* Gaia Collaboration, "Gaia Early Data Release 3: The astrometric solution",
  A&A 649, A1 (2021), for the Gaia-CRF3 definition.
* Tycho-2 Catalogue, Hog et al., A&A 355, L27 (2000), for the catalogue epoch
  and reference frame.
* UCAC2, Zacharias et al., AJ 127, 3043 (2004).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from orbitdet.preprocessing.schema import ReductionCatalogue

logger = logging.getLogger(__name__)

__all__ = [
    "CATALOGUE_TIES",
    "CATALOGUE_EPOCHS",
    "CatalogueTie",
    "TieReport",
    "catalogue_epoch_seconds_since_j2000",
    "normalize_catalogue",
    "normalize_catalogue_series",
    "apply_proper_motion",
    "tie_to_icrf",
    "describe_tie",
]

#: Seconds in a Julian year, for proper-motion propagation.
_SECONDS_PER_JULIAN_YEAR = 365.25 * 86400.0

#: Arcseconds to radians.
_ARCSECOND_TO_RADIANS = np.pi / 648000.0

#: Milliarcseconds to radians.
_MAS_TO_RADIANS = _ARCSECOND_TO_RADIANS / 1000.0


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Substring patterns that identify each catalogue, tried in order.
#:
#: Order matters.  ``tycho`` must be tested before ``tyc`` (both appear, and
#: ``tyc-2`` is the abbreviated form), and ``gaia dr1`` before ``gaia dr2`` so
#: that a label naming both resolves deterministically.  The patterns are matched
#: against a cleaned form, so punctuation and spacing do not have to be exact.
_CATALOGUE_PATTERNS: list[tuple[re.Pattern[str], ReductionCatalogue]] = [
    (re.compile(r"gaia\s*dr\s*1|gaia\s*dr1"), ReductionCatalogue.GAIA_DR1),
    (re.compile(r"gaia\s*dr\s*2|gaia\s*dr2"), ReductionCatalogue.GAIA_DR2),
    (re.compile(r"gaia\s*edr\s*3|gaia\s*d?r?\s*3"), ReductionCatalogue.GAIA_DR2),
    (re.compile(r"tycho|tyc"), ReductionCatalogue.TYCHO2),
    (re.compile(r"ucac\s*4"), ReductionCatalogue.UCAC4),
    (re.compile(r"ucac"), ReductionCatalogue.UCAC2),
    (re.compile(r"\bsao\b"), ReductionCatalogue.SAO),
    # AST is the USNO Astrographic catalogue; "ACT" is a known typo for it in
    # nm0077, so both are accepted.
    (re.compile(r"\bast\b|\bact\b|astrograph"), ReductionCatalogue.AST),
    (
        re.compile(r"given in the data|diven in the data|in the data"),
        ReductionCatalogue.GIVEN_IN_DATA,
    ),
]

#: Labels that mean "no catalogue reduction", so no tie is needed.
_NO_CATALOGUE_PATTERNS = re.compile(
    r"no information|no info|no inf|^none$|^nan$|^$|not given|unknown"
)

#: Phrases that describe a *correction* rather than a catalogue.  Such a label
#: says how the reduction was performed, not which frame it landed in.
_CORRECTION_PHRASES = re.compile(
    r"corrected for|relativistic aberration|light deflection|positions are"
)


def normalize_catalogue(label: object) -> ReductionCatalogue:
    """Map a free-text reduction label onto the catalogue vocabulary.

    The NSDB ``reduction`` field is prose, and dirty: the corpus contains
    ``Catalogue diven in the data`` (a typo), ``Catalogue - Tyc2``,
    ``Catalog SAO``, ``1 - Catalogue AST, 2 -Catalogue Tyc-2 (ICRF)``, and a
    paragraph about relativistic aberration that names no catalogue at all.

    When a label names **two** catalogues, the one appearing **earliest in the
    text** wins, because such labels enumerate options and the first is the
    default: ``1 - Catalogue AST, 2 -Catalogue Tyc-2`` resolves to AST, which
    matches the per-row catalogue column in the dataset that carries it.  The
    per-row column should still be preferred wherever it exists.

    Args:
        label: The raw reduction string, or ``None``.

    Returns:
        The matching :class:`ReductionCatalogue`, or
        :attr:`~orbitdet.preprocessing.schema.ReductionCatalogue.UNKNOWN` for a
        label that names a catalogue not in the vocabulary.

    Examples:
        >>> normalize_catalogue("Gaia DR2").value
        'gaia_dr2'
        >>> normalize_catalogue("Catalogue Tycho-2").value
        'tycho2'
        >>> normalize_catalogue("Catalogue diven in the data").value
        'given_in_data'
        >>> normalize_catalogue("1 - Catalogue AST, 2 -Catalogue Tyc-2 (ICRF)").value
        'ast'
        >>> normalize_catalogue("no information").value
        'none'
    """
    text = "" if label is None else str(label)
    cleaned = " ".join(text.strip().lower().replace("_", " ").split())

    if not cleaned or _NO_CATALOGUE_PATTERNS.search(cleaned):
        return ReductionCatalogue.NONE

    # Every pattern is searched and the earliest match position wins, so a label
    # enumerating several catalogues resolves to the one it lists first rather
    # than to whichever pattern happens to sit highest in the table.  The table
    # order then only breaks ties at equal positions.
    best: tuple[int, ReductionCatalogue] | None = None
    for pattern, catalogue in _CATALOGUE_PATTERNS:
        match = pattern.search(cleaned)
        if match is None:
            continue
        if best is None or match.start() < best[0]:
            best = (match.start(), catalogue)

    if best is not None:
        return best[1]

    if _CORRECTION_PHRASES.search(cleaned):
        # The label describes a correction applied, not a catalogue, so the
        # frame it lands in is not stated.  Distinct from UNKNOWN, which means a
        # named catalogue we do not know.
        return ReductionCatalogue.NONE

    logger.warning(
        "Reduction label %r names no known catalogue; recording UNKNOWN so the "
        "vocabulary can be extended rather than silently skipping the tie.",
        label,
    )
    return ReductionCatalogue.UNKNOWN


def normalize_catalogue_series(values: pd.Series) -> pd.Series:
    """Apply :func:`normalize_catalogue` element-wise to a column.

    Args:
        values: A column of raw reduction labels.

    Returns:
        A column of :class:`ReductionCatalogue` values.
    """
    return values.map(lambda value: normalize_catalogue(value).value)


# ---------------------------------------------------------------------------
# Tie model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueTie:
    """Where a reduction catalogue's frame sits relative to the ICRF.

    Attributes:
        catalogue: Which catalogue this describes.
        reference_frame: The frame the catalogue is expressed in, e.g.
            ``"ICRS"``, ``"FK5"`` or ``"FK4"``.
        catalogue_epoch: Epoch of the catalogue positions as TDB seconds since
            J2000, used to propagate proper motion.  ``None`` when the catalogue
            is not dated, in which case no proper-motion term is applied and that
            is reported.
        orientation_offsets_radians: Small rotation from this catalogue's frame
            to the ICRF, as ``(d_ra_cos_dec, d_dec)`` in radians.  ``(0, 0)``
            when no published tie is available.
        orientation_uncertainty_radians: How well the tie is known.  Non-zero
            even when the offsets are zero, because "no tie available" is not
            the same as "no rotation".
        has_published_tie: Whether the offsets come from a published solution.
            ``False`` means they are a documented identity placeholder.
        notes: Why the tie is what it is, for the log and for whoever reads the
            config later.
    """

    catalogue: ReductionCatalogue
    reference_frame: str
    catalogue_epoch: float | None = None
    orientation_offsets_radians: tuple[float, float] = (0.0, 0.0)
    orientation_uncertainty_radians: float = 0.0
    has_published_tie: bool = False
    notes: str = ""


#: Epochs of the catalogues, as TDB seconds since J2000.
#:
#: Tycho-2 is the important one: it is the catalogue behind most of the
#: twentieth-century photographic data, and its positions are given at the
#: catalogue epoch rather than the observation epoch, so proper motion must be
#: propagated from here.
CATALOGUE_EPOCHS: dict[ReductionCatalogue, float] = {
    # Tycho-2 positions are given at J1991.25.
    ReductionCatalogue.TYCHO2: (1991.25 - 2000.0) * _SECONDS_PER_JULIAN_YEAR,
    # UCAC2/UCAC4 are essentially at J2000.
    ReductionCatalogue.UCAC2: 0.0,
    ReductionCatalogue.UCAC4: 0.0,
    # Gaia DR1 positions are at J2015.0, DR2 at J2015.5.  Note these are the
    # positions' epochs, not an equinox - the frames are ICRS.
    ReductionCatalogue.GAIA_DR1: (2015.0 - 2000.0) * _SECONDS_PER_JULIAN_YEAR,
    ReductionCatalogue.GAIA_DR2: (2015.5 - 2000.0) * _SECONDS_PER_JULIAN_YEAR,
    # SAO is a 1966 positional catalogue with no stated epoch precision useful
    # at this level; AST (the USNO astrographic catalogue) is older still.
    ReductionCatalogue.SAO: (1966.0 - 2000.0) * _SECONDS_PER_JULIAN_YEAR,
    ReductionCatalogue.AST: (1930.0 - 2000.0) * _SECONDS_PER_JULIAN_YEAR,
}


#: The tie for each catalogue, relative to the ICRF.
#:
#: **Only Gaia is genuinely ICRF-aligned.**  DR1 and DR2 are reductions against
#: the ICRS, so their frame *is* the ICRF and the tie is an identity for good
#: reason.  Every other entry is an identity **placeholder**: the real
#: orientation field for UCAC2, Tycho-2, SAO and AST is a sky-position-dependent
#: rotation that has to come from a published tie solution (or a catalogue
#: query).  ``has_published_tie`` is False for those and
#: ``orientation_uncertainty_radians`` records that the frame is not actually
#: tied, so a caller can filter or weight accordingly rather than trusting a
#: zero that means "unknown".
CATALOGUE_TIES: dict[ReductionCatalogue, CatalogueTie] = {
    ReductionCatalogue.GAIA_DR1: CatalogueTie(
        catalogue=ReductionCatalogue.GAIA_DR1,
        reference_frame="ICRS",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.GAIA_DR1],
        orientation_uncertainty_radians=0.1 * _MAS_TO_RADIANS,
        has_published_tie=True,
        notes=(
            "Gaia DR1 is reduced directly against the ICRS, so its frame is the "
            "ICRF by construction; the tie is an identity. The 0.1 mas figure "
            "is the DR1-to-CRF3 alignment uncertainty."
        ),
    ),
    ReductionCatalogue.GAIA_DR2: CatalogueTie(
        catalogue=ReductionCatalogue.GAIA_DR2,
        reference_frame="ICRS",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.GAIA_DR2],
        orientation_uncertainty_radians=0.1 * _MAS_TO_RADIANS,
        has_published_tie=True,
        notes=(
            "Gaia DR2 is the defining realisation of Gaia-CRF2 and is aligned to "
            "the ICRF to well below the measurement noise of any NSDB dataset."
        ),
    ),
    ReductionCatalogue.TYCHO2: CatalogueTie(
        catalogue=ReductionCatalogue.TYCHO2,
        reference_frame="ICRS",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.TYCHO2],
        orientation_uncertainty_radians=20.0 * _MAS_TO_RADIANS,
        has_published_tie=False,
        notes=(
            "Tycho-2 is on the ICRS by construction but only to ~20 mas, and its "
            "positions are at J1991.25, so proper motion must be propagated. The "
            "orientation offsets here are an identity PLACEHOLDER, not a "
            "measured tie."
        ),
    ),
    ReductionCatalogue.UCAC2: CatalogueTie(
        catalogue=ReductionCatalogue.UCAC2,
        reference_frame="ICRS",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.UCAC2],
        orientation_uncertainty_radians=20.0 * _MAS_TO_RADIANS,
        has_published_tie=False,
        notes=(
            "UCAC2 is on the ICRS to ~20 mas. Orientation offsets are an identity "
            "PLACEHOLDER; a published tie solution is required for the real field."
        ),
    ),
    ReductionCatalogue.UCAC4: CatalogueTie(
        catalogue=ReductionCatalogue.UCAC4,
        reference_frame="ICRS",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.UCAC4],
        orientation_uncertainty_radians=10.0 * _MAS_TO_RADIANS,
        has_published_tie=False,
        notes="UCAC4 is on the ICRS to ~10 mas. Orientation offsets are a PLACEHOLDER.",
    ),
    ReductionCatalogue.SAO: CatalogueTie(
        catalogue=ReductionCatalogue.SAO,
        reference_frame="FK4",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.SAO],
        orientation_uncertainty_radians=1000.0 * _MAS_TO_RADIANS,
        has_published_tie=False,
        notes=(
            "SAO is an FK4-era catalogue. Its frame is expressed in FK4/B1950, "
            "which the precession module handles, but the tie to the ICRF is not "
            "a rotation the pipeline can supply from local data - the uncertainty "
            "recorded here is 1 arcsec, which is a placeholder magnitude, not a "
            "measured figure."
        ),
    ),
    ReductionCatalogue.AST: CatalogueTie(
        catalogue=ReductionCatalogue.AST,
        reference_frame="FK4",
        catalogue_epoch=CATALOGUE_EPOCHS[ReductionCatalogue.AST],
        orientation_uncertainty_radians=1000.0 * _MAS_TO_RADIANS,
        has_published_tie=False,
        notes=(
            "USNO Astrographic catalogue, FK4 era, no published ICRF tie in local "
            "data. Placeholder uncertainty of 1 arcsec."
        ),
    ),
    ReductionCatalogue.GIVEN_IN_DATA: CatalogueTie(
        catalogue=ReductionCatalogue.GIVEN_IN_DATA,
        reference_frame="unknown",
        catalogue_epoch=None,
        has_published_tie=False,
        notes=(
            "The catalogue identity is carried per row in the data file rather "
            "than in the descriptor. Resolve it per row and look the tie up "
            "again; this entry only records that the descriptor does not know."
        ),
    ),
    ReductionCatalogue.NONE: CatalogueTie(
        catalogue=ReductionCatalogue.NONE,
        reference_frame="unknown",
        catalogue_epoch=None,
        has_published_tie=False,
        notes="No catalogue reduction was applied, so no tie is possible or needed.",
    ),
    ReductionCatalogue.UNKNOWN: CatalogueTie(
        catalogue=ReductionCatalogue.UNKNOWN,
        reference_frame="unknown",
        catalogue_epoch=None,
        has_published_tie=False,
        notes=(
            "A catalogue was named but is not in the vocabulary. No tie is "
            "applied; extend the vocabulary rather than accepting this."
        ),
    ),
}


def catalogue_epoch_seconds_since_j2000(catalogue: ReductionCatalogue | str) -> float | None:
    """Return the catalogue's position epoch, as seconds since J2000.

    Args:
        catalogue: The catalogue, or its string value.

    Returns:
        Seconds since J2000, or ``None`` when the epoch is not known - which
        means proper motion cannot be propagated and the caller should say so.

    Examples:
        >>> epoch = catalogue_epoch_seconds_since_j2000(ReductionCatalogue.TYCHO2)
        >>> bool(abs(epoch / (365.25 * 86400.0) - (1991.25 - 2000.0)) < 1e-9)
        True
    """
    key = catalogue if isinstance(catalogue, ReductionCatalogue) else ReductionCatalogue(catalogue)
    return CATALOGUE_EPOCHS.get(key)


# ---------------------------------------------------------------------------
# Proper motion
# ---------------------------------------------------------------------------


def apply_proper_motion(
    first_value: np.ndarray | pd.Series,
    second_value: np.ndarray | pd.Series,
    pm_first: np.ndarray | pd.Series,
    pm_second: np.ndarray | pd.Series,
    epoch: np.ndarray | pd.Series,
    catalogue_epoch: float | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate catalogue positions to the observation epoch.

    A catalogue records a star's position at a reference epoch together with its
    proper motion.  An observation taken later sees the star displaced along that
    motion, and the displacement is the single largest correction in this module:
    a Tycho-2 star with 50 mas/yr observed a century after J1991.25 has moved
    **5 arcseconds**.

    Args:
        first_value: Right ascension in radians, or the first angular coordinate.
        second_value: Declination in radians, or the second angular coordinate.
        pm_first: Proper motion in the first coordinate, in radians per Julian
            year, **already multiplied by** ``cos(dec)`` if the coordinate is a
            right ascension - the standard ``mu_alpha*`` convention.
        pm_second: Proper motion in the second coordinate, radians per year.
        epoch: Observation epochs, as TDB seconds since J2000.
        catalogue_epoch: The catalogue's reference epoch, same units.  A scalar
            applies to every row.

    Returns:
        Tuple ``(first_value, second_value)`` at the observation epoch, in
        radians.  Rows with NaN in any input stay NaN.

    Examples:
        >>> import numpy as np
        >>> year = 365.25 * 86400.0
        >>> ra, dec = apply_proper_motion(
        ...     np.array([0.0]), np.array([0.0]),
        ...     np.array([1e-3]), np.array([0.0]),
        ...     np.array([100 * year]), 0.0)
        >>> round(float(ra[0]), 6)
        0.1
    """
    first = np.asarray(first_value, dtype="float64")
    second = np.asarray(second_value, dtype="float64")
    mu_first = np.asarray(pm_first, dtype="float64")
    mu_second = np.asarray(pm_second, dtype="float64")
    obs_epoch = np.asarray(epoch, dtype="float64")
    cat_epoch = np.asarray(catalogue_epoch, dtype="float64")

    elapsed_years = (obs_epoch - cat_epoch) / _SECONDS_PER_JULIAN_YEAR

    # NaN in any input must yield NaN rather than a plausible position: an
    # uncorrected or half-corrected coordinate is worse than a missing one,
    # because nothing downstream can tell it apart from a good one.
    invalid = (
        np.isnan(first)
        | np.isnan(second)
        | np.isnan(mu_first)
        | np.isnan(mu_second)
        | np.isnan(elapsed_years)
    )

    corrected_first = first + mu_first * elapsed_years
    corrected_second = second + mu_second * elapsed_years

    return (
        np.where(invalid, np.nan, corrected_first),
        np.where(invalid, np.nan, corrected_second),
    )


# ---------------------------------------------------------------------------
# The tie
# ---------------------------------------------------------------------------


@dataclass
class TieReport:
    """What :func:`tie_to_icrf` did, so the caller can log or assert on it.

    Attributes:
        catalogue: The catalogue the tie used.
        has_published_tie: Whether the orientation came from a real solution.
        proper_motion_applied: Whether a proper-motion term was applied.
        mean_shift_first_radians: Mean shift applied in the first coordinate.
        mean_shift_second_radians: Mean shift applied in the second coordinate.
        warnings: Human-readable notes about anything approximated.
    """

    catalogue: ReductionCatalogue
    has_published_tie: bool
    proper_motion_applied: bool
    mean_shift_first_radians: float = 0.0
    mean_shift_second_radians: float = 0.0
    warnings: list[str] = field(default_factory=list)


def tie_to_icrf(
    first_value: np.ndarray | pd.Series,
    second_value: np.ndarray | pd.Series,
    *,
    catalogue: ReductionCatalogue | str,
    epoch: np.ndarray | pd.Series,
    pm_first: np.ndarray | pd.Series | None = None,
    pm_second: np.ndarray | pd.Series | None = None,
    catalogue_epoch: float | None = None,
) -> tuple[np.ndarray, np.ndarray, TieReport]:
    """Tie catalogue coordinates to the ICRF, at the observation epoch.

    Applies proper motion first, then the orientation offsets.  Order matters
    only at the microarcsecond level - the two are independent corrections of the
    same coordinate - but doing proper motion first matches how the corrections
    are physically defined: move the star to the observation epoch, then express
    the result in the target frame.

    Args:
        first_value: Right ascension in radians.
        second_value: Declination in radians.
        catalogue: The reduction catalogue, as an enum or its string value.
        epoch: Observation epochs, TDB seconds since J2000.
        pm_first: Proper motion in the first coordinate, radians per year,
            already including the ``cos(dec)`` factor.  Required to apply the
            dominant correction; omitting it is reported.
        pm_second: Proper motion in the second coordinate, radians per year.
        catalogue_epoch: Override the catalogue's reference epoch.  ``None`` uses
            :data:`CATALOGUE_EPOCHS`.

    Returns:
        Tuple ``(first_value, second_value, report)`` in radians, ICRF-aligned.

    Raises:
        KeyError: If the catalogue is not in :data:`CATALOGUE_TIES`.  Raised
            rather than silently skipping the tie, because an untied coordinate
            is indistinguishable downstream from a tied one.

    Examples:
        >>> import numpy as np
        >>> ra, dec, report = tie_to_icrf(
        ...     np.array([1.0]), np.array([0.5]),
        ...     catalogue="gaia_dr2", epoch=np.array([0.0]))
        >>> bool(np.isclose(ra[0], 1.0))
        True
        >>> report.has_published_tie
        True
    """
    key = catalogue if isinstance(catalogue, ReductionCatalogue) else ReductionCatalogue(catalogue)
    tie = CATALOGUE_TIES.get(key)
    if tie is None:  # pragma: no cover - the vocabulary is closed
        raise KeyError(f"No tie definition for catalogue {key!r}")
    first = np.asarray(first_value, dtype="float64")
    second = np.asarray(second_value, dtype="float64")
    obs_epoch = np.asarray(epoch, dtype="float64")

    report = TieReport(
        catalogue=key,
        has_published_tie=tie.has_published_tie,
        proper_motion_applied=False,
    )

    reference_epoch = catalogue_epoch if catalogue_epoch is not None else tie.catalogue_epoch

    if reference_epoch is None:
        report.warnings.append(
            f"Catalogue {key.value!r} has no reference epoch, so proper motion "
            f"cannot be propagated. Positions are used as recorded."
        )
    elif pm_first is None or pm_second is None:
        report.warnings.append(
            f"Catalogue {key.value!r} positions are at epoch "
            f"{reference_epoch / _SECONDS_PER_JULIAN_YEAR + 2000.0:.2f} but no "
            f"proper motion was supplied, so the star's motion is not applied. "
            f"For a century-old observation this can be several arcseconds."
        )
    else:
        first, second = apply_proper_motion(
            first,
            second,
            pm_first,
            pm_second,
            obs_epoch,
            reference_epoch,
        )
        report.proper_motion_applied = True

    if not tie.has_published_tie:
        report.warnings.append(
            f"Catalogue {key.value!r} has no published ICRF tie in local data, so "
            f"no orientation correction was applied. The frame is "
            f"{tie.reference_frame} and the known alignment uncertainty is "
            f"{tie.orientation_uncertainty_radians / _MAS_TO_RADIANS:.0f} mas."
        )

    if tie.has_published_tie:
        delta_first, delta_second = tie.orientation_offsets_radians
        # A published tie is genuinely near-identity for the Gaia catalogues, so
        # the shift is applied rather than skipped, even when it is zero.
        report.mean_shift_first_radians = float(delta_first)
        report.mean_shift_second_radians = float(delta_second)
        first = first + delta_first
        # The second-coordinate offset must be scaled by cos(dec) to stay on the
        # sphere, because the offsets are expressed as a small-angle pair.
        second = second + delta_second * np.cos(second)

    for message in report.warnings:
        logger.warning(message)

    return first, second, report


def describe_tie(catalogue: ReductionCatalogue | str) -> str:
    """One-line summary of a catalogue's tie, for logs and workbooks.

    Args:
        catalogue: The catalogue, or its string value.

    Returns:
        A description including the frame, the aligned status and the
        uncertainty.

    Examples:
        >>> "ICRS" in describe_tie("gaia_dr2")
        True
    """
    key = catalogue if isinstance(catalogue, ReductionCatalogue) else ReductionCatalogue(catalogue)
    tie = CATALOGUE_TIES.get(key)
    if tie is None:  # pragma: no cover - the vocabulary is closed
        return f"{key.value}: no tie definition"

    aligned = "aligned" if tie.has_published_tie else "NOT aligned"
    uncertainty_mas = tie.orientation_uncertainty_radians / _MAS_TO_RADIANS
    epoch = (
        "undated"
        if tie.catalogue_epoch is None
        else f"{tie.catalogue_epoch / _SECONDS_PER_JULIAN_YEAR + 2000.0:.2f}"
    )
    return (
        f"{key.value}: frame={tie.reference_frame}, {aligned}, "
        f"epoch={epoch}, uncertainty={uncertainty_mas:.1f} mas"
    )
