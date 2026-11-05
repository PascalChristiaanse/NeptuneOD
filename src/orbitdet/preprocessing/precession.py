"""Precession of celestial coordinates between equinoxes and epochs.

An observation is reduced against whatever equator and equinox the observer's
star catalogue used, and that is rarely J2000.  NSDB alone contains J2000,
J2015.0, B1950, the mean equinox of date, and two ad-hoc Besselian epochs
(``1899.0``, ``1905.0``) that appear nowhere in any standard frame library.  This
module brings them all onto a single orientation, so the rest of the pipeline
never has to know which equinox a measurement came from.

Design
------
Everything here works on **plain numpy arrays**, never on dataframes.  The
functions are pure, take radians, and return radians.  A dataframe may *contain*
the call, but the conversion itself never reaches into one: this keeps the
functions testable with hand-computed values and keeps the standard observation
frame free of conversion logic.

Two frame families are involved, and they are genuinely different:

``FK5`` (Julian)
    Equinoxes named ``J2000.0``, ``J2015.0``.  These are mean equator and
    equinox, precessed with the IAU 1976/2006 precession models.  J2000 is what
    SPICE calls ``J2000`` and what the pipeline ultimately works in.

``FK4`` (Besselian)
    Equinoxes named ``B1950``, ``B1905``, ``1899.0``.  These use the older
    Newcomb precession with a different epoch definition (Besselian years begin
    at the instant the Sun's mean longitude is 280 degrees, not on January 1).
    Astropy handles the epoch arithmetic and the E-terms of aberration.

Astropy is used rather than SPICE for the historical epochs, because SPICE
recognises only ``J2000``, ``B1950`` and ``ECLIPJ2000``: it cannot represent
mean-of-date, true-of-date, ``J2015.0`` or an arbitrary Besselian epoch.

References
----------
* IAU 1976 precession (Lieske et al.), as implemented in ERFA ``pmat76`` and
  Astropy's ``FK5`` frame.
* IAU 2006 precession (Capitaine et al.), as implemented in Astropy's ``TETE``
  frame for true equator and equinox of date.
* Standish, E. M. (1982), "Conversion of positions and proper motions from B1950
  to the IAU system", for the FK4/FK5 relationship and the E-terms.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from astropy import units as u
from astropy.coordinates import FK4, FK5, TETE, FK4NoETerms, SkyCoord
from astropy.time import Time
from tudatpy.interface import spice

from orbitdet.preprocessing.time_scales import SECONDS_PER_DAY

logger = logging.getLogger(__name__)

__all__ = [
    "EquinoxSystem",
    "Fk4Convention",
    "Equinox",
    "JD_J2000",
    "SPICE_FRAMES",
    "is_spice_frame",
    "rotation_matrix",
    "parse_equinox",
    "equinox_system",
    "mean_of_date_equinox",
    "precess_radec",
    "radec_to_unit_vectors",
    "unit_vectors_to_radec",
]


#: Julian date of the J2000.0 epoch.
JD_J2000 = 2451545.0


# ---------------------------------------------------------------------------
# Equinox vocabulary
# ---------------------------------------------------------------------------


class EquinoxSystem(StrEnum):
    """Which precession model a coordinate frame belongs to.

    ``FK5`` and ``FK4`` use different precession models *and* different epoch
    definitions, so a Besselian epoch cannot simply be treated as a Julian one
    with a shifted origin.  For epochs around 1900 the difference is roughly
    20 arcseconds - an order of magnitude larger than the astrometric corrections
    the pipeline otherwise worries about.
    """

    FK5 = "fk5"
    FK4 = "fk4"
    OF_DATE = "of_date"
    TRUE_OF_DATE = "true_of_date"


class Fk4Convention(StrEnum):
    """Which definition of the FK4/B1950 frame to use for Besselian equinoxes.

    SPICE and Astropy disagree about FK4, and the difference is not negligible:
    they differ by a **constant 0.525 arcsec rotation about z** (verified over
    the whole sky), which is larger than the astrometric corrections applied
    elsewhere in the pipeline.

    ``SPICE``
        Use SPICE's ``B1950`` frame: a fixed rotation matrix that includes the
        E-terms of aberration.  This is what the existing Voyager and NSDB
        reductions were built and fitted against, so it is the default.
    ``ASTROPY``
        Use Astropy's ``FK4``, which applies the E-terms as a position-dependent
        correction.  Physically better justified, but shifting to it moves every
        B1950 observation by half an arcsecond.
    ``ASTROPY_NO_ETERMS``
        Use Astropy's ``FK4NoETerms``, the pure precession without E-terms.
        Provided for completeness and for comparison against catalogues that
        already have E-terms removed.

    Only Besselian 1950.0 is affected; every other equinox has no SPICE frame and
    always goes through Astropy.
    """

    SPICE = "spice"
    ASTROPY = "astropy"
    ASTROPY_NO_ETERMS = "astropy_no_eterms"


@dataclass(frozen=True)
class Equinox:
    """A parsed equinox specification.

    Attributes:
        system: Which precession model applies.
        label: The original label, preserved for logging and diagnostics.
        astropy_time: The equinox as an Astropy :class:`~astropy.time.Time`, for
            the FK4/FK5 families.  ``None`` for the of-date families, whose
            equinox depends on the observation epoch rather than being fixed.
        is_besselian: Whether the equinox is a Besselian (FK4) epoch.
        recognised: Whether the label was actually understood.  ``False`` means
            the label carried no information or was unknown and the parser fell
            back to J2000, so the *system* is only a placeholder.  Callers that
            need to distinguish a real J2000 equinox from a fallback - notably
            frame classification - must check this.
    """

    system: EquinoxSystem
    label: str
    astropy_time: Time | None = None
    is_besselian: bool = False
    recognised: bool = True

    @property
    def is_of_date(self) -> bool:
        """Whether this equinox is defined per-observation rather than fixed."""
        return self.system in (EquinoxSystem.OF_DATE, EquinoxSystem.TRUE_OF_DATE)

    @property
    def is_true_of_date(self) -> bool:
        """Whether nutation as well as precession must be applied."""
        return self.system is EquinoxSystem.TRUE_OF_DATE


#: Labels that mean "the mean equinox of the observation date".
_OF_DATE_LABELS = frozenset(
    {
        "date",
        "of date",
        "ofdate",
        "mean of date",
        "meanofdate",
        "mod",
        "mean equinox of date",
        "equinox of date",
    }
)

#: Labels that mean "the true equinox of the observation date" (precession *and*
#: nutation).
_TRUE_OF_DATE_LABELS = frozenset(
    {
        "true of date",
        "trueofdate",
        "tod",
        "true equinox of date",
        "apparent",
        "of apparent date",
    }
)

#: Labels that can be safely treated as J2000.
_J2000_LABELS = frozenset({"j2000", "j2000 0", "j2000 00", "2000.0", "2000"})

#: Labels carrying no usable information.
_UNKNOWN_LABELS = frozenset({"", "no information", "no inf", "unknown", "none", "nan"})

#: ``J2015`` style Julian equinox.  The ``J`` is mandatory: without it, a bare
#: year would be ambiguous, and NSDB uses bare years for Besselian epochs.
_JULIAN_PATTERN = re.compile(r"^j\s*(\d{4}(?:\.\d+)?)$")

#: ``B1950`` style Besselian equinox, written with an explicit B.
_BESSELIAN_PATTERN = re.compile(r"^b\s*(\d{4}(?:\.\d+)?)$")

#: A bare four-digit year such as ``1905.0``.  These are Besselian in the NSDB
#: catalogue (they accompany photographic reductions from the FK4 era), even
#: though the label omits the B.
_BARE_YEAR_PATTERN = re.compile(r"^(\d{4}(?:\.\d+)?)$")


def _normalise_equinox_label(label: str) -> str:
    """Lower-case and collapse punctuation in an equinox label."""
    text = str(label).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_equinox(label: str) -> Equinox:
    """Interpret an epoch-of-equinox label from an observation source.

    Understands the labels that occur in real NSDB metadata: the two SPICE
    frames (``J2000``, ``B1950``), arbitrary Julian and Besselian epochs
    (``J2015.0``, ``B1905``), the bare-year forms the catalogue uses for
    Besselian epochs (``1905.0``, ``1899.0``), and the of-date families
    (``date``, ``true of date``).

    An unparseable label resolves to :attr:`EquinoxSystem.FK5` at J2000 with a
    warning, so ingestion continues and the assumption is visible in the log
    rather than raising mid-run.

    Args:
        label: The raw equinox string.

    Returns:
        The parsed :class:`Equinox`.

    Examples:
        >>> parse_equinox("date").system is EquinoxSystem.OF_DATE
        True
        >>> parse_equinox("B1950").is_besselian
        True
        >>> parse_equinox("1905.0").is_besselian
        True
    """
    normalised = _normalise_equinox_label(label)

    if normalised in _TRUE_OF_DATE_LABELS:
        return Equinox(EquinoxSystem.TRUE_OF_DATE, str(label))

    if normalised in _OF_DATE_LABELS:
        return Equinox(EquinoxSystem.OF_DATE, str(label))

    if normalised in _UNKNOWN_LABELS:
        logger.warning("Epoch of equinox %r carries no information; assuming J2000.", label)
        return Equinox(EquinoxSystem.FK5, str(label), Time("J2000", scale="tt"), recognised=False)

    if normalised in _J2000_LABELS:
        return Equinox(EquinoxSystem.FK5, str(label), Time("J2000", scale="tt"))

    # Besselian with an explicit B.
    match = _BESSELIAN_PATTERN.match(normalised)
    if match:
        year = float(match.group(1))
        return Equinox(
            EquinoxSystem.FK4, str(label), FK4(equinox=f"B{year}").equinox, is_besselian=True
        )

    # A bare year.  NSDB uses these for Besselian epochs: they accompany the
    # photographic reductions of the FK4 era, and Standish (1982) places them on
    # the Besselian scale.  Checked before the Julian form, whose J is mandatory
    # precisely so that this case cannot be swallowed by it.
    match = _BARE_YEAR_PATTERN.match(normalised)
    if match:
        year = float(match.group(1))
        logger.info("Bare year %r treated as the Besselian equinox B%.1f.", label, year)
        return Equinox(
            EquinoxSystem.FK4, str(label), FK4(equinox=f"B{year}").equinox, is_besselian=True
        )

    # Julian with an explicit J.
    match = _JULIAN_PATTERN.match(normalised)
    if match:
        year = float(match.group(1))
        if year == 2000.0:
            return Equinox(EquinoxSystem.FK5, str(label), Time("J2000", scale="tt"))
        return Equinox(EquinoxSystem.FK5, str(label), FK5(equinox=f"J{year}").equinox)

    logger.warning(
        "Unrecognised epoch of equinox %r; assuming J2000. Add it to the "
        "vocabulary in orbitdet.preprocessing.precession if it denotes "
        "something else.",
        label,
    )
    return Equinox(EquinoxSystem.FK5, str(label), Time("J2000", scale="tt"), recognised=False)


def equinox_system(label: str) -> EquinoxSystem:
    """Return just the precession-model family for an equinox label.

    Args:
        label: The raw equinox string.

    Returns:
        The :class:`EquinoxSystem` the label belongs to.
    """
    return parse_equinox(label).system


def mean_of_date_equinox(epoch_tdb: float) -> Time:
    """Return the mean equinox of the date at a given epoch.

    Args:
        epoch_tdb: Epoch as TDB seconds since J2000.

    Returns:
        The equinox as an Astropy :class:`~astropy.time.Time`.
    """
    return Time(_seconds_since_j2000_to_jd(epoch_tdb), format="jd", scale="tdb")


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def radec_to_unit_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Convert right ascension and declination to unit vectors.

    Args:
        ra: Right ascension in radians.
        dec: Declination in radians.

    Returns:
        An ``(n, 3)`` array of unit vectors in the ``(x, y, z)`` convention used
        by both SPICE and TudatPy: ``x`` towards the equinox, ``z`` towards the
        pole.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    return np.column_stack(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    )


def unit_vectors_to_radec(
    vectors: np.ndarray, *, wrap: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit vectors to right ascension and declination.

    Args:
        vectors: An ``(n, 3)`` array of vectors.  Need not be normalised.
        wrap: If ``True``, wrap right ascension into ``[0, 2*pi)``.  If ``False``
            (default), return it in the principal interval ``[-pi, pi]``, which
            is the convention used inside this module.

    Returns:
        Tuple ``(ra, dec)`` in radians.
    """
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms == 0.0, 1.0, norms)
    unit = vectors / safe[:, None]

    ra = np.arctan2(unit[:, 1], unit[:, 0])
    dec = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))

    if wrap:
        ra = np.mod(ra, 2.0 * np.pi)

    ra = np.where(norms == 0.0, np.nan, ra)
    dec = np.where(norms == 0.0, np.nan, dec)
    return ra, dec


# ---------------------------------------------------------------------------
# Precession
# ---------------------------------------------------------------------------


def precess_radec(
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    from_equinox: str | Equinox,
    to_equinox: str | Equinox = "J2000",
    epoch_tdb: np.ndarray | float | None = None,
    wrap: bool = False,
    fk4_convention: Fk4Convention | str = Fk4Convention.SPICE,
) -> tuple[np.ndarray, np.ndarray]:
    """Precess right ascension and declination between equinoxes.

    Handles the four families that occur in practice:

    * **FK5 to FK5** (e.g. ``J2015.0`` to ``J2000``) - a rotation only.
    * **FK4 to FK5** (e.g. ``B1950`` to ``J2000``) - a rotation plus the frame
      bias, and optionally the E-terms of aberration.
    * **Of-date to J2000** - the equinox depends on the observation epoch, so
      ``epoch_tdb`` is required.
    * **True of date** - as above, but with nutation included.

    The observation epoch is required whenever *either* frame is of-date, and is
    ignored otherwise, so callers can always pass it.

    Args:
        ra: Right ascension in radians, in ``from_equinox``.
        dec: Declination in radians, in ``from_equinox``.
        from_equinox: The equinox of the input coordinates, as a label or a
            parsed :class:`Equinox`.
        to_equinox: The equinox to convert to.  Defaults to J2000.
        epoch_tdb: Observation epochs as TDB seconds since J2000.  Required when
            either equinox is of-date.  May be a scalar or an array matching
            ``ra``.
        wrap: If ``True``, wrap the output right ascension into ``[0, 2*pi)``.
        fk4_convention: Which B1950/FK4 definition to use.  See
            :class:`Fk4Convention`; the default preserves the behaviour the
            existing pipeline was fitted with.

    Returns:
        Tuple ``(ra, dec)`` in radians, in ``to_equinox``.  Right ascension is
        in ``[-pi, pi]`` unless ``wrap`` is set.

    Raises:
        ValueError: If an of-date equinox is requested without ``epoch_tdb``, or
            if ``epoch_tdb`` has a length that does not match ``ra``.

    Notes:
        The FK4/FK5 transformations are **not exactly invertible**.  The E-terms
        of aberration and the frame-bias rotation are designed as one-way
        corrections between catalogue systems, not as a group, so a
        ``B1950 -> 1905.0 -> B1950`` round trip returns to within about
        0.001 arcsec rather than to machine precision.  Inverting an observation
        is never the intent here - the pipeline only ever converts *towards*
        J2000 - so this is a documented property rather than a defect.

    Examples:
        Precessing J2000 to J2000 is the identity:

        >>> import numpy as np
        >>> ra, dec = precess_radec(np.array([1.0]), np.array([0.5]),
        ...                         from_equinox="J2000", to_equinox="J2000")
        >>> bool(np.isclose(ra[0], 1.0) and np.isclose(dec[0], 0.5))
        True
    """
    source = from_equinox if isinstance(from_equinox, Equinox) else parse_equinox(from_equinox)
    target = to_equinox if isinstance(to_equinox, Equinox) else parse_equinox(to_equinox)

    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))

    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec must have matching shapes, got {ra.shape} and {dec.shape}")

    if _same_equinox(source, target):
        return ra.copy(), dec.copy()

    epochs = _resolve_epochs(epoch_tdb, ra, source, target)

    # The FK4-to-FK5 rotation is the one place where SPICE and Astropy use
    # genuinely different definitions, so let the caller pick and default to the
    # one the existing data was reduced with.
    convention = Fk4Convention(fk4_convention)
    if convention is Fk4Convention.SPICE and _both_spice_frames(source, target):
        return _precess_via_spice(ra, dec, source, target, wrap=wrap)

    return _precess_via_astropy(ra, dec, source, target, epochs, convention, wrap=wrap)


def _both_spice_frames(source: Equinox, target: Equinox) -> bool:
    """Whether both equinoxes map onto frames SPICE can rotate between.

    Only the two fixed FK4/FK5 frames qualify: ``B1950`` and ``J2000``.  Of-date
    equinoxes and arbitrary epochs cannot be expressed as SPICE frames.
    """
    return _spice_frame_name(source) is not None and _spice_frame_name(target) is not None


def _spice_frame_name(equinox: Equinox) -> str | None:
    """Return the SPICE frame name for an equinox, or ``None`` if it has none.

    A fixed equinox maps to ``B1950`` or ``J2000`` only when it is Besselian
    1950.0 or Julian 2000.0 respectively; any other epoch has no SPICE frame.
    """
    if equinox.is_of_date or equinox.astropy_time is None:
        return None

    if equinox.system is EquinoxSystem.FK4 and np.isclose(
        equinox.astropy_time.jyear, 1950.0, atol=1e-6
    ):
        return "B1950"
    if equinox.system is EquinoxSystem.FK5 and np.isclose(
        equinox.astropy_time.jyear, 2000.0, atol=1e-6
    ):
        return "J2000"
    return None


def _precess_via_spice(
    ra: np.ndarray,
    dec: np.ndarray,
    source: Equinox,
    target: Equinox,
    *,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate between two fixed frames using SPICE.

    SPICE's ``B1950`` frame is a fixed rotation matrix that includes the E-terms
    of aberration, and differs from Astropy's ``FK4`` by a constant 0.525 arcsec
    rotation about ``z``.  That is a real convention difference, not a bug, and
    it is larger than the astrometric corrections applied elsewhere in the
    pipeline - so which one is used has to be an explicit choice.

    SPICE's version is the default because it is what the existing Voyager and
    NSDB reductions were built against, and silently changing it would shift
    every B1950 observation by half an arcsecond.
    """
    source_name = _spice_frame_name(source)
    target_name = _spice_frame_name(target)
    assert source_name is not None and target_name is not None

    vectors = radec_to_unit_vectors(ra, dec)
    matrix = rotation_matrix(source_name, target_name, 0.0)
    rotated = vectors @ matrix.T
    return unit_vectors_to_radec(rotated, wrap=wrap)


def _precess_via_astropy(
    ra: np.ndarray,
    dec: np.ndarray,
    source: Equinox,
    target: Equinox,
    epochs: np.ndarray | None,
    convention: Fk4Convention,
    *,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform through Astropy, supporting every equinox family."""
    # Decide which rows are usable *before* building the frames.  An of-date
    # frame is constructed from the epoch array itself, and Astropy rejects
    # non-finite times outright, so one NaN epoch in a long series would
    # otherwise abort the whole conversion instead of blanking that one row.
    valid = np.isfinite(ra) & np.isfinite(dec)
    if epochs is not None:
        valid &= np.isfinite(epochs)

    out_ra = np.full_like(ra, np.nan)
    out_dec = np.full_like(dec, np.nan)
    if not valid.any():
        return out_ra, out_dec

    # A per-epoch frame only needs the epochs that survived, so it is built from
    # the filtered subset and then used as-is: it already has one entry per
    # valid row, matching the filtered coordinates below.  A fixed frame ignores
    # the epochs entirely and is likewise used whole.
    frame_epochs = epochs[valid] if epochs is not None else None
    input_frame = _astropy_frame(source, frame_epochs, convention)
    output_frame = _astropy_frame(target, frame_epochs, convention)

    coordinates = SkyCoord(
        ra[valid] * u.rad,
        dec[valid] * u.rad,
        frame=input_frame,
    )
    converted = coordinates.transform_to(output_frame)
    out_ra[valid] = converted.ra.to(u.rad).value
    out_dec[valid] = converted.dec.to(u.rad).value

    # Astropy reports right ascension in [0, 2*pi); rewrap to the principal
    # interval the rest of the module uses, so a conversion never changes the
    # wrap convention behind the caller's back.
    if wrap:
        out_ra = np.mod(out_ra, 2.0 * np.pi)
    else:
        out_ra = np.remainder(out_ra + np.pi, 2.0 * np.pi) - np.pi

    return out_ra, out_dec


def _per_epoch(equinox: Equinox) -> bool:
    """Whether this equinox yields a different frame for each observation."""
    return equinox.is_of_date


def _same_equinox(a: Equinox, b: Equinox) -> bool:
    """Whether two parsed equinoxes denote the same orientation."""
    if a.system != b.system:
        return False
    if a.is_of_date:
        # Both of-date in the same family: identical by definition.
        return True
    if a.astropy_time is None or b.astropy_time is None:
        return False
    return bool(np.isclose(a.astropy_time.jd, b.astropy_time.jd, atol=1e-9))


def _resolve_epochs(
    epoch_tdb: np.ndarray | float | None,
    ra: np.ndarray,
    source: Equinox,
    target: Equinox,
) -> np.ndarray | None:
    """Validate and broadcast the observation epochs when they are needed."""
    needed = source.is_of_date or target.is_of_date
    if not needed:
        return None

    if epoch_tdb is None:
        raise ValueError(
            f"An observation epoch is required to precess to or from "
            f"{source.label!r} -> {target.label!r}: the equinox of date depends "
            f"on when the observation was made. Pass 'epoch_tdb'."
        )

    epochs = np.atleast_1d(np.asarray(epoch_tdb, dtype=float))
    if epochs.size == 1:
        epochs = np.full(ra.shape, float(epochs[0]))
    elif epochs.shape != ra.shape:
        raise ValueError(
            f"epoch_tdb has shape {epochs.shape} but ra has shape {ra.shape}; "
            f"they must match (or epoch_tdb must be scalar)"
        )
    return epochs


def _astropy_frame(
    equinox: Equinox,
    epochs: np.ndarray | None,
    convention: Fk4Convention = Fk4Convention.ASTROPY,
):
    """Build the Astropy frame for an equinox, per-epoch when needed."""
    if equinox.system is EquinoxSystem.TRUE_OF_DATE:
        assert epochs is not None
        return TETE(obstime=_jd_array_to_time(epochs))

    if equinox.system is EquinoxSystem.OF_DATE:
        assert epochs is not None
        return FK5(equinox=_jd_array_to_time(epochs))

    assert equinox.astropy_time is not None
    if equinox.system is EquinoxSystem.FK4:
        return _fk4_frame(equinox.astropy_time, convention)
    return FK5(equinox=equinox.astropy_time)


def _fk4_frame(
    equinox_time: Time,
    convention: Fk4Convention = Fk4Convention.ASTROPY,
):
    """Build an FK4 family frame for an astropy time.

    Astropy is used here for every convention except :attr:`Fk4Convention.SPICE`,
    which is short-circuited earlier by :func:`_precess_via_spice` and so never
    reaches this function.
    """
    if convention is Fk4Convention.ASTROPY_NO_ETERMS:
        return FK4NoETerms(equinox=equinox_time)
    return FK4(equinox=equinox_time)


def _jd_array_to_time(epochs: np.ndarray) -> Time:
    """Convert an array of TDB seconds since J2000 to an array of Astropy times."""
    return Time(_seconds_since_j2000_to_jd(epochs), format="jd", scale="tdb")


def _seconds_since_j2000_to_jd(epoch_tdb: np.ndarray | float) -> np.ndarray:
    """Convert TDB seconds since J2000 to Julian date (TDB)."""
    return np.asarray(epoch_tdb, dtype=float) / SECONDS_PER_DAY + JD_J2000


# ---------------------------------------------------------------------------
# SPICE frames
# ---------------------------------------------------------------------------

#: Frames SPICE recognises natively, verified against the installed CSPICE.
#: Everything else must be routed through Astropy.
SPICE_FRAMES = frozenset({"J2000", "B1950", "ECLIPJ2000", "ECLIPB1950"})


def is_spice_frame(frame: str) -> bool:
    """Whether SPICE can use a frame name directly.

    Args:
        frame: The frame name.

    Returns:
        ``True`` for the frames CSPICE knows, ``False`` otherwise.

    Examples:
        >>> is_spice_frame("J2000")
        True
        >>> is_spice_frame("DATE")
        False
    """
    return str(frame).strip().upper() in SPICE_FRAMES


def rotation_matrix(
    input_frame: str,
    output_frame: str,
    epoch_tdb: float,
) -> np.ndarray:
    """Return the rotation matrix between two SPICE frames.

    Args:
        input_frame: Source frame name.
        output_frame: Target frame name.
        epoch_tdb: Epoch as TDB seconds since J2000.

    Returns:
        A ``(3, 3)`` rotation matrix taking a vector from ``input_frame`` to
        ``output_frame``.

    Raises:
        ValueError: If either frame is not a SPICE frame.
        RuntimeError: If SPICE cannot build the rotation.
    """
    if not is_spice_frame(input_frame):
        raise ValueError(
            f"{input_frame!r} is not a SPICE frame. SPICE knows only "
            f"{sorted(SPICE_FRAMES)}; route anything else through "
            f"orbitdet.preprocessing.precession."
        )
    if not is_spice_frame(output_frame):
        raise ValueError(
            f"{output_frame!r} is not a SPICE frame. SPICE knows only "
            f"{sorted(SPICE_FRAMES)}; route anything else through "
            f"orbitdet.preprocessing.precession."
        )

    return np.asarray(
        spice.compute_rotation_matrix_between_frames(input_frame, output_frame, float(epoch_tdb)),
        dtype=float,
    )
