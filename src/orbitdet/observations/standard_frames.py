"""Observation factories that consume a standard observation frame.

This module replaces the per-modality NSDB factories (``absolute_ccd_nsdb``,
``relative_ccd_nsdb``, ``relative_xy_nsdb``) with four factories keyed on the two
normalised facts that actually select an observation model.

Why the key changed
-------------------
The old factories were registered under fused NSDB type strings that encode three
independent things at once - whether the measurement is absolute or relative,
which coordinates it holds, and which receptor recorded it - e.g.
``relative_X,_Y_photographic_nsdb``.  Only the first two affect the *model*.  The
consequences were concrete:

* 15+ distinct strings in the corpus, typos among them (``photograpihc`` twice,
  ``no inf`` once), each needing an exact-match registration;
* **51 of 82** committed configs named a type with no factory at all, so they
  could not load;
* the two commented-out registrations in ``relative_ccd_nsdb`` existed purely to
  work around this.

Keying on ``(observation_type, coordinate_kind)`` collapses that onto four
factories.  The fused names stay registered as aliases, so no committed config
has to be rewritten.

What these factories no longer do
---------------------------------
Every factory used to preprocess as well as build: read a raw file, infer columns
from prose headings, convert units and time, and reorient coordinates.  All of
that now happens in ``orbitdet.data`` (source-specific formatting) and
``orbitdet.preprocessing`` (computation), so a factory receives a frame that
already satisfies the standard schema and whose coordinates are already in the
requested orientation.  What remains is the part that genuinely needs Tudat: link
ends, models, and splitting by observatory.

References:
    :mod:`orbitdet.data.nsdb_formatter` for the formatting stage,
    :mod:`orbitdet.preprocessing.schema` for the frame contract.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observable_models_setup as obs_model_setup
import tudatpy.estimation.observations as obs
import tudatpy.estimation.observations_setup as obs_setup
from omegaconf import DictConfig

from orbitdet.data.nsdb_formatter import normalize_frame_center
from orbitdet.preprocessing.schema import CoordinateKind, FrameCenter, ObservationType

from .helpers import add_observatory_to_SOB, normalize_observatory_code
from .registry import dispatch_key, register_dataset_factory

logger = logging.getLogger(__name__)

#: TudatPy in this environment binds ``relative_angular_position_type`` to
#: ``angular_position_type``, so relative-angular sets would be tagged as
#: absolute.  The underlying C++ enum value is used directly instead.
RELATIVE_ANGULAR_POSITION_TYPE = obs_model_setup.model_settings.ObservableType(9)

#: The one column a factory requires beyond the schema: the receiver's epoch.
EPOCH_COLUMN = "epoch_tdb"

#: Columns the ingestion path adds for frames whose coordinates need geometry to
#: interpret.  ``convert_standard_frame`` produces them.
RA_COLUMN = "ra_rad"
DEC_COLUMN = "dec_rad"


# ---------------------------------------------------------------------------
# Shared frame handling
# ---------------------------------------------------------------------------


def require_standard_frame(dataframe: pd.DataFrame, dataset_cfg: DictConfig) -> pd.DataFrame:
    """Check that a dataframe is a usable standard frame, and return it.

    The check is deliberately shallow: :func:`orbitdet.preprocessing.schema.verify`
    is the authority on validity, and running it here would duplicate it.  What is
    checked is only what a factory needs in order to build link ends at all.

    Args:
        dataframe: The frame handed to the factory.
        dataset_cfg: Config, used to name the dataset in the error.

    Returns:
        The same frame.

    Raises:
        TypeError: If ``dataframe`` is not a DataFrame.
        ValueError: If a required column is absent.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Dataset {dataset_cfg.get('identifier', '?')}: factories consume a standard "
            f"frame, got {type(dataframe).__name__}."
        )
    missing = [name for name in ("coordinate_kind", EPOCH_COLUMN) if name not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Dataset {dataset_cfg.get('identifier', '?')}: frame is missing "
            f"{', '.join(missing)}. Run it through "
            f"orbitdet.preprocessing.schema.verify at the loading boundary."
        )
    return dataframe


def coordinate_kind_of(dataframe: pd.DataFrame) -> CoordinateKind:
    """Read the single coordinate kind a frame carries.

    Raises:
        ValueError: If the frame mixes kinds, which would need splitting first.
    """
    kinds = set(dataframe["coordinate_kind"].dropna().astype(str).unique())
    if len(kinds) != 1:
        raise ValueError(f"Expected one coordinate kind per frame, got {sorted(kinds) or 'none'}.")
    return CoordinateKind(kinds.pop())


def observatory_groups(
    dataframe: pd.DataFrame,
    dataset_cfg: DictConfig,
) -> list[tuple[str, pd.DataFrame]]:
    """Split a standard frame into one sub-frame per observing station.

    A Tudat observation set has exactly one receiver, so a dataset that observed
    from several stations - only ``nm0019`` and ``nm0083`` in the corpus - becomes
    several sets.

    Args:
        dataframe: The standard frame, carrying ``observatory_code``.
        dataset_cfg: Config, used only to name the dataset in errors.

    Returns:
        ``(station_name, rows)`` pairs, in order of first appearance.
    """
    if "observatory_code" not in dataframe.columns:
        raise ValueError(
            f"Dataset {dataset_cfg.get('identifier', '?')}: frame has no "
            f"'observatory_code' column, so the receiver cannot be identified."
        )

    clusters: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for code, cluster in dataframe.groupby("observatory_code", sort=False):
        station = normalize_observatory_code(code)
        if station in seen:
            continue
        seen.add(station)
        clusters.append((station, cluster))
    return clusters


def build_link_ends(
    dataset_cfg: DictConfig,
    station_name: str,
    *,
    relative_to: str | None = None,
    frame_center: FrameCenter | str | None = None,
) -> dict:
    """Assemble the link ends for one observation set.

    Absolute observations need a transmitter and a receiver.  Relative ones
    additionally need ``transmitter2`` - the body the measurement is referred to -
    because a relative observable is built from two directions at the receiver.

    The receiver is chosen by the frame centre, and the three cases are genuinely
    different kinds of link end rather than variations on one:

    ``topocentric``
        A ground station, so a *reference point* on Earth.
    ``spacecraft``
        The observer is itself a body (Voyager 2), so a body origin.  Resolving
        it as a ground station would send the station lookup into the MPC
        catalogue, where a spacecraft does not exist.
    ``geocentric``
        Earth's centre, which is a body origin rather than a station.

    Args:
        dataset_cfg: Dataset config, supplying the target satellite.
        station_name: Normalised observatory code, or the observer's body name.
        relative_to: Name of the reference body, or ``None`` for absolute data.
        frame_center: The frame centre recorded in the standard frame.  Falls back
            to the config's own field, then to topocentric, so the NSDB path is
            unaffected.

    Returns:
        The link-ends mapping, ready for a ``LinkDefinition``.
    """
    target = next(iter(dataset_cfg.satellites))
    link_ends: dict = {}

    if relative_to is not None:
        # transmitter2 carries the reference body; transmitter the target.  The
        # position angle is measured *of* the target *from* the reference
        # direction, which is the order the model documents.
        link_ends[obs_model_setup.links.transmitter] = (
            obs_model_setup.links.body_origin_link_end_id(relative_to)
        )
        link_ends[obs_model_setup.links.transmitter2] = (
            obs_model_setup.links.body_origin_link_end_id(target)
        )
    else:
        link_ends[obs_model_setup.links.transmitter] = (
            obs_model_setup.links.body_origin_link_end_id(target)
        )

    center = frame_center
    if center is None:
        center = dataset_cfg.get("center_of_frame") or dataset_cfg.get("centre_of_frame") or ""
    if not isinstance(center, FrameCenter):
        # Go through the vocabulary normaliser rather than the enum constructor.
        # The configs hold the *source's* prose, and NSDB writes the British
        # spelling (``topocentre``, 77 datasets) plus typo variants
        # (``geocentre``, ``geoocentre``) that are not enum members at all.  The
        # normaliser is what the ingestion path already used to fill the frame's
        # ``frame_center`` column, so parsing the config any other way would
        # disagree with the frame it is meant to describe.
        center = normalize_frame_center(center)

    if not receiver_needs_observatory(center):
        if center is FrameCenter.SPACECRAFT:
            # The observer is itself a body (Voyager 2), so a body origin.
            link_ends[obs_model_setup.links.receiver] = (
                obs_model_setup.links.body_origin_link_end_id(station_name)
            )
        else:
            # Geocentric: Earth's centre, which is a body origin, not a station.
            link_ends[obs_model_setup.links.receiver] = (
                obs_model_setup.links.body_origin_link_end_id("Earth")
            )
    else:
        link_ends[obs_model_setup.links.receiver] = (
            obs_model_setup.links.body_reference_point_link_end_id("Earth", station_name)
        )
    return link_ends


def frame_center_of(dataframe: pd.DataFrame) -> FrameCenter:
    """Read the single frame centre a standard frame carries.

    Raises:
        ValueError: If the frame mixes centres, which would need splitting first.
    """
    values = set(dataframe["frame_center"].dropna().astype(str).unique())
    if len(values) != 1:
        raise ValueError(f"Expected one frame centre per frame, got {sorted(values) or 'none'}.")
    return FrameCenter(values.pop())


def receiver_needs_observatory(frame_center: FrameCenter) -> bool:
    """Whether the receiver must be added to the environment as a ground station.

    Only a ground station is a point on Earth that has to be created from the MPC
    catalogue.  A spacecraft is a body the environment already provides, and a
    geocentric observer is Earth's own centre, so neither has a station to look
    up - and trying would search the catalogue for a body name and fail.

    This is deliberately expressed as "which of the two special cases is it"
    rather than "is it topocentric", because it must agree with the link-end
    branch in :func:`build_link_ends`.  Anything the vocabulary could not place -
    an unrecognised label, or one of NSDB's typos that the normaliser happened
    not to match - falls through to the ground-station path there, so it has to
    be treated as one here too.  Asking for equality with ``TOPOCENTRIC`` would
    answer ``False`` for such a frame and skip the station setup while still
    building a station link end.
    """
    return frame_center not in (FrameCenter.GEOCENTRIC, FrameCenter.SPACECRAFT)


def _relative_to_body_name(dataset_cfg: DictConfig) -> str | None:
    """Resolve the reference body for a relative dataset, or ``None``.

    NSDB encodes this four ways - the literal ``"absolute"``, the literal
    ``"planet"``, a planet name, or a designation such as ``"N1-Triton"`` - and
    the two old relative factories parsed it differently, one of them raising on
    the literal ``"absolute"`` that every absolute dataset contains.

    The literal ``"planet"`` is resolved through the config's own ``planet.name``
    rather than being returned as a body name, because no body is called
    ``"planet"``; leaving it unresolved makes every lookup in the system of
    bodies fail.
    """
    designation = None
    relative_to = dataset_cfg.get("relative_to")
    if relative_to:
        designation = relative_to.get("designation")
    if designation is None:
        designation = dataset_cfg.get("relative_to_object")
    text = "" if designation is None else str(designation).strip()
    if not text or text.lower() == "absolute":
        return None
    if text.lower() == "planet":
        name = (dataset_cfg.get("planet") or {}).get("name")
        return str(name).strip() if name else None
    if " - " in text:
        return text.split(" - ")[1].strip()
    if "-" in text:
        return text.split("-")[-1].strip()
    return text


def finalize(
    observation_sets: list,
    observation_models: list,
) -> tuple:
    """Return a single set/model when there is one, else a collection.

    Preserves the previous contract: a single-observatory dataset yields its set
    and model directly, so downstream code does not have to unwrap a
    one-element collection.
    """
    if len(observation_sets) == 1:
        return observation_sets[0], observation_models[0]
    return obs.ObservationCollection(observation_sets), observation_models


# ---------------------------------------------------------------------------
# Observable construction
# ---------------------------------------------------------------------------


def _as_observation_data(values: np.ndarray) -> list[np.ndarray]:
    """Shape a 1-D or 2-D measurement into the ``(n, 1)`` columns Tudat expects."""
    array = np.asarray(values, dtype="float64")
    if array.ndim == 1:
        return [np.array([[value]]) for value in array.tolist()]
    return [array[index].reshape(-1, 1) for index in range(array.shape[0])]


def _usable_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop rows whose required values or epoch are not finite.

    Also drops ``NOT_GIVEN`` entries: the schema keeps a one-dimensional
    observation as a real row with a sentinel for the missing half, and such a row
    cannot be passed to a 2-D observable.
    """
    from orbitdet.preprocessing.schema import is_not_given

    keep = np.isfinite(pd.to_numeric(frame[EPOCH_COLUMN], errors="coerce"))
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        keep &= np.isfinite(values) & ~is_not_given(values)
    return frame[keep]


# ---------------------------------------------------------------------------
# Factory 1: absolute RA/Dec
# ---------------------------------------------------------------------------


@register_dataset_factory(
    dispatch_key(ObservationType.ABSOLUTE, CoordinateKind.RA_DEC),
    aliases=(
        "absolute_absolute_CCD_nsdb",
        "absolute_absolute_photographic_nsdb",
        "absolute_absolute_meridian transit circle_nsdb",
        "absolute_absolute_visual_nsdb",
        "absolute_absolute_vidicon_nsdb",
        # Voyager 2's absolute RA/Dec tables are the same observable as an
        # absolute NSDB dataset, so they share this factory.  The alias keeps
        # the committed `type: voyager` configs loading while the registry
        # learns to dispatch them on their normalised fields.
        "voyager",
    ),
)
def create_absolute_radec_dataset(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    *,
    frame: pd.DataFrame,
) -> tuple:
    """Build absolute angular-position sets from a standard RA/Dec frame.

    The observable is the direction from the observatory to the satellite, which
    is exactly what ``first_value`` / ``second_value`` hold once reoriented.

    Args:
        cfg: Experiment config, supplying ``global_frame_orientation``.
        dataset_cfg: Dataset config.
        system_of_bodies: The environment providing the observatories.

    Returns:
        ``(SingleObservationSet, ObservationModelSettings)`` for a
        single-observatory dataset, or an ``ObservationCollection`` and a list of
        models when the dataset observed from more than one station.
    """
    dataframe = require_standard_frame(frame, dataset_cfg)
    if len(dataset_cfg.satellites) > 1:
        raise NotImplementedError(
            f"Dataset {dataset_cfg.identifier}: multiple satellites in one file are not "
            f"supported yet."
        )

    observation_sets = []
    observation_models = []
    center = frame_center_of(dataframe)
    for station_name, cluster in observatory_groups(dataframe, dataset_cfg):
        if receiver_needs_observatory(center):
            add_observatory_to_SOB(cfg, system_of_bodies, station_name)
        link_ends = build_link_ends(dataset_cfg, station_name, frame_center=center)
        link_definition = obs_model_setup.links.LinkDefinition(link_ends)

        rows = _usable_rows(cluster, [RA_COLUMN, DEC_COLUMN])
        times = rows[EPOCH_COLUMN].to_numpy(dtype="float64").tolist()
        data = _as_observation_data(rows[[RA_COLUMN, DEC_COLUMN]].to_numpy(dtype="float64"))

        observation_sets.append(
            obs.create_single_observation_set(
                obs_model_setup.model_settings.angular_position_type,
                link_ends,
                data,
                times,
                obs_model_setup.links.LinkEndType.receiver,
            )
        )
        observation_models.append(obs_model_setup.model_settings.angular_position(link_definition))

    return finalize(observation_sets, observation_models)


# ---------------------------------------------------------------------------
# Factory 2: relative RA/Dec (differential)
# ---------------------------------------------------------------------------


@register_dataset_factory(
    dispatch_key(ObservationType.RELATIVE, CoordinateKind.RA_DEC),
    aliases=("relative_differential_CCD_nsdb",),
)
def create_relative_radec_dataset(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    *,
    frame: pd.DataFrame,
) -> tuple:
    """Build relative angular-position sets from a differential RA/Dec frame.

    ``first_value`` / ``second_value`` are already the offset of the satellite
    from the reference body, so no geometry is needed beyond reorienting them -
    which the ingestion path has done.

    Args:
        cfg: Experiment config.
        dataset_cfg: Dataset config.
        system_of_bodies: The environment providing the observatories.

    Returns:
        One set and model, or a collection and list of models.
    """
    dataframe = require_standard_frame(frame, dataset_cfg)
    if len(dataset_cfg.satellites) > 1:
        raise NotImplementedError(
            f"Dataset {dataset_cfg.identifier}: multiple satellites in one file are not "
            f"supported yet."
        )

    relative_to = _relative_to_body_name(dataset_cfg)
    if relative_to is None:
        raise ValueError(
            f"Dataset {dataset_cfg.identifier} is registered as relative but names no "
            f"reference body, so transmitter2 cannot be built."
        )

    observation_sets = []
    observation_models = []
    center = frame_center_of(dataframe)
    for station_name, cluster in observatory_groups(dataframe, dataset_cfg):
        if receiver_needs_observatory(center):
            add_observatory_to_SOB(cfg, system_of_bodies, station_name)
        link_ends = build_link_ends(
            dataset_cfg, station_name, relative_to=relative_to, frame_center=center
        )
        link_definition = obs_model_setup.links.LinkDefinition(link_ends)

        rows = _usable_rows(cluster, [RA_COLUMN, DEC_COLUMN])
        times = rows[EPOCH_COLUMN].to_numpy(dtype="float64").tolist()
        data = _as_observation_data(rows[[RA_COLUMN, DEC_COLUMN]].to_numpy(dtype="float64"))

        observation_sets.append(
            obs.create_single_observation_set(
                RELATIVE_ANGULAR_POSITION_TYPE,
                link_ends,
                data,
                times,
                obs_model_setup.links.LinkEndType.receiver,
            )
        )
        observation_models.append(
            obs_model_setup.model_settings.relative_angular_position(link_definition)
        )

    return finalize(observation_sets, observation_models)


# ---------------------------------------------------------------------------
# Factory 3: relative position angle / separation (one-dimensional)
# ---------------------------------------------------------------------------


@register_dataset_factory(
    dispatch_key(ObservationType.RELATIVE, CoordinateKind.POSITION_ANGLE_SEPARATION),
    aliases=(
        "relative_position_angle_and_separation_micrometric_nsdb",
        "relative_position_angle_and_separation_photographic_nsdb",
        "relative_position_angle_and_separation_photograpihc_nsdb",
        "relative_position_angle_and_distance_micrometric_nsdb",
        "relative_Position_angle,_Distance_visual_nsdb",
    ),
)
def create_relative_position_angle_dataset(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    *,
    frame: pd.DataFrame,
) -> tuple:
    """Build one-dimensional relative sets from a position-angle/separation frame.

    Each row carries a position angle, a separation, or - across 34 datasets -
    exactly one of the two, with the other marked not-given.  That makes many rows
    genuinely one-dimensional, which is why two observation **models** are needed
    rather than one combined observable: the combined model returns a two-element
    observation and cannot accept a row whose half was never measured.

    The models are built per cluster and appended in step with the sets, so a
    dataset holding both kinds of row yields two models for one station.

    Args:
        cfg: Experiment config.
        dataset_cfg: Dataset config.
        system_of_bodies: The environment providing the observatories.

    Returns:
        One set and model, or a collection and list of models.
    """
    from orbitdet.preprocessing.schema import is_not_given

    dataframe = require_standard_frame(frame, dataset_cfg)
    if len(dataset_cfg.satellites) > 1:
        raise NotImplementedError(
            f"Dataset {dataset_cfg.identifier}: multiple satellites in one file are not "
            f"supported yet."
        )

    relative_to = _relative_to_body_name(dataset_cfg)
    if relative_to is None:
        raise ValueError(
            f"Dataset {dataset_cfg.identifier} is registered as relative but names no "
            f"reference body, so transmitter2 cannot be built."
        )

    # A position angle is measured from a pole, and the pole is part of the
    # measurement's definition rather than a correction applied afterwards.  The
    # ingestion path records the source's equinox, so the model's pole can be set
    # to match what the observers actually used.
    ancillary = obs_setup.ancillary_settings.position_angle_ancillary_settings(
        reference_frame=_position_angle_reference_frame(
            dataframe.attrs.get("source_equinox") or dataset_cfg.get("epoch_of_equinox")
        ),
    )

    observation_sets: list = []
    observation_models: list = []
    center = frame_center_of(dataframe)

    for station_name, cluster in observatory_groups(dataframe, dataset_cfg):
        if receiver_needs_observatory(center):
            add_observatory_to_SOB(cfg, system_of_bodies, station_name)
        link_ends = build_link_ends(
            dataset_cfg, station_name, relative_to=relative_to, frame_center=center
        )
        link_definition = obs_model_setup.links.LinkDefinition(link_ends)

        angle = pd.to_numeric(cluster["first_value"], errors="coerce")
        separation = pd.to_numeric(cluster["second_value"], errors="coerce")
        finite = np.isfinite(pd.to_numeric(cluster[EPOCH_COLUMN], errors="coerce"))
        has_angle = finite & np.isfinite(angle) & ~is_not_given(angle)
        has_separation = finite & np.isfinite(separation) & ~is_not_given(separation)

        for mask, observable_type, model_factory, column in (
            (
                has_angle,
                obs_model_setup.model_settings.ObservableType.position_angle_type,
                obs_model_setup.model_settings.position_angle,
                "first_value",
            ),
            (
                has_separation,
                obs_model_setup.model_settings.ObservableType.separation_distance_type,
                obs_model_setup.model_settings.separation_distance,
                "second_value",
            ),
        ):
            selected = cluster[mask]
            if selected.empty:
                continue
            observation_sets.append(
                obs.create_single_observation_set(
                    observable_type,
                    link_ends,
                    _as_observation_data(selected[column].to_numpy(dtype="float64")),
                    selected[EPOCH_COLUMN].to_numpy(dtype="float64").tolist(),
                    obs_model_setup.links.LinkEndType.receiver,
                    ancillary_settings=ancillary,
                )
            )
            observation_models.append(model_factory(link_definition))

    if not observation_sets:
        # Every row was not-given on both components, which is a data problem
        # rather than an empty dataset.  Raising keeps it visible: a silently
        # empty collection would let the dataset vanish from a fit.
        raise ValueError(
            f"Dataset {dataset_cfg.identifier}: no row carries a position angle or a "
            f"separation, so no observation set could be built."
        )

    return finalize(observation_sets, observation_models)


def _position_angle_reference_frame(equinox: object):
    """Map an equinox label onto the model's pole enumeration.

    The position-angle model is given a *reference frame* whose north pole defines
    the angle's zero point, not an epoch.  Equinoxes of the same type share a pole,
    so anything the enumeration cannot express is reported rather than
    approximated.
    """
    from orbitdet.preprocessing.precession import EquinoxSystem, parse_equinox

    frame = obs_setup.ancillary_settings.PositionAngleReferenceFrame
    label = "" if equinox is None else str(equinox).strip()

    parsed = parse_equinox(label)
    if not parsed.recognised:
        logger.warning(
            "Position-angle dataset declares equinox %r, which is not recognised; "
            "using the J2000 pole.",
            label,
        )
        return frame.j2000

    if parsed.system is EquinoxSystem.OF_DATE:
        return frame.mean_of_date_iau_2006
    if parsed.system is EquinoxSystem.TRUE_OF_DATE:
        return frame.true_of_date_iau_2006_2000a
    if parsed.is_besselian:
        return frame.b1950
    return frame.j2000


# ---------------------------------------------------------------------------
# Factory 4: relative tangent-plane X/Y
# ---------------------------------------------------------------------------


@register_dataset_factory(
    dispatch_key(ObservationType.RELATIVE, CoordinateKind.CARTESIAN_XY),
    aliases=(
        "relative_X,Y_CCD_nsdb",
        "relative_X,_Y_photographic_nsdb",
    ),
)
def create_relative_xy_dataset(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    system_of_bodies: env.SystemOfBodies,
    *,
    frame: pd.DataFrame,
) -> tuple:
    """Build relative angular-position sets from tangent-plane X/Y offsets.

    X/Y offsets are standard coordinates on a plane tangent to the celestial
    sphere at the reference body, so recovering a direction needs the body's
    topocentric position at each epoch.  That is the one place a factory still does
    geometry, because it is the only layer with the system of bodies; the
    trigonometry itself lives in
    :func:`orbitdet.preprocessing.gnomonic.standard_to_relative_radec`.

    The recovered direction is carried into the global frame *together with* the
    tangent point.  Rotating the offsets alone would reorient the tangent plane's
    origin rather than the two directions measured from it, which is a silent
    error proportional to the frame difference - hundreds of arcseconds for a
    B1950 dataset.

    Args:
        cfg: Experiment config, supplying ``global_frame_orientation``.
        dataset_cfg: Dataset config.
        system_of_bodies: The environment providing ephemerides and observatories.

    Returns:
        One set and model, or a collection and list of models.
    """
    from orbitdet.preprocessing.frames import convert_radec_frame
    from orbitdet.preprocessing.gnomonic import standard_to_relative_radec

    dataframe = require_standard_frame(frame, dataset_cfg)
    if len(dataset_cfg.satellites) > 1:
        raise NotImplementedError(
            f"Dataset {dataset_cfg.identifier}: multiple satellites in one file are not "
            f"supported yet."
        )

    relative_to = _relative_to_body_name(dataset_cfg)
    if relative_to is None:
        raise ValueError(
            f"Dataset {dataset_cfg.identifier} is registered as relative but names no "
            f"reference body, so the tangent point is unknown."
        )

    source_equinox = dataframe.attrs.get("source_equinox") or dataset_cfg.get("epoch_of_equinox")
    global_frame = cfg.global_frame_orientation

    observation_sets = []
    observation_models = []
    center = frame_center_of(dataframe)
    for station_name, cluster in observatory_groups(dataframe, dataset_cfg):
        if receiver_needs_observatory(center):
            add_observatory_to_SOB(cfg, system_of_bodies, station_name)
        link_ends = build_link_ends(
            dataset_cfg, station_name, relative_to=relative_to, frame_center=center
        )
        link_definition = obs_model_setup.links.LinkDefinition(link_ends)

        rows = _usable_rows(cluster, ["first_value", "second_value"])
        epochs = rows[EPOCH_COLUMN].to_numpy(dtype="float64")
        if rows.empty:
            continue

        # The tangent point: the body's topocentric direction, expressed in the
        # orientation the source's plate was reduced in.
        tangent_ra, tangent_dec = _topocentric_body_direction(
            system_of_bodies, relative_to, station_name, epochs
        )
        tangent = pd.DataFrame(
            {
                "ra_deg": np.rad2deg(tangent_ra),
                "dec_deg": np.rad2deg(tangent_dec),
                EPOCH_COLUMN: epochs,
            }
        )
        tangent = convert_radec_frame(
            tangent,
            "ra_deg",
            "dec_deg",
            input_frame=global_frame,
            output_frame=source_equinox,
            time_column=EPOCH_COLUMN,
        )
        ra0 = np.deg2rad(tangent["ra_deg"].to_numpy(dtype="float64"))
        dec0 = np.deg2rad(tangent["dec_deg"].to_numpy(dtype="float64"))

        # Invert the projection to get the offset's direction in the source frame.
        delta_ra, delta_dec = standard_to_relative_radec(
            rows["first_value"].to_numpy(dtype="float64"),
            rows["second_value"].to_numpy(dtype="float64"),
            ra0,
            dec0,
        )

        # Carry the satellite direction and the tangent point into the global
        # frame *together*, in one batched conversion.  Two reasons: a relative
        # geometry only survives a rotation if both ends rotate the same way, and
        # doing it in one call keeps the pair from drifting apart if the
        # conversion is ever refined.  The satellite rows come first so the two
        # halves can be split back out below.
        count = len(rows)
        paired = pd.DataFrame(
            {
                "ra_deg": np.rad2deg(np.concatenate([ra0 + delta_ra, ra0])),
                "dec_deg": np.rad2deg(np.concatenate([dec0 + delta_dec, dec0])),
                EPOCH_COLUMN: np.concatenate([epochs, epochs]),
            }
        )
        paired = convert_radec_frame(
            paired,
            "ra_deg",
            "dec_deg",
            input_frame=source_equinox,
            output_frame=global_frame,
            time_column=EPOCH_COLUMN,
        )

        converted_ra = np.deg2rad(paired["ra_deg"].to_numpy(dtype="float64"))
        converted_dec = np.deg2rad(paired["dec_deg"].to_numpy(dtype="float64"))
        satellite_ra, satellite_dec = converted_ra[:count], converted_dec[:count]
        body_ra, body_dec = converted_ra[count:], converted_dec[count:]

        # The observable is the offset of the satellite from the reference body,
        # as seen from the observatory - the same convention the model computes
        # from the global-frame state vectors.
        delta = np.column_stack(
            [
                np.arctan2(np.sin(satellite_ra - body_ra), np.cos(satellite_ra - body_ra)),
                satellite_dec - body_dec,
            ]
        )

        observation_sets.append(
            obs.create_single_observation_set(
                RELATIVE_ANGULAR_POSITION_TYPE,
                link_ends,
                _as_observation_data(delta),
                epochs.tolist(),
                obs_model_setup.links.LinkEndType.receiver,
            )
        )
        observation_models.append(
            obs_model_setup.model_settings.relative_angular_position(link_definition)
        )

    if not observation_sets:
        raise ValueError(
            f"Dataset {dataset_cfg.identifier}: no usable X/Y row, so no observation set "
            f"could be built."
        )
    return finalize(observation_sets, observation_models)


def _topocentric_body_direction(
    system_of_bodies: env.SystemOfBodies,
    body_name: str,
    station_name: str,
    epochs_tdb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Direction from an observatory to a body's centre at each epoch.

    Uses the body and Earth ephemerides plus Earth's rotation, so it matches the
    geometry the ``relative_angular_position`` model computes from the same state
    vectors.

    Args:
        system_of_bodies: The environment.
        body_name: Body to look at, e.g. ``"Neptune"``.
        station_name: Normalised observatory code.
        epochs_tdb: Epochs as TDB seconds since J2000.

    Returns:
        Tuple ``(ra_radians, dec_radians)`` in the global frame orientation.
    """
    earth = system_of_bodies.get("Earth")
    body = system_of_bodies.get(body_name)
    station = earth.get_ground_station(station_name)

    epochs = np.asarray(epochs_tdb, dtype="float64")
    ra = np.empty(epochs.shape, dtype="float64")
    dec = np.empty(epochs.shape, dtype="float64")

    for index, epoch in enumerate(epochs.tolist()):
        station_fixed = np.ravel(station.station_state.get_cartesian_position(epoch))
        rotation = np.array(earth.rotation_model.body_fixed_to_inertial_rotation(epoch))
        station_inertial = np.ravel(earth.ephemeris.cartesian_position(epoch)) + (
            rotation @ station_fixed
        )
        body_inertial = np.ravel(body.ephemeris.cartesian_position(epoch))

        direction = body_inertial - station_inertial
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            ra[index] = np.nan
            dec[index] = np.nan
            continue
        unit = direction / norm
        ra[index] = np.arctan2(unit[1], unit[0])
        dec[index] = np.arcsin(np.clip(unit[2], -1.0, 1.0))

    return ra, dec
