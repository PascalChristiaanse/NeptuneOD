"""Choose the reader that turns a dataset config into a standard frame.

Every observation source is read through this module, so the collection builder
has a single call - ``standard_frame_for`` - rather than a growing chain of
per-modality branches.  A source registers a reader for its ``type``, and a
dataset whose type has no reader simply has no frame: the factory for that
modality (simulated, which generates its own data) still runs.

Adding a modality therefore means writing a reader and registering it here.  The
collection builder does not change.

Why a registry rather than an ``if`` chain
------------------------------------------
The dataframe is meant to be *the* interface between reading data and building
models, and the number of sources is the thing that grows.  An ``if/elif`` chain
puts that growth in the shared builder, where every new modality risks the ones
already working; a registry localises it to one line next to the reader.

Examples:
    Build a frame for any configured dataset::

        >>> frame = standard_frame_for(dataset_cfg, cfg=cfg)   # doctest: +SKIP
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

from .nsdb_source import NSDBSource

logger = logging.getLogger(__name__)

__all__ = [
    "register_frame_source",
    "standard_frame_for",
    "registered_frame_sources",
]

#: Reader callables, keyed by the ``type`` value a dataset config declares.
_FRAME_SOURCES: dict[str, object] = {}


def register_frame_source(dataset_type: str, *, aliases: tuple[str, ...] = ()):
    """Register a standard-frame reader for a dataset type.

    Args:
        dataset_type: The config ``type`` this reader handles, e.g. ``"voyager"``.
        aliases: Additional ``type`` values that resolve to the same reader.

    Returns:
        The decorator, so a reader is registered at its definition.

    Raises:
        ValueError: If any name is already taken.  Checked for every name before
            anything is registered, so a failed call leaves the registry as it
            was rather than half-applied.
    """
    names = (dataset_type, *aliases)

    clashes = [name for name in names if name in _FRAME_SOURCES]
    if clashes:
        raise ValueError(
            f"A standard-frame reader is already registered for {', '.join(sorted(clashes))}."
        )

    def decorator(reader):
        for name in names:
            _FRAME_SOURCES[name] = reader
        return reader

    return decorator


# The NSDB reader takes an already-built :class:`NSDBSource`, because that object
# caches a parser, a formatter and the observatory catalogue across the datasets
# of one collection; rebuilding it per dataset would re-read the catalogue for
# every file.  Other readers take only the config.
@register_frame_source("nsdb", aliases=("nsdb_absolute", "nsdb_relative"))
def _nsdb_reader(dataset_cfg, *, cfg=None, nsdb_source=None, system_of_bodies=None):
    """Read an NSDB dataset through a shared :class:`NSDBSource`."""
    return (nsdb_source or NSDBSource()).frame_from_config(dataset_cfg)


@register_frame_source("voyager", aliases=("voyager_tabulated",))
def _voyager_reader(dataset_cfg, *, cfg=None, nsdb_source=None, system_of_bodies=None):
    """Read the Voyager 2 manual tables."""
    from .voyager_source import VoyagerSource

    return VoyagerSource(cfg).frame_from_config(dataset_cfg)


@register_frame_source("gaia", aliases=("gaia_archive", "gaia_fpr"))
def _gaia_reader(dataset_cfg, *, cfg=None, nsdb_source=None, system_of_bodies=None):
    """Read Gaia FPR astrometry from the archive, correcting it first.

    Gaia is the one source whose frame depends on the environment: the
    photocenter and light-deflection corrections are computed from ephemerides,
    and the archived coordinates are meaningless for orbit fitting without them.
    Applying them here rather than in the factory keeps the factory a pure
    frame-to-model transform like every other modality.
    """
    from .gaia_source import GaiaSource

    return GaiaSource(cfg, system_of_bodies=system_of_bodies).frame_from_config(dataset_cfg)


def _reader_for(dataset_cfg) -> object | None:
    """Find the reader for a dataset config, or ``None``.

    Dispatch is on the config's ``type``.  It also accepts a config that declares
    ``data_source`` instead, because that is the field the standard frame carries
    and the two are meant to agree.

    Args:
        dataset_cfg: A dataset config.

    Returns:
        The reader, or ``None`` when this dataset has no standard frame.
    """
    for field in ("data_source", "type"):
        value = dataset_cfg.get(field)
        if value is None:
            continue
        name = str(value).strip().lower()
        if name in _FRAME_SOURCES:
            return _FRAME_SOURCES[name]

    # NSDB's fused type strings all end in ``_nsdb`` and its download manager
    # writes ``data_source: nsdb``; neither is a key on its own, so the marker is
    # matched as a suffix.  Kept as a fallback rather than the primary rule so an
    # exact registration always wins.
    dataset_type = str(dataset_cfg.get("type") or "").strip().lower()
    if "nsdb" in dataset_type:
        return _FRAME_SOURCES["nsdb"]
    if "voyager" in dataset_type:
        return _FRAME_SOURCES["voyager"]
    return None


def standard_frame_for(
    dataset_cfg,
    *,
    cfg: DictConfig | None = None,
    nsdb_source: NSDBSource | None = None,
    system_of_bodies=None,
) -> pd.DataFrame | None:
    """Return the standard frame a dataset config describes, if it has one.

    Returning ``None`` rather than raising for a dataset with no reader is
    deliberate: ``simulated`` builds its observations from nothing at all, and the
    collection builder treats a missing frame as "this modality does not consume
    one" rather than as an error.

    Args:
        dataset_cfg: A dataset config.
        cfg: The experiment config.  Passed through because a reader may need it:
            the Voyager tables are located via the shared merge in
            :mod:`orbitdet.data.voyager_data`, which inspects ``bodies_to_create``.
        nsdb_source: An existing :class:`NSDBSource` to reuse, so one collection
            does not re-read the observatory catalogue per dataset.
        system_of_bodies: The environment.  Gaia needs it because its astrometric
            corrections (photocenter, light deflection) are computed from
            ephemerides and must be applied to the table *before* the frame is
            built - so the reader cannot be a pure table transform.

    Returns:
        The standard frame, or ``None`` when no reader is registered for this
        dataset's type.
    """
    reader = _reader_for(dataset_cfg)
    if reader is None:
        return None
    return reader(dataset_cfg, cfg=cfg, nsdb_source=nsdb_source, system_of_bodies=system_of_bodies)


def registered_frame_sources() -> list[str]:
    """Return a sorted list of the dataset types a frame can be built for."""
    return sorted(_FRAME_SOURCES.keys())
