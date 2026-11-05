"""Observation collection builder."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, OmegaConf

from orbitdet.data.frame_sources import standard_frame_for
from orbitdet.data.nsdb_source import NSDBSource
from orbitdet.preprocessing.frames import convert_standard_frame
from orbitdet.preprocessing.schema import OBSERVATION_SET_INDEX_COLUMN

from .factory import create_observation_dataset
from .helpers import normalize_observatory_code
from .standard_frames import EPOCH_COLUMN
from .utils import get_set_identifier

logger = logging.getLogger(__name__)


def _frame_candidates(frame: pd.DataFrame, single_set) -> np.ndarray:
    """Return the positions in ``frame`` the set could have been built from.

    A factory splits a frame by observatory, so the frame's own
    ``observatory_code`` says which rows belong to which set - and constraining
    the search to them is what keeps two stations that observed at the same
    instant from being attributed to each other.  Sets whose identifier is not a
    code in the frame (a spacecraft's body name, a geocentric observer) fall back
    to every row, which is sound because such a dataset yields a single set.
    """
    every_row = np.arange(len(frame), dtype=int)
    if "observatory_code" not in frame.columns:
        return every_row
    want = normalize_observatory_code(get_set_identifier(single_set))
    codes = frame["observatory_code"].astype(str).map(normalize_observatory_code)
    matches = np.flatnonzero(codes.to_numpy(dtype=object) == want)
    return matches if matches.size else every_row


def _rows_of_set(frame: pd.DataFrame, single_set, candidates: np.ndarray) -> np.ndarray:
    """Return the positions in ``frame`` of the rows ``single_set`` was built from.

    A Tudat ``SingleObservationSet`` keeps only its epochs, so a set is matched
    back to the frame it came from on that column.  The frame is not recoverable
    from the set afterwards, which is why the link is made here rather than being
    written out by each factory - and why the factories stay untouched.

    Rows are consumed in file order, so a frame holding the same epoch twice maps
    its two rows to the set's two observations in the order both were built in:
    the clustering and ``_usable_rows`` both preserve file order.  Each set is
    matched independently, because a position angle and a separation in one
    dataset are built from the *same* rows and neither consumes the other's.

    Args:
        frame: The standard frame the set was built from.
        single_set: The observation set.
        candidates: Positions in ``frame`` the set could have been built from.

    Returns:
        Positions into ``frame``, in the order the set's epochs appear.  Empty
        when the frame carries no such epoch, which leaves those rows' set index
        null rather than guessing one.
    """
    if EPOCH_COLUMN not in frame.columns or candidates.size == 0:
        return np.array([], dtype=int)

    epochs = pd.to_numeric(frame[EPOCH_COLUMN], errors="coerce").to_numpy(dtype="float64")
    used = np.zeros(epochs.shape, dtype=bool)
    positions: list[int] = []
    for epoch in single_set.observation_times:
        close = np.isclose(epochs, float(epoch), rtol=1e-12, atol=1e-6)
        matches = candidates[close[candidates] & ~used[candidates]]
        if matches.size == 0:
            continue
        position = int(matches[0])
        used[position] = True
        positions.append(position)
    return np.asarray(positions, dtype=int)


def create_observation_collection(
    cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, list[Any], pd.DataFrame]:
    """Build an observation collection from multiple dataset configs.

    Iterates through dataset configurations in a collection, dispatches each through
    the central factory, and aggregates the resulting observation sets.

    Args:
        cfg: Hydra config with 'datasets' list. Each entry should be a
                       dataset config with a 'type' field.
        system_of_bodies: The environment containing the bodies for which to create observations.

    Returns:
        Tuple of:
            - ObservationCollection containing all observation sets.
            - List of observation model settings.
            - The concatenated standard frame: every row the frame-backed datasets
              contributed, in dataset order, carrying the full schema (provenance,
              units, epochs and measurements).  This is the same data the factories
              consumed, so it is the single description of the observations; there
              is no parallel metadata mapping to keep in step with it.  The
              ``observation_set_index`` column holds the index of the set each row
              was used to build, so a row can be traced back with
              ``collection.get_single_observation_sets()[index]``; it is null for
              a row no set was built from.  A row whose two values became two sets
              (a position angle and a separation) names the first.  Datasets that
              build their observations from another source (a simulated or
              hand-written table) contribute no rows; a collection made only of
              those returns an empty frame carrying the standard columns.
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    datasets = OmegaConf.select(cfg, "datasets")
    if datasets is None:
        raise ValueError("Collection config must have a 'datasets' list")

    logger.info(f"Creating observation collection with {len(datasets)} dataset(s)")

    observation_sets = []
    model_setting = []
    # Standard frames, one per frame-backed dataset, in dataset order.  They are
    # concatenated at the end and returned alongside the collection, so callers
    # hold the data the observations were built from rather than a second,
    # hand-maintained description of it.
    standard_frames: list[pd.DataFrame] = []
    # Every single observation set the factories produced, as
    # ``(set, metadata, frame_position)``.  ``frame_position`` is the position in
    # ``standard_frames`` of the standard frame the set was built from, or ``None``
    # for a modality that built from its own source.  A Tudat
    # ``merge_observation_collections`` **sorts** sets by link ends and observable
    # type but preserves each set's **object identity**, so the final collection is
    # walked below and every set looked up here by ``id``.  A FIFO queue keyed by
    # ``set_id`` cannot stand in for that: where a station several datasets observed
    # from, the merge need not order those datasets' sets in dataset order, and a
    # queue would then hand a set the wrong dataset's frame and leave its rows
    # unmarked.
    set_records: list[tuple[obs.SingleObservationSet, dict, int | None]] = []
    # One reader per collection: parsing the descriptor and re-reading the
    # observatory catalogue per dataset would be pure repetition.  The NSDB root
    # is not read from ``cfg.data_folder``: every experiment config already uses
    # that key for the *kernel* folder (``data/other``), so the descriptor is
    # located from each dataset's own data file instead (see
    # :meth:`orbitdet.data.nsdb_source.NSDBSource.descriptor_for`).
    nsdb_source = NSDBSource()

    for idx, (set_name, dataset_cfg) in enumerate(datasets.items()):
        try:
            logger.debug(f"Creating dataset {set_name} ({idx + 1}/{len(datasets)})")
            # NSDB datasets are read into a standard frame here, at the boundary,
            # so the factory only ever sees data that satisfies the schema.  A
            # modality with no frame of its own returns None and the factory
            # falls back to its own source.
            frame = standard_frame_for(
                dataset_cfg,
                cfg=cfg,
                nsdb_source=nsdb_source,
                system_of_bodies=system_of_bodies,
            )
            if frame is not None:
                # Carry RA/Dec frames into the global orientation before the
                # factory sees them.  A frame records the equinox *its source*
                # used, and a factory should not have to know that; reorienting
                # here means every factory receives coordinates already in the
                # global frame.  Non-RA/Dec kinds are passed through untouched
                # (they cannot be rotated on their own - see
                # :func:`orbitdet.preprocessing.frames.convert_standard_frame`).
                frame = convert_standard_frame(
                    frame, str(cfg.get("global_frame_orientation", "J2000"))
                )
                standard_frames.append(frame)
            frame_position = None if frame is None else len(standard_frames) - 1
            dataset, model_settings = create_observation_dataset(
                cfg, dataset_cfg, system_of_bodies, frame=frame
            )
            observation_sets.append(dataset)
            model_setting.append(model_settings)

            # Record each set the factory produced, paired with the standard frame
            # it was built from (``None`` for a modality that built from its own
            # source).  The frame is the description of the observations; there is
            # no parallel metadata dict to keep in step with it.
            if isinstance(dataset, obs.ObservationCollection):
                single_sets = dataset.get_single_observation_sets()
            elif isinstance(dataset, obs.SingleObservationSet):
                single_sets = [dataset]
            else:
                single_sets = []
            for single_set in single_sets:
                set_records.append((single_set, frame_position))

            logger.debug(f"Successfully created dataset {set_name} ({idx + 1}/{len(datasets)})")
        except Exception as e:
            logger.error(
                f"Failed to create dataset {idx + 1}/{len(datasets)}: {dataset_cfg}. Error: {e}"
            )
            raise

    # Sort ObservationCollections and SingleObservationSets into separate lists to merge
    # separately, then combine
    observation_collections = [
        s for s in observation_sets if isinstance(s, obs.ObservationCollection)
    ]
    single_observation_sets = [
        s for s in observation_sets if not isinstance(s, obs.ObservationCollection)
    ]

    single_set_collection = (
        obs.ObservationCollection(single_observation_sets) if single_observation_sets else None
    )
    total_collection = obs.merge_observation_collections(
        observation_collections + ([single_set_collection] if single_set_collection else [])
    )

    # Assign the final index of each single observation set.  ``merge_observation_collections``
    # orders sets internally (by observable type, then link ends), so we cannot rely on
    # insertion order.  It does preserve each set's object identity, though, so we resolve
    # every set in the final collection by ``id`` back to the frame and metadata it was
    # created with.  This mirrors how ``WeightEngine``/``OutlierEngine`` iterate the
    # collection, so ``index`` always matches the position a caller would retrieve via
    # ``total_collection.get_single_observation_sets()[index]``.  A ``set_id`` FIFO queue
    # cannot stand in for identity: where several datasets observed from one station the
    # merge need not order their sets in dataset order, and the queue would then attribute
    # one dataset's rows to another dataset's set.
    by_identity = {id(single_set): position for single_set, position in set_records}

    # One column buffer per frame, filled during the walk and attached afterwards.
    set_index_columns = [
        np.full(len(data_frame), np.nan, dtype="float64") for data_frame in standard_frames
    ]

    for set_index, single_set in enumerate(total_collection.get_single_observation_sets()):
        set_id = get_set_identifier(single_set)
        if id(single_set) not in by_identity:
            logger.warning(
                "Observation set '%s' at index %d was not created by this builder; its "
                "rows cannot be resolved.",
                set_id,
                set_index,
            )
            continue
        position = by_identity[id(single_set)]
        if position is not None:
            data_frame = standard_frames[position]
            rows = _rows_of_set(data_frame, single_set, _frame_candidates(data_frame, single_set))
            # First set wins.  One row can feed more than one set - a
            # position-angle-separation row carries both a position angle and a
            # separation, and each becomes its own set - and a single float cannot
            # name both.  Keeping the first means the column points at the set
            # built from the row's *first* component, which is the primary one.
            fresh = rows[np.isnan(set_index_columns[position][rows])]
            if fresh.size:
                set_index_columns[position][fresh] = float(set_index)

    logger.info(f"Successfully created observation collection with {len(observation_sets)} set(s)")

    # Flatten model settings list — factories may return a single model or a list of models,
    # and we accumulate them by appending, producing a nested list [[m1, m2], [m3], ...].
    flat_models = []
    for ms in model_setting:
        if isinstance(ms, (list, tuple)):
            flat_models.extend(ms)
        else:
            flat_models.append(ms)

    if standard_frames:
        standard_frame = pd.concat(standard_frames, ignore_index=True)
        standard_frame[OBSERVATION_SET_INDEX_COLUMN] = np.concatenate(set_index_columns)
    else:
        # No frame-backed dataset in this collection (a Gaia-only, simulated-only
        # one, or a hand-written table), so there are no source rows to return.
        # An empty frame carrying the standard columns is returned rather than
        # ``None`` so that consumers can index it unconditionally.
        from orbitdet.preprocessing.schema import empty_frame

        standard_frame = empty_frame()

    return total_collection, flat_models, standard_frame
