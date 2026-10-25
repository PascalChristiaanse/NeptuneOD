"""Observation collection builder."""

import logging
from typing import Any

import tudatpy.dynamics.environment as env
import tudatpy.estimation.observations as obs
from omegaconf import DictConfig, ListConfig, OmegaConf

from .factory import create_observation_dataset
from .utils import get_set_identifier

logger = logging.getLogger(__name__)


def _source_name(dataset_cfg: DictConfig, set_name: str) -> str:
    """Extract a short academic source name from the dataset config.

    Prefers the first line of the ``reference`` field (e.g. "Veiga C.H.,
    Vieira Martins R. (1996)"), falling back to the dataset key.
    """
    ref = dataset_cfg.get("reference", "")
    if ref:
        first_line = ref.strip().split("\n")[0].strip()
        if first_line:
            return first_line
    return set_name


def _observatory_code(dataset_cfg: DictConfig) -> str:
    """Extract the observatory code from the dataset config.

    Handles both list form (NSDB configs) and dict form (voyager config).

    ``observatory``
      - List: ``[{- code: 689, name: ...}]`` → ``"689"``
      - Dict: ``{- code: -31, name: ...}``  → ``"-31"``
    """
    obs_val = dataset_cfg.get("observatory", None)
    if obs_val is None:
        return ""
    if isinstance(obs_val, ListConfig) and len(obs_val) > 0:
        code = obs_val[0].get("code", "")
        return str(code).strip()
    if isinstance(obs_val, DictConfig):
        code = obs_val.get("code", "")
        return str(code).strip()
    return ""


def _coordinate_type_label(dataset_type: str) -> tuple[str, str]:
    """Map a dataset type string to human-readable RA/DEC coordinate labels.

    Returns
    -------
    tuple[str, str]
        ``(ra_type, dec_type)`` — e.g. ``(r"$\\Delta\\alpha\\cos\\delta$", r"$\\Delta\\delta$")``
        for relative data, or ``(r"$\\alpha$", r"$\\delta$")`` for absolute.
    """
    t = dataset_type.lower()
    if "relative" in t:
        return (r"$\Delta\alpha\cos\delta$", r"$\Delta\delta$")
    if "absolute" in t or "voyager" in t:
        return (r"$\alpha$", r"$\delta$")
    if "position" in t and "angle" in t:
        return (r"$\theta$", r"$\rho$")
    return ("", "")


def create_observation_collection(
    cfg: DictConfig, system_of_bodies: env.SystemOfBodies
) -> tuple[obs.ObservationCollection, list[Any], dict[str, dict]]:
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
            - Dict mapping set_id (observatory code) to a list of metadata dicts, each
              with ``name`` (source author), ``coordinate_type`` (human-readable label),
              and ``index`` (position of the set in the final collection, so it can be
              retrieved via ``collection.get_single_observation_sets()[index]``).
    """
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(cfg)}")

    datasets = OmegaConf.select(cfg, "datasets")
    if datasets is None:
        raise ValueError("Collection config must have a 'datasets' list")

    logger.info(f"Creating observation collection with {len(datasets)} dataset(s)")

    observation_sets = []
    model_setting = []
    dataset_metadata: dict[str, list[dict]] = {}
    # Metadata entries per dataset, kept in dataset order.  Each entry is a
    # ``(set_id, metadata_dict)`` pair.  The ``index`` field (position of the set
    # in the final merged collection) is filled in below, once the merge order is
    # known, so that callers can retrieve a set via
    # ``collection.get_single_observation_sets()[index]``.
    per_dataset_metadata: list[list[tuple[str, dict]]] = []

    for idx, (set_name, dataset_cfg) in enumerate(datasets.items()):
        try:
            logger.debug(f"Creating dataset {set_name} ({idx + 1}/{len(datasets)})")
            dataset, model_settings = create_observation_dataset(cfg, dataset_cfg, system_of_bodies)
            observation_sets.append(dataset)
            model_setting.append(model_settings)

            # Extract metadata from the SingleObservationSets produced by this dataset
            source_name = _source_name(dataset_cfg, set_name)
            dataset_id = dataset_cfg.get("identifier", set_name)
            obs_code = _observatory_code(dataset_cfg)
            ra_type, dec_type = _coordinate_type_label(dataset_cfg.get("type", ""))
            dataset_entries: list[tuple[str, dict]] = []
            if isinstance(dataset, obs.ObservationCollection):
                single_sets = dataset.get_single_observation_sets()
            elif isinstance(dataset, obs.SingleObservationSet):
                single_sets = [dataset]
            else:
                single_sets = []
            for single_set in single_sets:
                set_id = get_set_identifier(single_set)
                dataset_entries.append(
                    (
                        set_id,
                        {
                            "name": source_name,
                            "dataset_id": dataset_id,
                            "observatory_code": obs_code or set_id,
                            "ra_type": ra_type,
                            "dec_type": dec_type,
                            "index": None,
                        },
                    )
                )
            per_dataset_metadata.append(dataset_entries)

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
    # insertion order.  Instead we walk the final collection in the exact order returned by
    # ``get_single_observation_sets()`` and consume a FIFO queue of metadata entries per
    # ``set_id``.  This mirrors how ``WeightEngine``/``OutlierEngine`` iterate the collection,
    # so ``index`` always matches the position a caller would retrieve via
    # ``total_collection.get_single_observation_sets()[index]``.
    meta_queues: dict[str, list[dict]] = {}
    for entries in per_dataset_metadata:
        for set_id, meta in entries:
            meta_queues.setdefault(set_id, []).append(meta)

    for set_index, single_set in enumerate(total_collection.get_single_observation_sets()):
        set_id = get_set_identifier(single_set)
        queue = meta_queues.get(set_id)
        if not queue:
            logger.warning(
                "No dataset metadata available for observation set '%s' at index %d.",
                set_id,
                set_index,
            )
            continue
        meta = queue.pop(0)
        meta["index"] = set_index
        dataset_metadata.setdefault(set_id, []).append(meta)

    logger.info(f"Successfully created observation collection with {len(observation_sets)} set(s)")

    # Flatten model settings list — factories may return a single model or a list of models,
    # and we accumulate them by appending, producing a nested list [[m1, m2], [m3], ...].
    flat_models = []
    for ms in model_setting:
        if isinstance(ms, (list, tuple)):
            flat_models.extend(ms)
        else:
            flat_models.append(ms)

    return total_collection, flat_models, dataset_metadata
