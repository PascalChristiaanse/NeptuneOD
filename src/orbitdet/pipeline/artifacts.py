"""Reading and writing the pipeline's tudatpy artifacts.

Every stage boundary in this pipeline is a file that tudatpy serializes
natively. Two details make that less trivial than it looks:

Extra ``.tudat`` suffix
    ``save_to_binary`` appends ``.tudat`` to whatever path it is given, even
    when the path already ends in ``.tudat``. Passing ``"obs.tudat"`` therefore
    writes ``obs.tudat.tudat``. Every helper here normalizes paths so callers
    can always pass the intended final path and get exactly that file.

Collection-level lookups
    The ``ObservationCollection`` stores observations per set, and several
    operations (weighting, outlier rejection) need to know which *dataset*
    a set came from. That mapping is not stored in the collection, so it travels
    alongside it as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import tudatpy.estimation.observations as obs

logger = logging.getLogger(__name__)

#: Suffix tudatpy's ``save_to_binary`` appends to every path it is given.
TUDAT_SUFFIX = ".tudat"


def tudat_stem(path: str | Path) -> str:
    """Return *path* without its ``.tudat`` suffix.

    ``save_to_binary`` always appends ``.tudat``, so this is what must be passed
    to it in order to land on the path the DAG declared.
    """
    path = Path(path)
    return str(path.with_suffix("") if path.suffix == TUDAT_SUFFIX else path)


def save_collection(collection: obs.ObservationCollection, path: str | Path) -> Path:
    """Serialize *collection* to *path*, returning the file actually written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    collection.save_to_binary(tudat_stem(path))
    if not path.exists():
        raise RuntimeError(
            f"Expected {path} after save_to_binary, but it was not created. "
            f"Directory contains: {sorted(p.name for p in path.parent.iterdir())}"
        )
    return path


def load_collection(path: str | Path) -> obs.ObservationCollection:
    """Load an ``ObservationCollection`` written by :func:`save_collection`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No serialized observation collection at {path}")
    collection = obs.ObservationCollection.load_from_binary(tudat_stem(path))
    logger.info(
        "Loaded %d observation set(s) from %s",
        len(collection.get_single_observation_sets()),
        path.name,
    )
    return collection


def save_metadata(metadata: dict, path: str | Path) -> Path:
    """Write ``dataset_metadata.json``.

    The per-set metadata (``dataset_id``, index, coordinate labels) is not
    recoverable from the serialized collection, so it travels beside it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str))
    return path


def load_metadata(path: str | Path) -> dict:
    """Load ``dataset_metadata.json``, tolerating absence."""
    path = Path(path)
    if not path.exists():
        logger.warning("No dataset metadata at %s; continuing without it.", path)
        return {}
    return json.loads(path.read_text())


def load_propagation_results(path: str | Path):
    """Load a serialized propagation result.

    ``SimulationResults.load_from_binary`` is bound per *concrete* class, and it
    rejects a file whose recorded root type does not match the class it was
    called on (``checkRootType`` in ``file_io.h``). A dynamics propagation
    writes a ``SingleArcSimulationResults``, so loading through the abstract
    ``SimulationResults`` fails with::

        The file contains 'SingleArcSimulationResults<...>', but the selected
        load method expects 'SimulationResults<...>'.

    There is no polymorphic loader exposed, so this tries the concrete class
    first and falls back to the others. Keep the list in sync with the
    ``SingleArc``/``MultiArc``/``HybridArc`` bindings in
    ``expose_propagation_results.cpp``.
    """
    from tudatpy.dynamics.propagation import (
        HybridArcSimulationResults,
        HybridArcVariationalSimulationResults,
        MultiArcSimulationResults,
        MultiArcVariationalSimulationResults,
        SingleArcSimulationResults,
        SingleArcVariationalSimulationResults,
    )

    stem = tudat_stem(path)
    # Concrete classes first, then the (rarely correct) variational variants.
    candidates = (
        SingleArcSimulationResults,
        SingleArcVariationalSimulationResults,
        MultiArcSimulationResults,
        MultiArcVariationalSimulationResults,
        HybridArcSimulationResults,
        HybridArcVariationalSimulationResults,
    )

    errors: list[str] = []
    for cls in candidates:
        try:
            results = cls.load_from_binary(stem)
        except RuntimeError as exc:
            # A root-type mismatch means this is the wrong class; keep trying.
            errors.append(f"{cls.__name__}: {exc}")
            continue
        logger.info("Loaded %s from %s", cls.__name__, Path(path).name)
        return results

    raise RuntimeError(
        f"Could not load propagation results from {path} with any known class.\n"
        + "\n".join(errors)
    )
