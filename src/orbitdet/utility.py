"""Copyright (c) 2010-2023, Delft University of Technology All rights reserved.

This file is part of the Tudat. Redistribution and use in source and binary forms, with or without
modification, are permitted exclusively under the terms of the Modified BSD license. You should have
received a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import logging
import re
from pathlib import Path

from tudatpy.dynamics.environment_setup import create_system_of_bodies, get_default_body_settings
from tudatpy.dynamics.propagation import SimulationResults
from tudatpy.dynamics.propagation.dependent_variable_dictionary import DependentVariableDictionary
from tudatpy.dynamics.propagation_setup.dependent_variable import (
    get_dependent_variable_shape,
)
from tudatpy.estimation import observations as obs
from tudatpy.util import result2array

logger = logging.getLogger(__name__)


def create_dependent_variable_dictionary(
    propagation_results: SimulationResults,
) -> DependentVariableDictionary:
    """Construct a dictionary-like object (`DependentVariableDictionary`) which maps which maps
    dependent variables to their time histories. See the documentation of
    `DependentVariableDictionary` to learn more about how time histories are saved, and how the time
    history of a given dependent variable can be retrieved.

    NOTE: DOES NOT WORK WITH:
        - vehicle_panel_body_fixed_surface_normals
        - vehicle_surface_panel_radiation_pressure_force
        - paneled_radiation_source_geometry
        - illuminated_panel_fraction
        - full_body_paneled_geometry

    Arguments
    ---------
    dynamics_simulator : SimulationResults
        `SimulationResults` (Derived) object containing the results of the numerical propagation

    Returns
    ------
    dependent_variable_dictionary : DependentVariableDictionary
        `DependentVariableDictionary` of propagation
    """
    # --------------------------------------------------------------------
    # %% RETRIEVE DEPENDENT VARIABLE DATA
    # --------------------------------------------------------------------

    # Retrieve dependent variable settings objects
    dependent_variable_settings = propagation_results.ordered_dependent_variable_settings

    # Retrieve /transposed/ time and dependent variable histories
    time_history = result2array(propagation_results.dependent_variable_history).T[0, :]
    dependent_variable_history = result2array(propagation_results.dependent_variable_history).T[
        1:, :
    ]

    # Calculate total number of epochs of propagation
    n = len(time_history)

    # --------------------------------------------------------------------
    # %% CONSTRUCT DEPENDENT VARIABLE DICTIONARY
    # --------------------------------------------------------------------

    # Create degenerate system of bodies to retrieve dependent variable shapes
    bodies = create_system_of_bodies(get_default_body_settings(["Sun"]))

    # Construct dependent variable matrices
    dependent_variable_matrices = []
    for (i, m), dependent_variable in dependent_variable_settings.items():
        # Retrieve dependent variable shape
        A, B = get_dependent_variable_shape(dependent_variable, bodies)

        # Save dependent variable history as a tensor of (A, B)-sized
        # matrices with `n` entries, where `n` is the number of epochs
        dependent_variable_matrices.append(
            dependent_variable_history[
                # From index i to index i+m (the flattened dimension of the dependent variable)
                i : i + m,
                :,
            ].T.reshape((n, A, B))
        )

    # Construct dependent variable dictionary
    dependent_variable_dictionary = DependentVariableDictionary(
        {
            dependent_variable: {
                epoch: dependent_variable_matrices[i_depv][i_epoch].squeeze()
                for i_epoch, epoch in enumerate(time_history)
            }
            for i_depv, dependent_variable in enumerate(dependent_variable_settings.values())
        }
    )

    return dependent_variable_dictionary


def save_tudat_object(obj: object, file_path: str | Path) -> Path:
    """Serialize a TudatPy object to a binary ``.tudat`` file.

    Uses the object's native ``save_to_binary`` method.  Only objects that
    provide this native serialization are supported (e.g.
    :class:`~tudatpy.estimation.observations.ObservationCollection`).

    Parameters
    ----------
    obj : object
        TudatPy object with a native ``save_to_binary`` method.
    file_path : str | Path
        Target file path.  If the path does not end in ``.tudat`` the
        extension is appended automatically.

    Returns
    -------
    Path
        The resolved path the object was saved to.

    Raises
    ------
    TypeError
        If *obj* does not have a ``save_to_binary`` method.
    """
    if not hasattr(obj, "save_to_binary"):
        raise TypeError(
            f"Object of type {type(obj).__name__} does not provide a native save_to_binary method."
        )

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj.save_to_binary(str(path))
    return path


def select_observation_collection_by_tag(
    observations: obs.ObservationCollection,
    dataset_metadata: dict[str, list[dict]],
    tag: str,
    exclude: bool = False,
) -> obs.ObservationCollection | None:
    """Build an ObservationCollection containing only the sets matching *tag*.

    Uses the ``index`` bookkeeping in ``dataset_metadata`` (populated by
    :func:`orbitdet.observations.create_observation_collection`) to retrieve the
    matching sets from ``observations``.  The *tag* is matched case-insensitively
    against the ``dataset_id`` field of each metadata entry, so any dataset can be
    selected (e.g. ``"gaia"`` for Gaia, ``"voyager"`` for Voyager 2).

    This is useful when a plot only applies to a subset of the collection — for
    example the scan-frame plots
    (:class:`~orbitdet.visualization.ResidualsScan`) require the archived Gaia
    ``position_angle_scan``, which is only available for Gaia observations.

    Parameters
    ----------
    observations : obs.ObservationCollection
        The full observation collection.
    dataset_metadata : dict[str, list[dict]]
        Metadata mapping produced by ``create_observation_collection``.  Each
        entry must contain a ``dataset_id`` and an ``index`` field.
    tag : str
        Substring to match (case-insensitively) against ``dataset_id``.
    exclude : bool, optional
        When ``True``, return the sets that do *not* match *tag* instead of the
        ones that do.  Defaults to ``False``.

    Returns
    -------
    obs.ObservationCollection | None
        A collection holding only the matching (or, with ``exclude=True``, the
        non-matching) sets, or ``None`` when no set qualifies.
    """
    all_sets = observations.get_single_observation_sets()
    tag_lower = tag.lower()
    selected_sets: list[obs.SingleObservationSet] = []
    for metas in dataset_metadata.values():
        for meta in metas:
            matches = tag_lower in str(meta.get("dataset_id", "")).lower()
            if matches == exclude:
                continue
            index = meta.get("index")
            if index is None or not (0 <= index < len(all_sets)):
                logger.warning(
                    "Metadata entry for tag %r has invalid index %r; skipping.", tag, index
                )
                continue
            selected_sets.append(all_sets[index])
    if not selected_sets:
        return None
    return obs.ObservationCollection(selected_sets)


#: Seconds per unit for :func:`parse_duration_to_seconds`. Month and year use
#: their mean Gregorian lengths (365.2425 days per year, 12 months per year),
#: so a duration such as ``"6 months"`` maps to a deterministic number of
#: seconds independent of the calendar date.
_DURATION_UNITS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "w": 604800.0,
    "week": 604800.0,
    "weeks": 604800.0,
    "mo": 2629746.0,
    "month": 2629746.0,
    "months": 2629746.0,
    "y": 31556952.0,
    "yr": 31556952.0,
    "yrs": 31556952.0,
    "year": 31556952.0,
    "years": 31556952.0,
}

#: Matches one ``<number><unit>`` term, e.g. ``"6 months"`` or ``"1.5h"``.
_DURATION_TERM = re.compile(r"([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)")


def parse_duration_to_seconds(value: float | int | str) -> float:
    """Parse a duration into seconds.

    Accepts either a plain number (already in seconds) or a human-readable
    string such as ``"6 months"``, ``"180 days"``, ``"1 year"``, ``"12 h"``
    or ``"3600 s"``. Compound strings (``"1 year 6 months"``) are summed.

    Month and year are interpreted with their mean Gregorian lengths
    (365.2425 days per year, 12 months per year), so the result is
    deterministic and independent of the calendar date.

    Parameters
    ----------
    value : float | int | str
        Duration to parse. Numbers are interpreted as seconds.

    Returns
    -------
    float
        Duration in seconds.

    Raises
    ------
    TypeError
        If *value* is a boolean or an unsupported type.
    ValueError
        If a string cannot be parsed or contains an unknown unit.
    """
    if isinstance(value, bool):
        raise TypeError("A boolean is not a valid duration.")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Cannot parse duration from {type(value).__name__!r}.")

    text = value.strip().lower()
    if not text:
        raise ValueError("Cannot parse an empty duration string.")

    # A bare number is interpreted as seconds.
    try:
        return float(text)
    except ValueError:
        pass

    terms = _DURATION_TERM.findall(text)
    if not terms:
        raise ValueError(f"Could not parse duration {value!r}.")

    # Reject leftover characters (e.g. a typo'd unit) so mistakes are loud.
    consumed = "".join(f"{number}{unit}" for number, unit in terms)
    if re.sub(r"[\s,]", "", text) != consumed:
        raise ValueError(f"Could not parse duration {value!r}.")

    total = 0.0
    for number, unit in terms:
        if unit not in _DURATION_UNITS:
            raise ValueError(
                f"Unknown duration unit {unit!r} in {value!r}. "
                f"Known units: {sorted(set(_DURATION_UNITS))}."
            )
        total += float(number) * _DURATION_UNITS[unit]
    return total
