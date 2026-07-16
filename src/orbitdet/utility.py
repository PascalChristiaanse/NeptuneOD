"""Copyright (c) 2010-2023, Delft University of Technology All rights reserved.

This file is part of the Tudat. Redistribution and use in source and binary forms, with or without
modification, are permitted exclusively under the terms of the Modified BSD license. You should have
received a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from tudatpy.dynamics.environment_setup import create_system_of_bodies, get_default_body_settings
from tudatpy.dynamics.propagation import SimulationResults
from tudatpy.dynamics.propagation.dependent_variable_dictionary import DependentVariableDictionary
from tudatpy.dynamics.propagation_setup.dependent_variable import (
    get_dependent_variable_shape,
)
from tudatpy.util import result2array


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
