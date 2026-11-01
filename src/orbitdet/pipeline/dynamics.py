"""Building propagator settings and observation models.

The estimation stage needs three things built from config plus a collection:
the acceleration model, the propagator settings, and the observation model
settings. The propagation-only stage (used for pre/post-fit residuals) needs
the first two.

Bundling them here keeps the two callers from duplicating the construction
order, which matters because the propagator settings depend on the environment
*and* on the estimated state's initial epoch.
"""

from __future__ import annotations

import logging

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tudatpy.estimation import observations as obs
from tudatpy.estimation.observations_setup import observations_simulation_settings as obs_sim_setup

from orbitdet.pipeline.observation_models import rebuild_observation_models
from orbitdet.simulation import (
    get_dynamical_model,
    get_integrator_settings,
    get_propagator_settings,
)

logger = logging.getLogger(__name__)


def _dependent_variable_settings():
    """Build the dependent-variable settings objects for the propagation.

    The first entry is the relative position of the ephemeris body with respect
    to the propagated body, which the RSW figures use. It must stay first so the
    figures keep referring to the same quantity.
    """
    import tudatpy.dynamics.propagation_setup as prop_setup

    return [
        prop_setup.dependent_variable.relative_position("Triton Spice", "Triton"),
        prop_setup.dependent_variable.keplerian_state("Triton", "Neptune"),
        prop_setup.dependent_variable.keplerian_state("Triton Spice", "Neptune"),
    ]


def build_dynamics(cfg: DictConfig, ctx, bodies):
    """Build acceleration and propagator settings.

    Returns
    -------
    tuple
        ``(propagator_settings, dependent_variables)``.
    """
    accelerations = get_dynamical_model(cfg, ctx, bodies)
    integrator = get_integrator_settings(cfg, ctx)
    dependent_variables = _dependent_variable_settings()

    propagator_settings = get_propagator_settings(
        cfg, ctx, accelerations, integrator, dependent_variables_to_save=dependent_variables
    )
    logger.info("Dynamical model and propagator settings built.")
    return propagator_settings, dependent_variables


def build_propagator(cfg: DictConfig, ctx, bodies, observations: obs.ObservationCollection):
    """Build propagator settings *and* observation models from a collection.

    Returns
    -------
    tuple
        ``(propagator_settings, observation_models)``.
    """
    propagator_settings, _ = build_dynamics(cfg, ctx, bodies)
    observation_models = rebuild_observation_models(observations)
    return propagator_settings, observation_models


def build_observation_simulators(bodies, observations: obs.ObservationCollection):
    """Build observation simulators from a collection's preserved structure."""
    observation_models = rebuild_observation_models(observations)
    return obs_sim_setup.create_observation_simulators(observation_models, bodies)


def propagate(cfg: DictConfig, ctx, bodies, propagator_settings, initial_states=None):
    """Run a dynamics simulation and return the simulator.

    Parameters
    ----------
    initial_states:
        When given, overrides the propagator's initial state. Used by the
        post-fit stage to propagate the *estimated* state.
    """
    from tudatpy.dynamics import simulator as sim

    if initial_states is not None:
        propagator_settings.initial_states = np.asarray(initial_states)
        logger.info("Overrode initial state with %d value(s).", len(initial_states))

    return sim.create_dynamics_simulator(bodies, propagator_settings)


def state_history_arrays(propagation_results) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(epochs, states)`` from a propagation result's state history."""
    history = propagation_results.state_history
    epochs = np.array(list(history.keys()))
    states = np.array(list(history.values()))
    return epochs, states


def covariance_output_times(cfg: DictConfig, ctx) -> np.ndarray:
    """Return the epochs at which the covariance is propagated.

    Both directions from the **initial** epoch, so the formal errors cover the
    full arc. Using ``start_epoch`` here instead would evaluate the state
    transition matrix tens of years outside the fitted arc and inflate the
    formal errors by many orders of magnitude.
    """
    start = float(ctx.initial_epoch.to_float())
    end = float(ctx.end_epoch.to_float())
    step_days = OmegaConf.select(cfg, "propagation.step_days", default=10.0)
    return np.arange(start, end, step_days * 86400.0)
