import logging

from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.interface import spice

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)


def get_propagator_settings(
    cfg: DictConfig,
    ctx: RuntimeContext,
    acceleration_settings: env.SystemOfBodies,
    integrator_settings: prop_setup.integrator.IntegratorSettings,
    dependent_variables_to_save: list[
        prop_setup.dependent_variable.SingleAccelerationDependentVariableSaveSettings
    ],
) -> prop_setup.propagator.TranslationalStatePropagatorSettings:
    """Create TudatPy propagator settings based on the configuration settings.

    Args:
        cfg (DictConfig): Configuration settings for the simulation
        ctx (RuntimeContext): Runtime context containing initialized resources
        bodies (env.SystemOfBodies): System of bodies in the simulation

    Returns:
        Configured propagator settings for propagation
    """

    initial_state = []
    for body, settings in cfg.bodies_to_propagate.items():
        if settings.initial_state is None:
            # emit warning if initial state is not defined and use default state from SPICE kernels
            logger.warning(
                f"""Initial state for {body} is not defined in configuration."""
                """Using default state from SPICE kernels."""
            )

            state = spice.get_body_cartesian_state_at_epoch(
                target_body_name=body,
                observer_body_name=settings.central_body,
                reference_frame_name=cfg.global_frame_orientation,
                aberration_corrections="none",
                ephemeris_time=ctx.start_epoch,
            )
            initial_state.extend(state)
        else:
            initial_state.extend(settings.initial_state)

    # Create termination settings
    termination_condition_end = prop_setup.propagator.time_termination(ctx.end_epoch)
    termination_condition_start = prop_setup.propagator.time_termination(ctx.start_epoch)
    termination_settings = prop_setup.propagator.non_sequential_termination(
        termination_condition_end, termination_condition_start
    )

    central_bodies = [settings.central_body for body, settings in cfg.bodies_to_propagate.items()]
    return prop_setup.propagator.translational(
        central_bodies,
        acceleration_settings,
        list(cfg.bodies_to_propagate.keys()),
        initial_state,
        ctx.start_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables_to_save,
    )
