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
    initial_state: list[float] | None = None,
) -> prop_setup.propagator.TranslationalStatePropagatorSettings:
    """Create TudatPy propagator settings based on the configuration settings.

    Args:
        cfg (DictConfig): Configuration settings for the simulation
        ctx (RuntimeContext): Runtime context containing initialized resources
        acceleration_settings (env.SystemOfBodies): System of bodies in the simulation
        integrator_settings (prop_setup.integrator.IntegratorSettings): Integrator settings
        dependent_variables_to_save (list): Dependent variables to save during propagation
        initial_state (list[float] | None): Optional explicit initial state for the
            propagated bodies. When provided, it overrides the per-body
            ``initial_state`` entries in the configuration. When ``None``, the initial
            state is taken from the configuration, falling back to the SPICE ephemeris
            state for any body whose ``initial_state`` is not defined.

    Returns:
        Configured propagator settings for propagation
    """

    if initial_state is None:
        initial_state = []
        for body, settings in cfg.bodies_to_propagate.items():
            if settings.initial_state is None:
                # emit warning if initial state is not defined and use default state
                # from SPICE kernels
                logger.warning(
                    f"""Initial state for {body} is not defined in configuration."""
                    """Using default state from SPICE kernels."""
                )

                state = spice.get_body_cartesian_state_at_epoch(
                    target_body_name=body,
                    observer_body_name=settings.central_body,
                    reference_frame_name=cfg.global_frame_orientation,
                    aberration_corrections="none",
                    ephemeris_time=ctx.initial_epoch,
                )
                logger.info(
                    "Initial state for %s taken from SPICE ephemeris (central body: %s, "
                    "frame: %s, epoch: %s): %s",
                    body,
                    settings.central_body,
                    cfg.global_frame_orientation,
                    ctx.initial_epoch,
                    list(state),
                )
                initial_state.extend(state)
            else:
                logger.info(
                    "Initial state for %s taken from configuration (central body: %s): %s",
                    body,
                    settings.central_body,
                    list(settings.initial_state),
                )
                initial_state.extend(settings.initial_state)
    else:
        logger.info(
            "Using explicitly provided initial state for bodies %s: %s",
            list(cfg.bodies_to_propagate.keys()),
            list(initial_state),
        )

    logger.info(
        "Propagator initial state (bodies: %s, epoch: %s): %s",
        list(cfg.bodies_to_propagate.keys()),
        ctx.initial_epoch,
        list(initial_state),
    )

    # Create termination settings
    termination_condition_end = prop_setup.propagator.time_termination(ctx.end_epoch)
    termination_condition_start = prop_setup.propagator.time_termination(ctx.start_epoch)
    termination_settings = prop_setup.propagator.non_sequential_termination(
        termination_condition_end, termination_condition_start
    )

    if cfg.propagation.set_integrated_result:
        logger.info("Propagator settings will set the integrated result in the runtime context.")

    central_bodies = [settings.central_body for body, settings in cfg.bodies_to_propagate.items()]
    settings = prop_setup.propagator.translational(
        central_bodies,
        acceleration_settings,
        list(cfg.bodies_to_propagate.keys()),
        initial_state,
        ctx.initial_epoch,
        integrator_settings,
        termination_settings,
        output_variables=dependent_variables_to_save,
    )
    settings.processing_settings.set_integrated_result = cfg.propagation.set_integrated_result
    return settings
