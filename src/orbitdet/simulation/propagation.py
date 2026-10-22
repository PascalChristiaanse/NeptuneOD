import logging

from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.interface import spice
from tudatpy.math import interpolators

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)

#: Lagrange order used by TudatPy when no integrator-result interpolator is
#: configured. The default lives in the C++ ``PropagatorProcessingSettings``
#: constructor; keeping it here lets the project detect and log when it is in
#: effect.
TUDATPY_DEFAULT_INTERPOLATOR_ORDER = 6


def build_integrator_result_interpolator_settings(
    interpolator_order: int,
) -> interpolators.LagrangeInterpolatorSettings:
    """Build the Lagrange interpolator settings used for the integrator result.

    This is the interpolator behind ``processing_settings.interpolator_settings``,
    i.e. the one that turns the sampled state history into the propagated body's
    ephemeris whenever ``set_integrated_result`` is enabled. That ephemeris is
    what the observation models evaluate to build simulated observations, so its
    order directly fixes the accuracy of synthetic observations — and it is what
    the orbit determination manager reads (it forces ``set_integrated_result`` on
    internally).

    The boundary handling mirrors the C++ default
    (``lagrange_cubic_spline_boundary_interpolation`` and
    ``throw_exception_at_boundary``): the comparison faces a hard failure when
    asked to interpolate outside the propagated arc, rather than silently
    extrapolating a polynomial.

    Parameters
    ----------
    interpolator_order : int
        Lagrange interpolator order (>= 1). Must not exceed the number of sample
        epochs minus one, which the propagation enforces implicitly.

    Returns
    -------
    interpolators.LagrangeInterpolatorSettings
        Settings ready to assign to ``processing_settings.interpolator_settings``.

    Raises
    ------
    ValueError
        If ``interpolator_order`` is below 1.
    """
    if interpolator_order < 1:
        raise ValueError(f"Interpolator order must be at least 1, got {interpolator_order}.")

    return interpolators.LagrangeInterpolatorSettings(
        int(interpolator_order),
        False,
        interpolators.hunting_algorithm,
        interpolators.lagrange_cubic_spline_boundary_interpolation,
        interpolators.throw_exception_at_boundary,
    )


def resolve_interpolator_order(cfg: DictConfig) -> int:
    """Read the integrator-result interpolator order from the configuration.

    Looks up ``propagation.interpolator_order`` and falls back to the TudatPy
    default when the key is absent. A ``None`` value is treated as absent so a
    config that declares the key but leaves it unset behaves like an older
    config that never had it.

    Parameters
    ----------
    cfg : DictConfig
        The resolved experiment configuration.

    Returns
    -------
    int
        The Lagrange order to use for the integrator result.
    """
    propagation = cfg.get("propagation")
    order = None
    if propagation is not None:
        order = propagation.get("interpolator_order")
    if order is None:
        return TUDATPY_DEFAULT_INTERPOLATOR_ORDER
    return int(order)


#: Translational propagator formulations available in TudatPy, keyed by the
#: lower-case name used in configuration files. ``cowell`` propagates Cartesian
#: elements directly, while the other formulations propagate a different set of
#: elements internally (the state history is still reported as Cartesian).
TRANSLATIONAL_PROPAGATORS = {
    name: value
    for name, value in (
        ("cowell", prop_setup.propagator.TranslationalPropagatorType.cowell),
        ("encke", prop_setup.propagator.TranslationalPropagatorType.encke),
        (
            "gauss_keplerian",
            prop_setup.propagator.TranslationalPropagatorType.gauss_keplerian,
        ),
        (
            "gauss_modified_equinoctial",
            prop_setup.propagator.TranslationalPropagatorType.gauss_modified_equinoctial,
        ),
        (
            "unified_state_model_quaternions",
            prop_setup.propagator.TranslationalPropagatorType.unified_state_model_quaternions,
        ),
        (
            "unified_state_model_modified_rodrigues_parameters",
            prop_setup.propagator.TranslationalPropagatorType.unified_state_model_modified_rodrigues_parameters,
        ),
        (
            "unified_state_model_exponential_map",
            prop_setup.propagator.TranslationalPropagatorType.unified_state_model_exponential_map,
        ),
    )
}


def build_translational_propagator_type(
    propagator_type: str,
):
    """Resolve a propagator formulation name to a ``TranslationalPropagatorType``.

    Parameters
    ----------
    propagator_type : str
        Formulation name, e.g. ``"cowell"``, ``"encke"``, ``"gauss_keplerian"``.

    Returns
    -------
    prop_setup.propagator.TranslationalPropagatorType
        The matching enum member.

    Raises
    ------
    ValueError
        If the name is not one of the supported formulations.
    """
    try:
        return TRANSLATIONAL_PROPAGATORS[propagator_type]
    except KeyError:
        known = ", ".join(sorted(TRANSLATIONAL_PROPAGATORS))
        raise ValueError(
            f"Unknown translational propagator '{propagator_type}'. Known: {known}"
        ) from None


def get_propagator_settings(
    cfg: DictConfig,
    ctx: RuntimeContext,
    acceleration_settings: env.SystemOfBodies,
    integrator_settings: prop_setup.integrator.IntegratorSettings,
    dependent_variables_to_save: list[
        prop_setup.dependent_variable.SingleAccelerationDependentVariableSaveSettings
    ],
    initial_state: list[float] | None = None,
    propagator_type: str = "cowell",
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
        propagator_type (str): Translational propagator formulation, one of
            :data:`TRANSLATIONAL_PROPAGATORS` (default ``"cowell"``). The formulation
            only changes the elements in which the equations of motion are integrated;
            the reported state history remains Cartesian.

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
        propagator=build_translational_propagator_type(propagator_type),
        output_variables=dependent_variables_to_save,
    )
    settings.processing_settings.set_integrated_result = cfg.propagation.set_integrated_result

    # The interpolator behind the propagated body's ephemeris. It matters
    # whenever set_integrated_result is on (always, for estimation, since the
    # orbit determination manager forces it): synthetic observations are built
    # by evaluating that ephemeris.
    interpolator_order = resolve_interpolator_order(cfg)
    if interpolator_order == TUDATPY_DEFAULT_INTERPOLATOR_ORDER:
        logger.info(
            "Integrator-result interpolator: Lagrange-%d (TudatPy default). Set "
            "propagation.interpolator_order to override.",
            interpolator_order,
        )
    else:
        logger.info(
            "Integrator-result interpolator: Lagrange-%d (overriding the TudatPy default of %d).",
            interpolator_order,
            TUDATPY_DEFAULT_INTERPOLATOR_ORDER,
        )
    settings.processing_settings.interpolator_settings = (
        build_integrator_result_interpolator_settings(interpolator_order)
    )

    return settings
