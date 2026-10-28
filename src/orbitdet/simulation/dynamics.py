import logging

from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)


def get_dynamical_model(
    cfg: DictConfig, ctx: RuntimeContext, bodies: env.SystemOfBodies
) -> dict[str, dict[str, list[prop_setup.acceleration.AccelerationSettings]]]:
    """Create a TudatPy DynamicalModel based on the configuration settings.

    Args:
        cfg (DictConfig): Configuration settings for the simulation
        ctx (RuntimeContext): Runtime context containing initialized resources

    Returns:
        Configured dynamical model for propagation
    """
    accelerations = {}
    for affected_body in cfg.bodies_to_propagate.keys():
        if affected_body not in cfg.bodies_to_create:
            raise RuntimeError(
                f"Cannot propagate {affected_body} because it is not defined in bodies_to_create"
            )
        for perturbing_body, settings in cfg.bodies_to_create.items():
            if affected_body == perturbing_body:
                logger.debug(
                    f"Skipping self-acceleration for {affected_body} (perturbing body is the same)"
                )
                continue  # skip self-acceleration

            gravity_type = getattr(settings, "gravity", None)
            if gravity_type is None:
                continue  # skip bodies with no explicit gravity model

            match gravity_type:
                case "central":
                    accelerations.setdefault(affected_body, {})[perturbing_body] = [
                        prop_setup.acceleration.point_mass_gravity()
                    ]
                case "Jacobson2009":
                    if perturbing_body != "Neptune":
                        raise RuntimeError(
                            f"""Gravity model 'Jacobson2009' is only defined for Neptune, 
                            but {affected_body} is set to use it"""
                        )
                    accelerations.setdefault(affected_body, {})[perturbing_body] = [
                        prop_setup.acceleration.spherical_harmonic_gravity(4, 0),
                    ]

        # Non-gravity accelerations requested for this propagated body.
        body_cfg = cfg.bodies_to_propagate[affected_body]
        central_body = body_cfg.central_body
        for acceleration in _build_configured_accelerations(affected_body, body_cfg):
            accelerations.setdefault(affected_body, {}).setdefault(central_body, []).append(
                acceleration
            )

    central_bodies = [settings.central_body for body, settings in cfg.bodies_to_propagate.items()]
    return prop_setup.create_acceleration_models(
        bodies, accelerations, list(cfg.bodies_to_propagate.keys()), central_bodies
    )


def _build_configured_accelerations(
    affected_body: str, body_cfg: DictConfig
) -> list[prop_setup.acceleration.AccelerationSettings]:
    """Build extra (non-gravity) accelerations configured for a propagated body.

    Configuration lives on the propagated body under an ``accelerations`` list.
    Gravity-derived accelerations are handled separately by the caller.

    Currently supported entries:

    - ``direct_tidal_dissipation_time_lag``: Lainey et al. style direct tidal
      dissipation parameterised by the tidal time lag (``time_lag``).
    - ``direct_tidal_dissipation_inv_quality``: the same model parameterised by
      the inverse tidal quality factor and tidal period
      (``inverse_tidal_quality_factor`` + ``tidal_period``). This variant is
      mandatory if the dissipation is to be estimated through
      ``inverse_tidal_quality_factor`` (see
      :func:`orbitdet.estimation.estimators.get_estimatable_parameter_settings`),
      because the corresponding estimatable parameter reads 1/Q and the tidal
      period straight from the acceleration model.

    Args:
        affected_body: Name of the body the accelerations act on.
        body_cfg: The propagated body's configuration node.

    Returns:
        List of configured :class:`AccelerationSettings`.

    Raises:
        ValueError: If an unknown acceleration type is requested, or a required
            field is missing from an acceleration entry.
    """
    configured = body_cfg.get("accelerations")
    if configured is None:
        return []

    built: list[prop_setup.acceleration.AccelerationSettings] = []
    for entry in configured:
        accel_type = entry.get("type")
        try:
            match accel_type:
                case "direct_tidal_dissipation_time_lag":
                    use_tide_raised_on_planet = bool(entry.get("use_tide_raised_on_planet", True))
                    include_direct_radial_component = bool(
                        entry.get("include_direct_radial_component", False)
                    )
                    k2 = float(entry["k2_love_number"])
                    time_lag = float(entry["time_lag"])
                    logger.info(
                        f"Adding direct tidal dissipation on {affected_body} "
                        f"(time_lag={time_lag}s, k2={k2}, "
                        f"tide_on_planet={use_tide_raised_on_planet})"
                    )
                    built.append(
                        prop_setup.acceleration.direct_tidal_dissipation_acceleration(
                            k2,
                            time_lag,
                            include_direct_radial_component,
                            use_tide_raised_on_planet,
                        )
                    )
                case "direct_tidal_dissipation_inv_quality":
                    use_tide_raised_on_planet = bool(entry.get("use_tide_raised_on_planet", True))
                    include_direct_radial_component = bool(
                        entry.get("include_direct_radial_component", False)
                    )
                    k2 = float(entry["k2_love_number"])
                    inv_q = float(entry["inverse_tidal_quality_factor"])
                    tidal_period = float(entry["tidal_period"])
                    logger.info(
                        f"Adding direct tidal dissipation on {affected_body} "
                        f"(1/Q={inv_q}, T={tidal_period}s, k2={k2}, "
                        f"tide_on_planet={use_tide_raised_on_planet})"
                    )
                    built.append(
                        prop_setup.acceleration.direct_tidal_dissipation_acceleration(
                            k2,
                            inv_q,
                            tidal_period,
                            include_direct_radial_component,
                            use_tide_raised_on_planet,
                        )
                    )
                case _:
                    raise ValueError(
                        f"Unknown acceleration type '{accel_type}' for {affected_body}."
                    )
        except KeyError as exc:
            raise ValueError(
                f"Acceleration of type '{accel_type}' for {affected_body} requires field {exc}."
            ) from exc

    return built
