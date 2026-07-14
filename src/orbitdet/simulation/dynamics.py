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
                    logger.warning("Fix gravity")
                    accelerations.setdefault(affected_body, {})[perturbing_body] = [
                        prop_setup.acceleration.point_mass_gravity(),
                        prop_setup.acceleration.spherical_harmonic_gravity(4, 0),
                    ]

    central_bodies = [settings.central_body for body, settings in cfg.bodies_to_propagate.items()]
    return prop_setup.create_acceleration_models(
        bodies, accelerations, list(cfg.bodies_to_propagate.keys()), central_bodies
    )
