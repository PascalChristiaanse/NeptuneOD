from omegaconf import DictConfig
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext


def get_integrator_settings(
    cfg: DictConfig, ctx: RuntimeContext
) -> prop_setup.integrator.IntegratorSettings:
    match cfg.integrator.type:
        case "RKF78":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_78,
            )
        case _:
            raise ValueError(
                f"Unknown integrator type {cfg.integrator.type} specified in configuration"
            )
