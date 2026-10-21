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
        case "RK4":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rk_4,
            )
        case "Euler":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.euler_forward,
            )
        case "RK3":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rk_3,
            )
        case "Ralston4":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.ralston_4,
            )
        case "RKF89":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_89,
            )
        case "RKV89":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkv_89,
            )
        case "RKF108":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_108,
            )
        case "RKF1210":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_1210,
            )
        case "RKF1412":
            return prop_setup.integrator.runge_kutta_fixed_step(
                cfg.integrator.fixed_step_size,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_1412,
            )
        case "BulirschStoer6":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                cfg.integrator.fixed_step_size,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=6,
            )
        case "BulirschStoer8":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                cfg.integrator.fixed_step_size,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=8,
            )
        case "BulirschStoer10":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                cfg.integrator.fixed_step_size,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=10,
            )
        case _:
            raise ValueError(
                f"Unknown integrator type {cfg.integrator.type} specified in configuration"
            )
