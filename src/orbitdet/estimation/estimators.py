import logging

from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import parameters as param
from tudatpy.dynamics import parameters_setup as param_setup
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)


def get_estimatable_parameter_settings(
    cfg: DictConfig,
    ctx: RuntimeContext,
    prop_settings: prop_setup.propagator.PropagatorSettings,
    bodies: env.SystemOfBodies,
) -> list[param.EstimatableParameter]:
    parameters_to_estimate = cfg.estimation.parameters_to_estimate
    estimated_parameters: list[param.EstimatableParameter] = []
    if parameters_to_estimate.get("initial_state", False):
        logger.warning(
            """Initial state must be estimated for the propagator to work correctly."""
            """ Adding it anyway..."""
        )
    estimated_parameters.extend(param_setup.initial_states(prop_settings, bodies))

    logger.info(f"Parameters to estimate ({len(estimated_parameters)})")
    for param_name, should_estimate in parameters_to_estimate.items():
        if not should_estimate or param_name == "initial_state":
            continue

        match param_name:
            case "iau_rotation_model_pole":
                estimated_parameters.append(param_setup.iau_rotation_model_pole("Neptune"))
                logger.info("\t - IAU rotation model pole")
            case "iau_rotation_model_pole_rate":
                estimated_parameters.append(param_setup.iau_rotation_model_pole_rate("Neptune"))
                logger.info("\t - IAU rotation model pole rate")
            case "iau_rotation_model_pole_librations":
                freqs: list[float] = []
                estimated_parameters.append(
                    param_setup.iau_rotation_model_pole_librations("Neptune", freqs)
                )
                logger.info(f"\t - IAU rotation model pole librations (frequencies: {freqs})")
            case "neptune_GM":
                estimated_parameters.append(param_setup.gravitational_parameter("Neptune"))
                logger.info("\t - Neptune gravitational parameter")
            case "triton_GM":
                estimated_parameters.append(param_setup.gravitational_parameter("Triton"))
                logger.info("\t - Triton gravitational parameter")
            case "neptune_j2_j4":
                block_indices = [
                    (2, 0),  # C20 (J2)
                    (4, 0),  # C40 (J4)
                ]

                # Create the estimatable parameter for these specific coefficients
                estimated_parameters.append(
                    param_setup.spherical_harmonics_c_coefficients_block(
                        body="Neptune", block_indices=block_indices
                    )
                )
                logger.info("\t - Neptune J2 and J4 spherical harmonics coefficients")
            case _:
                raise ValueError(f"Unknown parameter {param_name} specified for estimation")

    return estimated_parameters


def get_estimatable_parameters(
    cfg: DictConfig,
    ctx: RuntimeContext,
    prop_settings: prop_setup.propagator.PropagatorSettings,
    bodies: env.SystemOfBodies,
) -> param.EstimatableParameterSet:
    estimated_parameters = get_estimatable_parameter_settings(cfg, ctx, prop_settings, bodies)
    return param_setup.create_parameter_set(estimated_parameters, bodies, prop_settings)
