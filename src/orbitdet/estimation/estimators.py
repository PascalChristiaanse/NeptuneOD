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
    # `initial_states` returns a sequence of EstimatableParameterSettings;
    # ensure we start with a flat list instead of nesting the sequence.
    estimated_parameters = list(param_setup.initial_states(prop_settings, bodies))
    for parameters in cfg.estimation.parameters_to_estimate:
        # Handle both string entries ("initial_state") and OmegaConf dict entries
        # ({"initial_state": {"apriori": [1.0e7, 1.0e2]}})
        if isinstance(parameters, DictConfig):
            param_name = next(iter(parameters.keys()))
        else:
            param_name = parameters

        match param_name:
            case "initial_state":
                continue  # initial state is already added as a group parameter
            case "iau_rotation_model_pole":
                estimated_parameters.append(param_setup.iau_rotation_model_pole("Neptune"))
            case "neptune_GM":
                estimated_parameters.append(param_setup.gravitational_parameter("Neptune"))
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
