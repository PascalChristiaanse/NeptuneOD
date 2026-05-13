import logging

from omegaconf import DictConfig
from tudatpy.dynamics import environment as env
from tudatpy.dynamics import parameters_setup as param_setup
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext

logger = logging.getLogger(__name__)


def get_estimatable_parameters(
    cfg: DictConfig,
    ctx: RuntimeContext,
    prop_settings: prop_setup.propagator.PropagatorSettings,
    bodies: env.SystemOfBodies,
):
    # `initial_states` returns a sequence of EstimatableParameterSettings;
    # ensure we start with a flat list instead of nesting the sequence.
    estimated_parameters = list(param_setup.initial_states(prop_settings, bodies))
    for param in cfg.estimation.parameters_to_estimate:
        match param:
            case "initial_state":
                continue  # initial state is already added as a group parameter
            case "iau_rotation_model_pole":
                estimated_parameters.append(param_setup.iau_rotation_model_pole("Neptune"))
            case _:
                raise ValueError(f"Unknown parameter {param} specified for estimation")

    return param_setup.create_parameter_set(estimated_parameters, bodies, prop_settings)


# def get_a_priori_covariance_matrix(
#     cfg: DictConfig, ctx: RuntimeContext, parameter_set: param_setup.ParameterSet
# ):
#     pass


# def get_estimation_settings(
#     cfg: DictConfig,
#     ctx: RuntimeContext,
#     prop_settings: prop_setup.propagator.PropagatorSettings,
#     bodies: env.SystemOfBodies,
#     parameter_set: param_setup.ParameterSet,
# ):
#     estimator = estimation_analysis.Estimator(
#         bodies,
#         parameter_set,
#         pseudo_observations_settings,
#         propagator_settings,
#     )
