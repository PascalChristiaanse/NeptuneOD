import logging

from omegaconf import DictConfig
from tudatpy.dynamics import propagation_setup as prop_setup

from orbitdet.reproducibility.runtime import RuntimeContext

#: Extra keyword arguments required by the Bulirsch-Stoer integrators, keyed by
#: integrator type. All other types are fully determined by the timestep.
_BULIRSCH_STOER_STEPS = {
    "BulirschStoer6": 6,
    "BulirschStoer8": 8,
    "BulirschStoer10": 10,
}

logger = logging.getLogger(__name__)


def build_integrator_settings(
    timestep: float,
    integrator_type: str,
    cfg: DictConfig | None = None,
) -> prop_setup.integrator.IntegratorSettings:
    """Create fixed-step integrator settings for an arbitrary timestep and type.

    Unlike :func:`get_integrator_settings`, the timestep is an explicit argument
    rather than being read from ``cfg.integrator.fixed_step_size``. This makes
    the factory usable from validation scripts that propagate the same model at
    several step sizes (e.g. Richardson extrapolation or a reference run).

    Parameters
    ----------
    timestep : float
        Fixed integration step [s]. Its sign selects forward (positive) or
        backward (negative) propagation.
    integrator_type : str
        Integrator name, e.g. ``"RKF78"``, ``"RK4"``, ``"BulirschStoer6"``.
    cfg : DictConfig | None
        Optional configuration, only consulted for
        ``integrator.maximum_number_of_steps`` when ``integrator_type`` is the
        generic ``"BulirschStoer"`` entry. May be omitted for all other types.

    Returns
    -------
    prop_setup.integrator.IntegratorSettings
        Integrator settings ready to be passed to a propagator.

    Raises
    ------
    ValueError
        If ``integrator_type`` is not recognised.
    """
    maximum_number_of_steps = None
    if cfg is not None:
        maximum_number_of_steps = cfg.get("integrator", {}).get("maximum_number_of_steps")

    match integrator_type:
        case "RKF78":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_78,
            )
        case "RK4":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rk_4,
            )
        case "Euler":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.euler_forward,
            )
        case "RK3":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rk_3,
            )
        case "Ralston4":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.ralston_4,
            )
        case "RKF89":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_89,
            )
        case "RKV89":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkv_89,
            )
        case "RKF108":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_108,
            )
        case "RKF1210":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_1210,
            )
        case "RKF1412":
            return prop_setup.integrator.runge_kutta_fixed_step(
                timestep,
                coefficient_set=prop_setup.integrator.CoefficientSets.rkf_1412,
            )
        case "BulirschStoer6" | "BulirschStoer8" | "BulirschStoer10":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                timestep,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=_BULIRSCH_STOER_STEPS[integrator_type],
            )
        case "BulirschStoer":
            if maximum_number_of_steps is None:
                raise ValueError(
                    "Integrator 'BulirschStoer' requires 'integrator.maximum_number_of_steps' "
                    "in the configuration."
                )
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                timestep,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=maximum_number_of_steps,
            )
        case _:
            raise ValueError(f"Unknown integrator type: {integrator_type}")


def get_integrator_settings(
    cfg: DictConfig, ctx: RuntimeContext
) -> prop_setup.integrator.IntegratorSettings:
    """Create integrator settings from the configuration.

    Thin wrapper around :func:`build_integrator_settings` that reads the
    timestep and type from ``cfg.integrator``.
    """
    logger.info(
        f"Creating integrator settings: type={cfg.integrator.type}, "
        f"fixed_step_size={cfg.integrator.fixed_step_size}"
    )
    if getattr(cfg.integrator, "maximum_number_of_steps", None) is not None:
        logger.info(f"Integrator maximum_number_of_steps={cfg.integrator.maximum_number_of_steps}")
    return build_integrator_settings(
        cfg.integrator.fixed_step_size,
        cfg.integrator.type,
        cfg,
    )
