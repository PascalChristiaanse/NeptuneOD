"""Integrator validation via Richardson extrapolation and forward-backward round-trip.

Two strategies are available (set via ``validation.strategy``):

**richardson** (default)
    Propagate with Δt and Δt/2. The difference between the two propagated
    states at common epochs is taken as an estimate of the local truncation
    error (Richardson extrapolation).

**forward_backward**
    Propagate forward from start to end, then backward from end to start
    using the same timestep Δt. The difference between the forward state
    and the backward-interpolated state at each epoch shows how the error
    accumulates over the round-trip.

Results are saved as NPZ files for later analysis.

Usage
-----
Single run::

    python scripts/validate_integrator.py

Sweep over timesteps and integrator types::

    python scripts/validate_integrator.py --multirun --config-name sweep/validate_integrator
"""

import logging
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tudatpy.astro.time_representation import DateTime, iso_string_to_epoch_time_object
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.dynamics import simulator as sim

from orbitdet.data import KernelManager
from orbitdet.reproducibility import (
    RuntimeContext,
    aim_log_artifact_reference,
    enforce_initialization,
    initialize,
)
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_propagator_settings,
)

logger = logging.getLogger(__name__)


def _make_integrator_settings(timestep: float, integrator_type: str, cfg: DictConfig):
    """Create integrator settings for a given timestep and type.

    This mirrors the logic in ``orbitdet.simulation.get_integrator_settings``
    but allows overriding the timestep at runtime.
    """
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
        case "BulirschStoer":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                timestep,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=cfg.integrator.get("maximum_number_of_steps", 6),
            )
        case "BulirschStoer6":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                timestep,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=6,
            )
        case "BulirschStoer8":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                timestep,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=8,
            )
        case "BulirschStoer10":
            return prop_setup.integrator.bulirsch_stoer_fixed_step(
                timestep,
                extrapolation_sequence=(
                    prop_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
                ),
                maximum_number_of_steps=10,
            )
        case _:
            raise ValueError(f"Unknown integrator type: {integrator_type}")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/validate_integrator",
)
@enforce_initialization
def main(cfg: DictConfig):
    logger.info(f"Starting validate_integrator.py on PID {os.getpid()}")

    # --- Initialization ---
    ctx: RuntimeContext = initialize(cfg)
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)
    logger.info(
        "Validation arc: %s to %s",
        DateTime.from_epoch_time_object(ctx.start_epoch).to_iso_string(),
        DateTime.from_epoch_time_object(ctx.end_epoch).to_iso_string(),
    )

    # --- Load kernels ---
    logger.info("Loading kernels...")
    km = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    logger.info("Kernels loaded.")

    # --- Build environment ---
    bodies = get_environment(cfg, ctx)
    acc = get_dynamical_model(cfg, ctx, bodies)
    logger.info("Environment and dynamics built.")

    # --- Validation parameters ---
    timestep = cfg.validation.timestep
    integrator_type = cfg.integrator.type
    strategy = cfg.validation.get("strategy", "richardson")
    logger.info(
        "Validating integrator '%s' with Δt = %.1f s, strategy = '%s'",
        integrator_type,
        timestep,
        strategy,
    )

    if strategy == "richardson":
        half_timestep = timestep / 2.0
        logger.info("Half timestep for Richardson: %.1f s", half_timestep)

        # --- Propagate with Δt (coarse) ---
        logger.info("Propagating with Δt = %.1f s ...", timestep)
        integ_coarse = _make_integrator_settings(timestep, integrator_type, cfg)
        prop_coarse = get_propagator_settings(cfg, ctx, acc, integ_coarse, [])
        t0_coarse = time.perf_counter()
        result_coarse = sim.create_dynamics_simulator(bodies, prop_coarse)
        t_coarse = time.perf_counter() - t0_coarse
        state_history_coarse = result_coarse.propagation_results.state_history
        logger.info("Coarse propagation done: %d output epochs in %.2f s.", len(state_history_coarse), t_coarse)

        # --- Propagate with Δt/2 (fine) ---
        logger.info("Propagating with Δt/2 = %.1f s ...", half_timestep)
        integ_fine = _make_integrator_settings(half_timestep, integrator_type, cfg)
        prop_fine = get_propagator_settings(cfg, ctx, acc, integ_fine, [])
        t0_fine = time.perf_counter()
        result_fine = sim.create_dynamics_simulator(bodies, prop_fine)
        t_fine = time.perf_counter() - t0_fine
        state_history_fine = result_fine.propagation_results.state_history
        logger.info("Fine propagation done: %d output epochs in %.2f s.", len(state_history_fine), t_fine)

        # --- Build common epoch grid ---
        coarse_epochs = np.array(sorted(state_history_coarse.keys()))
        fine_epochs = np.array(sorted(state_history_fine.keys()))
        fine_states_arr = np.array([state_history_fine[e] for e in fine_epochs])

        # Interpolate fine states to coarse epochs
        states_fine_at_coarse = np.column_stack([
            np.interp(coarse_epochs, fine_epochs, fine_states_arr[:, i])
            for i in range(6)
        ])
        states_coarse_arr = np.array([state_history_coarse[e] for e in coarse_epochs])

        # --- Compute differences (Richardson error estimate) ---
        # Skip the first epoch: the initial state can have a spurious mismatch
        # between coarse and fine propagations due to interpolation artifacts.
        skip_first = 1
        differences = states_coarse_arr[skip_first:] - states_fine_at_coarse[skip_first:]
        pos_diff_norm = np.linalg.norm(differences[:, :3], axis=1)
        vel_diff_norm = np.linalg.norm(differences[:, 3:], axis=1)

        logger.info(
            "Richardson differences computed: position RMS = %.6e m, velocity RMS = %.6e m/s",
            np.sqrt(np.mean(pos_diff_norm**2)),
            np.sqrt(np.mean(vel_diff_norm**2)),
        )

        # --- Save results ---
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        npz_path = output_dir / "validation_results.npz"
        np.savez(
            npz_path,
            epochs=coarse_epochs[skip_first:],
            states_coarse=states_coarse_arr[skip_first:],
            states_fine=states_fine_at_coarse[skip_first:],
            differences=differences,
            pos_diff_norm=pos_diff_norm,
            vel_diff_norm=vel_diff_norm,
            timestep=timestep,
            integrator_type=integrator_type,
            strategy=strategy,
            integration_time=t_coarse + t_fine,
        )
        logger.info("Results saved to %s", npz_path)
        aim_log_artifact_reference(npz_path)

        metadata = {
            "strategy": strategy,
            "integrator_type": integrator_type,
            "timestep": timestep,
            "half_timestep": half_timestep,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "initial_epoch": cfg.initial_epoch,
            "n_coarse_epochs": len(coarse_epochs),
            "n_fine_epochs": len(fine_epochs),
            "pos_diff_rms": float(np.sqrt(np.mean(pos_diff_norm**2))),
            "vel_diff_rms": float(np.sqrt(np.mean(vel_diff_norm**2))),
            "integration_time": t_coarse + t_fine,
        }

    elif strategy == "forward_backward":
        # --- Propagate forward from start to end ---
        logger.info("Forward propagation with Δt = %.1f s ...", timestep)
        integ_fwd = _make_integrator_settings(timestep, integrator_type, cfg)
        prop_fwd = get_propagator_settings(cfg, ctx, acc, integ_fwd, [])
        t0_fwd = time.perf_counter()
        result_fwd = sim.create_dynamics_simulator(bodies, prop_fwd)
        t_fwd = time.perf_counter() - t0_fwd
        fwd_state_history = result_fwd.propagation_results.state_history
        logger.info("Forward propagation done: %d output epochs in %.2f s.", len(fwd_state_history), t_fwd)

        # --- Get the final forward state ---
        fwd_epochs = np.array(sorted(fwd_state_history.keys()))
        fwd_states = np.array([fwd_state_history[e] for e in fwd_epochs])
        final_state = fwd_states[-1]
        final_epoch = fwd_epochs[-1]
        logger.info("Final forward state at epoch %.2f (s since J2000)", final_epoch)

        # --- Propagate backward from end to start ---
        # Build a fresh propagator starting at the final epoch, going backward
        logger.info("Backward propagation with Δt = %.1f s ...", timestep)
        integ_bwd = _make_integrator_settings(timestep, integrator_type, cfg)
        t0_bwd = time.perf_counter()

        # Termination: backward means end_epoch → start_epoch
        bwd_termination_end = prop_setup.propagator.time_termination(
            float(ctx.start_epoch.to_float())
        )
        bwd_termination_start = prop_setup.propagator.time_termination(
            float(ctx.end_epoch.to_float())
        )
        bwd_termination = prop_setup.propagator.non_sequential_termination(
            bwd_termination_end, bwd_termination_start
        )

        central_bodies = [
            s.central_body for _, s in cfg.bodies_to_propagate.items()
        ]
        prop_bwd = prop_setup.propagator.translational(
            central_bodies,
            acc,
            list(cfg.bodies_to_propagate.keys()),
            final_state,
            final_epoch,
            integ_bwd,
            bwd_termination,
        )

        result_bwd = sim.create_dynamics_simulator(bodies, prop_bwd)
        t_bwd = time.perf_counter() - t0_bwd
        bwd_state_history = result_bwd.propagation_results.state_history
        logger.info("Backward propagation done: %d output epochs in %.2f s.", len(bwd_state_history), t_bwd)

        # --- Build common epoch grid (forward epochs) ---
        bwd_epochs = np.array(sorted(bwd_state_history.keys()))
        bwd_states = np.array([bwd_state_history[e] for e in bwd_epochs])

        # Interpolate backward states to forward epochs
        states_bwd_at_fwd = np.column_stack([
            np.interp(fwd_epochs, bwd_epochs, bwd_states[:, i])
            for i in range(6)
        ])

        # --- Compute differences (forward - backward) ---
        # Skip the first epoch: the initial state can have a spurious mismatch
        # between forward and backward propagations.
        skip_first = 1
        differences = fwd_states[skip_first:] - states_bwd_at_fwd[skip_first:]
        pos_diff_norm = np.linalg.norm(differences[:, :3], axis=1)
        vel_diff_norm = np.linalg.norm(differences[:, 3:], axis=1)

        # The closure error is the difference at the start epoch (index 0)
        closure_pos_error = np.linalg.norm((fwd_states[0] - states_bwd_at_fwd[0])[:3])
        closure_vel_error = np.linalg.norm((fwd_states[0] - states_bwd_at_fwd[0])[3:])

        logger.info(
            "Forward-backward differences computed: position RMS = %.6e m, "
            "velocity RMS = %.6e m/s",
            np.sqrt(np.mean(pos_diff_norm**2)),
            np.sqrt(np.mean(vel_diff_norm**2)),
        )
        logger.info(
            "Round-trip closure error at start epoch: position = %.6e m, velocity = %.6e m/s",
            closure_pos_error,
            closure_vel_error,
        )

        # --- Save results ---
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        npz_path = output_dir / "validation_results.npz"
        np.savez(
            npz_path,
            epochs=fwd_epochs[skip_first:],
            states_forward=fwd_states[skip_first:],
            states_backward=states_bwd_at_fwd[skip_first:],
            differences=differences,
            pos_diff_norm=pos_diff_norm,
            vel_diff_norm=vel_diff_norm,
            timestep=timestep,
            integrator_type=integrator_type,
            strategy=strategy,
            closure_pos_error=closure_pos_error,
            closure_vel_error=closure_vel_error,
            integration_time=t_fwd + t_bwd,
        )
        logger.info("Results saved to %s", npz_path)
        aim_log_artifact_reference(npz_path)

        metadata = {
            "strategy": strategy,
            "integrator_type": integrator_type,
            "timestep": timestep,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "initial_epoch": cfg.initial_epoch,
            "n_forward_epochs": len(fwd_epochs),
            "n_backward_epochs": len(bwd_epochs),
            "pos_diff_rms": float(np.sqrt(np.mean(pos_diff_norm**2))),
            "vel_diff_rms": float(np.sqrt(np.mean(vel_diff_norm**2))),
            "closure_pos_error": float(closure_pos_error),
            "closure_vel_error": float(closure_vel_error),
            "integration_time": t_fwd + t_bwd,
        }

    else:
        raise ValueError(f"Unknown validation strategy: {strategy}")

    # --- Save metadata ---
    import json

    metadata_path = output_dir / "validation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    logger.info("Integrator validation complete.")


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except BaseException as e:
    #     logger.error("Unhandled exception: %s", e, exc_info=True)
    # finally:
    #     sys.exit(0)