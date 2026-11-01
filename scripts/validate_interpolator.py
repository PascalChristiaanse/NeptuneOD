"""Interpolator validation using a known-good reference interpolator.

Propagates Triton with RKF78 at 1800 s to create a dense "truth" trajectory.
A reference Lagrange-8 interpolator is built from this trajectory. For each
test configuration (timestep, Lagrange order), the truth is downsampled to
the test timestep, a test Lagrange interpolator is created, and both
interpolators are evaluated at the same query points. The difference is the
interpolation error estimate.

To avoid Runge's phenomenon, a buffer is excluded from each end of the
comparison. The buffer is specified as a duration (e.g. "6 months") and is
floored at ceil(max(order)/2) reference epochs, since a Lagrange-N
interpolator needs N/2 samples on each side of a query point.

Results are saved as NPZ files for later analysis.

Usage
-----
Single run::

    python scripts/validate_interpolator.py

Sweep over timesteps and Lagrange orders::

    python scripts/validate_interpolator.py --multirun --config-name sweep/validate_interpolator
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import DateTime, iso_string_to_epoch_time_object
from tudatpy.dynamics import propagation_setup as prop_setup
from tudatpy.dynamics import simulator as sim
from tudatpy.math import interpolators as interp

from orbitdet.data import KernelManager
from orbitdet.reproducibility import (
    RuntimeContext,
    enforce_initialization,
    initialize,
)
from orbitdet.simulation import (
    get_dynamical_model,
    get_environment,
    get_propagator_settings,
)
from orbitdet.utility import parse_duration_to_seconds

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/validate_interpolator",
)
@enforce_initialization
def main(cfg: DictConfig):
    logger.info(f"Starting validate_interpolator.py on PID {os.getpid()}")

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
    test_lagrange_order = cfg.validation.lagrange_order
    ref_timestep = cfg.validation.get("reference_timestep", 1800)
    ref_lagrange_order = cfg.validation.get("reference_lagrange_order", 8)
    buffer_cfg = cfg.validation.get("buffer", "0 s")
    integrator_type = cfg.integrator.type
    logger.info(
        "Validating Lagrange-%d interpolator with Δt = %.1f s, "
        "reference: Lagrange-%d at Δt_ref = %.1f s, buffer=%s (integrator: %s)",
        test_lagrange_order,
        timestep,
        ref_lagrange_order,
        ref_timestep,
        buffer_cfg,
        integrator_type,
    )

    # --- Propagate reference trajectory (shared across sweep jobs) ---
    # Cache in the sweep root directory so all jobs in this sweep share it.
    sweep_root = Path(HydraConfig.get().runtime.output_dir).parent
    ref_cache_path = sweep_root / "reference_trajectory.npz"
    if ref_cache_path.exists():
        logger.info("Loading cached reference trajectory from %s", ref_cache_path)
        cached = np.load(ref_cache_path)
        ref_epochs = cached["epochs"]
        ref_states = cached["states"]
        logger.info("Loaded %d reference epochs from cache.", len(ref_epochs))
    else:
        logger.info("Propagating reference trajectory with Δt = %.1f s ...", ref_timestep)
        integ_ref = prop_setup.integrator.runge_kutta_fixed_step(
            ref_timestep,
            coefficient_set=prop_setup.integrator.CoefficientSets.rkf_78,
        )
        prop_ref = get_propagator_settings(cfg, ctx, acc, integ_ref, [])
        result_ref = sim.create_dynamics_simulator(bodies, prop_ref)
        ref_state_history = result_ref.propagation_results.state_history
        logger.info("Reference propagation done: %d output epochs.", len(ref_state_history))

        ref_epochs = np.array(sorted(ref_state_history.keys()))
        ref_states = np.array([ref_state_history[e] for e in ref_epochs])

        # Cache for subsequent sweep jobs
        ref_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref_cache_path, epochs=ref_epochs, states=ref_states)
        logger.info("Reference trajectory cached to %s", ref_cache_path)

    # --- Build reference Lagrange-8 interpolator from full trajectory ---
    ref_data = {float(e): ref_states[i] for i, e in enumerate(ref_epochs)}
    ref_interp_settings = interp.lagrange_interpolation(ref_lagrange_order)
    ref_interpolator = interp.create_one_dimensional_vector_interpolator(
        ref_data, ref_interp_settings
    )
    logger.info(
        "Reference Lagrange-%d interpolator created from %d points.",
        ref_lagrange_order,
        len(ref_data),
    )

    # --- Downsample to test timestep ---
    start_epoch_float = float(ctx.start_epoch.to_float())
    test_epochs = ref_epochs[np.abs((ref_epochs - start_epoch_float) % timestep) < 1e-6]
    test_data = {float(e): ref_states[i] for i, e in enumerate(ref_epochs) if e in test_epochs}
    logger.info(
        "Downsampled to Δt = %.1f s: %d epochs.",
        timestep,
        len(test_data),
    )

    # --- Create test Lagrange interpolator ---
    test_interp_settings = interp.lagrange_interpolation(test_lagrange_order)
    test_interpolator = interp.create_one_dimensional_vector_interpolator(
        test_data, test_interp_settings
    )
    logger.info(
        "Test Lagrange-%d interpolator created from %d points.",
        test_lagrange_order,
        len(test_data),
    )

    # --- Query points: exclude boundary points to avoid Runge's phenomenon ---
    # The buffer is specified as a duration (e.g. "6 months") and converted to
    # a number of reference epochs. A Lagrange-N interpolator needs N/2 samples
    # on each side of a query point, so the buffer is additionally floored at
    # ceil(max(order)/2) *test* samples (converted to reference epochs) to
    # guarantee the test interpolator never reaches past the ends of its own
    # sample set.
    buffer_seconds = parse_duration_to_seconds(buffer_cfg)
    n_skip_time = int(np.ceil(buffer_seconds / ref_timestep))
    n_skip_order_samples = int(np.ceil(max(ref_lagrange_order, test_lagrange_order) / 2))
    n_skip_order = int(np.ceil(n_skip_order_samples * timestep / ref_timestep))
    n_skip = max(n_skip_time, n_skip_order)
    if len(ref_epochs) <= 2 * n_skip:
        raise ValueError(
            f"Buffer of {buffer_cfg} ({n_skip} reference epochs) leaves nothing "
            f"to compare out of {len(ref_epochs)} epochs."
        )
    query_indices = slice(n_skip, len(ref_epochs) - n_skip)
    query_epochs = ref_epochs[query_indices]
    n_query = len(query_epochs)
    logger.info(
        "Querying at %d epochs (skipping %d reference epochs = %.1f days at each "
        "boundary; time buffer %d epochs, order/2 floor %d test samples = %d epochs).",
        n_query,
        n_skip,
        n_skip * ref_timestep / 86400.0,
        n_skip_time,
        n_skip_order_samples,
        n_skip_order,
    )

    # --- Interpolate both at query epochs ---
    ref_interp_values = np.zeros((n_query, 6))
    test_interp_values = np.zeros((n_query, 6))
    for i, epoch in enumerate(query_epochs):
        ref_interp_values[i] = ref_interpolator.interpolate(float(epoch))
        test_interp_values[i] = test_interpolator.interpolate(float(epoch))

    # --- Compute differences (test - reference) ---
    differences = test_interp_values - ref_interp_values
    pos_diff_norm = np.linalg.norm(differences[:, :3], axis=1)
    vel_diff_norm = np.linalg.norm(differences[:, 3:], axis=1)

    logger.info(
        "Differences computed: position RMS = %.6e m, velocity RMS = %.6e m/s",
        np.sqrt(np.mean(pos_diff_norm**2)),
        np.sqrt(np.mean(vel_diff_norm**2)),
    )

    # --- Save results ---
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    npz_path = output_dir / "validation_results.npz"
    np.savez(
        npz_path,
        epochs=query_epochs,
        ref_interp=ref_interp_values,
        test_interp=test_interp_values,
        differences=differences,
        pos_diff_norm=pos_diff_norm,
        vel_diff_norm=vel_diff_norm,
        timestep=timestep,
        integrator_type=integrator_type,
        interpolator_type=f"lagrange_{test_lagrange_order}",
        reference_timestep=ref_timestep,
        reference_lagrange_order=ref_lagrange_order,
        test_lagrange_order=test_lagrange_order,
        buffer=buffer_cfg,
        buffer_seconds=buffer_seconds,
        buffer_epochs=n_skip,
    )
    logger.info("Results saved to %s", npz_path)

    # --- Save metadata ---
    import json

    metadata = {
        "integrator_type": integrator_type,
        "interpolator_type": f"lagrange_{test_lagrange_order}",
        "timestep": timestep,
        "reference_timestep": ref_timestep,
        "reference_lagrange_order": ref_lagrange_order,
        "test_lagrange_order": test_lagrange_order,
        "buffer": buffer_cfg,
        "buffer_seconds": buffer_seconds,
        "buffer_epochs": n_skip,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "initial_epoch": cfg.initial_epoch,
        "n_query_epochs": n_query,
        "n_test_samples": len(test_data),
        "pos_diff_rms": float(np.sqrt(np.mean(pos_diff_norm**2))),
        "vel_diff_rms": float(np.sqrt(np.mean(vel_diff_norm**2))),
    }
    metadata_path = output_dir / "validation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    logger.info("Interpolator validation complete.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.error("Unhandled exception: %s", e, exc_info=True)
    finally:
        sys.exit(0)
        sys.exit(0)
