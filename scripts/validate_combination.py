"""Final validation of one integrator+propagator combination against a reference.

The sweep-based validators answer "which integrator is good enough" by tracing
a convergence curve. This script answers the production question instead:

    *Given the combination I intend to use, how far does it drift from a
    trusted reference over the full arc?*

Two propagations are run on the same dynamical model with the same initial
state; only the integrator (type + timestep) and the propagator formulation
differ:

* **test** — ``integrator.type`` at ``integrator.fixed_step_size``
  (e.g. ``RKF89 @ 3600 s``);
* **reference** — ``reference_integrator.type`` at
  ``reference_integrator.fixed_step_size`` (e.g. ``RKF1412 @ 900 s``).

The difference between the two Cartesian state histories is the result. It is
reported as RMS/max position and velocity differences, decomposed into the RSW
frame, and plotted over time.

Epoch alignment
---------------
The comparison runs on the coarser of the two epoch grids. Epochs shared by
both histories are compared with *no* interpolation; only the remaining epochs
use a Lagrange interpolator. With commensurate timesteps (3600 s vs 900 s)
every compared epoch is exact, so the reported difference is pure integration
error. This is a strict improvement over the Richardson strategy in
``validate_integrator.py``, where the linear interpolation between the Δt and
Δt/2 grids contributes error of the same order as the integrator itself.

Choosing a reference
--------------------
The reference must be much more accurate than the test combination; otherwise
the reported difference measures the *reference's* error instead. Refine
``reference_integrator.fixed_step_size`` (and/or raise its order) until the
result stops changing. The ``fractional_pos_*`` metrics express the difference
relative to Triton's orbital radius, and ``*_exact_only`` metrics repeat the
statistics over just the epochs compared without interpolation.

The shipped default (RKF1412 @ 900 s) was checked this way: against RKF1412 @
225 s over a 2-year arc it differed by 0.05 m RMS, so that bounds the
reference's own error and leaves it far below any test combination's error.

Note that the Richardson runs in ``validate_integrator.py`` are *not* a usable
substitute for this check. Their Δt vs Δt/2 comparison relies on a linear
interpolation between the two grids, so their reported error is dominated by
that interpolation rather than by the integrator.

Usage
-----
::

    python scripts/validate_combination.py                       # defaults
    python scripts/validate_combination.py integrator.type=RKF108 \
        integrator.fixed_step_size=7200
    python scripts/validate_combination.py validation.propagator=encke

Outputs (in the Hydra run directory):

* ``combination_comparison.pdf`` — the three-panel figure;
* ``combination_validation.npz`` — the compared time series;
* ``combination_metadata.json`` — scalar metrics and the run provenance.
"""

import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tudatpy.astro.time_representation import DateTime, iso_string_to_epoch_time_object

from orbitdet.data import KernelManager
from orbitdet.reproducibility import (
    RuntimeContext,
    enforce_initialization,
    initialize,
)
from orbitdet.simulation import get_dynamical_model, get_environment
from orbitdet.validation import (
    build_combination_settings,
    compare_outcomes,
    format_metrics,
    run_propagation,
)
from orbitdet.visualization import CombinationComparison

logger = logging.getLogger(__name__)


def _combination_label(integrator_type: str, timestep: float, propagator: str) -> str:
    """Build the human-readable label used in logs, titles and metadata."""
    return f"{integrator_type} @ {timestep:.0f} s ({propagator})"


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/validate_combination",
)
@enforce_initialization
def main(cfg: DictConfig):
    logger.info(f"Starting validate_combination.py on PID {os.getpid()}")

    # --- Configuration ------------------------------------------------------
    test_type = cfg.integrator.type
    test_timestep = float(cfg.integrator.fixed_step_size)
    test_propagator = cfg.validation.get("propagator", "cowell")

    ref_type = cfg.reference_integrator.type
    ref_timestep = float(cfg.reference_integrator.fixed_step_size)
    ref_propagator = cfg.validation.get("reference_propagator", "cowell")

    buffer = int(cfg.validation.get("buffer", 20))
    interpolator_order = int(cfg.validation.get("interpolator_order", 8))
    save_npz = bool(cfg.validation.get("save_npz", True))

    if (test_type, test_timestep) == (ref_type, ref_timestep) and test_propagator == ref_propagator:
        raise ValueError(
            "The test and reference combinations are identical; there is nothing "
            "to validate. Override integrator.*, reference_integrator.* or "
            "validation.propagator."
        )

    if ref_timestep > test_timestep:
        logger.warning(
            "Reference timestep (%.0f s) is coarser than the test timestep (%.0f s). "
            "The reference will not necessarily be the more accurate of the two.",
            ref_timestep,
            test_timestep,
        )

    logger.info(
        "Test combination:      %s",
        _combination_label(test_type, test_timestep, test_propagator),
    )
    logger.info(
        "Reference combination: %s",
        _combination_label(ref_type, ref_timestep, ref_propagator),
    )

    # --- Initialization -----------------------------------------------------
    ctx: RuntimeContext = initialize(cfg)
    ctx.start_epoch = iso_string_to_epoch_time_object(cfg.start_date)
    ctx.end_epoch = iso_string_to_epoch_time_object(cfg.end_date)
    ctx.initial_epoch = iso_string_to_epoch_time_object(cfg.initial_epoch)
    logger.info(
        "Validation arc: %s to %s",
        DateTime.from_epoch_time_object(ctx.start_epoch).to_iso_string(),
        DateTime.from_epoch_time_object(ctx.end_epoch).to_iso_string(),
    )

    # --- Load kernels -------------------------------------------------------
    logger.info("Loading kernels...")
    km = KernelManager(cfg)
    km.download_all_kernels()
    km.furnish()
    logger.info("Kernels loaded.")

    # --- Build environment (shared by both combinations) --------------------
    bodies = get_environment(cfg, ctx)
    accelerations = get_dynamical_model(cfg, ctx, bodies)
    logger.info("Environment and dynamics built (shared by both propagations).")

    # --- Build both propagator settings ------------------------------------
    # The initial state is resolved once, from the reference combination's
    # settings, and then handed to both propagators verbatim. This guarantees
    # the comparison measures accumulated integration error only, not an offset
    # between two independently resolved initial states.
    reference_settings = build_combination_settings(
        cfg, ctx, accelerations, ref_type, ref_timestep, ref_propagator, initial_state=None
    )
    shared_initial_state = np.array(reference_settings.initial_states, dtype=float).flatten()
    logger.info("Shared initial state: %s", shared_initial_state.tolist())

    test_settings = build_combination_settings(
        cfg,
        ctx,
        accelerations,
        test_type,
        test_timestep,
        test_propagator,
        initial_state=shared_initial_state,
    )

    # --- Propagate both combinations ---------------------------------------
    reference = run_propagation(
        bodies,
        reference_settings,
        label=_combination_label(ref_type, ref_timestep, ref_propagator),
        integrator_type=ref_type,
        timestep=ref_timestep,
        propagator=ref_propagator,
    )
    test = run_propagation(
        bodies,
        test_settings,
        label=_combination_label(test_type, test_timestep, test_propagator),
        integrator_type=test_type,
        timestep=test_timestep,
        propagator=test_propagator,
    )

    # --- Compare ------------------------------------------------------------
    comparison = compare_outcomes(
        reference,
        test,
        buffer=buffer,
        interpolator_order=interpolator_order,
    )

    logger.info(
        "Comparison result (%s):\n%s",
        comparison.mode,
        format_metrics(comparison.metrics),
    )

    # --- Save ---------------------------------------------------------------
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    metadata = {
        "test": {
            "integrator_type": test_type,
            "timestep": test_timestep,
            "propagator": test_propagator,
            "label": test.label,
            "n_epochs": test.n_epochs,
            "integration_time": test.integration_time,
        },
        "reference": {
            "integrator_type": ref_type,
            "timestep": ref_timestep,
            "propagator": ref_propagator,
            "label": reference.label,
            "n_epochs": reference.n_epochs,
            "integration_time": reference.integration_time,
        },
        "buffered_epochs": buffer,
        "interpolator_order": interpolator_order,
        "mode": comparison.mode,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "initial_epoch": cfg.initial_epoch,
        "metrics": comparison.metrics,
    }

    metadata_path = output_dir / "combination_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    logger.info("Metadata saved to %s", metadata_path)

    if save_npz:
        npz_path = output_dir / "combination_validation.npz"
        np.savez(
            npz_path,
            epochs=comparison.epochs,
            pos_diff=comparison.pos_diff,
            vel_diff=comparison.vel_diff,
            rsw_diff=comparison.rsw_diff,
            pos_diff_norm=comparison.pos_diff_norm,
            vel_diff_norm=comparison.vel_diff_norm,
            fractional_pos_diff=comparison.fractional_pos_diff,
            n_epochs_exact=comparison.metrics["n_epochs_exact"],
            mode=comparison.mode,
            test_integrator_type=test_type,
            test_timestep=test_timestep,
            test_propagator=test_propagator,
            reference_integrator_type=ref_type,
            reference_timestep=ref_timestep,
            reference_propagator=ref_propagator,
        )
        logger.info("Comparison time series saved to %s", npz_path)

    # --- Metrics to Aim -----------------------------------------------------

    # --- Figure -------------------------------------------------------------
    logger.info("Plotting comparison figure...")
    combination_plotter = CombinationComparison(cfg, comparison, reference, test)
    fig, axes = combination_plotter._make_figure()
    pdf_path = output_dir / "combination_comparison.pdf"
    fig.savefig(pdf_path)
    logger.info("Comparison figure saved to %s", pdf_path)

    from matplotlib import pyplot as plt

    plt.close(fig)

    logger.info("Combination validation complete.")


if __name__ == "__main__":
    main()
