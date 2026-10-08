"""Analyze integrator/interpolator validation sweep results.

Reads the NPZ files produced by ``validate_integrator.py`` or
``validate_interpolator.py`` from a sweep output directory, groups them by
integrator/interpolator type, and produces convergence plots.

Usage
-----
::

    python scripts/analyze_validation.py --input-dir results/validate_integrator/<run>/
    python scripts/analyze_validation.py --input-dir results/validate_interpolator/<run>/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from orbitdet.visualization import (
    IntegratorConvergence,
    IntegratorErrorVsTime,
    IntegratorErrorVsTimeFamily,
    InterpolatorConvergence,
)

logger = logging.getLogger(__name__)


def _discover_results(input_dir: Path) -> list[dict]:
    """Walk ``input_dir`` recursively and collect all ``validation_results.npz`` files.

    Returns a list of dicts, each with keys: ``npz_path``, ``metadata_path``,
    and the loaded metadata (if available).
    """
    results = []
    for npz_path in sorted(input_dir.rglob("validation_results.npz")):
        # Look for companion metadata file
        metadata_path = npz_path.parent / "validation_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        results.append({
            "npz_path": npz_path,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "output_dir": npz_path.parent,
        })
    return results


def _group_by_type_and_strategy(results: list[dict]) -> dict[str, dict[str, dict[float, dict]]]:
    """Group results by (strategy, integrator/interpolator type), then by timestep.

    Warns if multiple results share the same key (only the last one is kept).

    Returns a nested dict: ``strategy -> type -> timestep -> result_data``.
    """
    grouped: dict[str, dict[str, dict[float, dict]]] = {}
    for r in results:
        meta = r["metadata"]
        strategy = meta.get("strategy", "richardson")
        type_key = meta.get("interpolator_type") or meta.get("integrator_type", "unknown")
        timestep = meta.get("timestep", 0.0)

        # Load the NPZ data
        data = np.load(r["npz_path"])
        pos_diff = data["pos_diff_norm"]

        # Warn about duplicate keys
        existing = grouped.get(strategy, {}).get(type_key, {}).get(timestep)
        if existing is not None:
            logger.warning(
                "Overwriting duplicate key strategy=%s type=%s timestep=%s "
                "(old: %s, new: %s)",
                strategy, type_key, timestep,
                existing["output_dir"], r["output_dir"],
            )

        result_data = {
            "epochs": data["epochs"],
            "pos_diff_norm": pos_diff,
            "vel_diff_norm": data["vel_diff_norm"],
            "integration_time": float(data.get("integration_time", np.nan)),
            "output_dir": r["output_dir"],
        }

        grouped.setdefault(strategy, {}).setdefault(type_key, {})[timestep] = result_data

    return grouped


def analyze_integrator(input_dir: Path):
    """Analyze integrator validation results and produce convergence plots."""
    results = _discover_results(input_dir)
    if not results:
        logger.error("No validation_results.npz found under %s", input_dir)
        return

    logger.info("Found %d result files under %s", len(results), input_dir)
    grouped = _group_by_type_and_strategy(results)

    for strategy, type_groups in grouped.items():
        for integrator_type, type_results in type_groups.items():
            logger.info(
                "Plotting convergence for integrator '%s' (%s, %d timesteps)",
                integrator_type,
                strategy,
                len(type_results),
            )
            from omegaconf import OmegaConf

            dummy_cfg = OmegaConf.create({})
            plotter = IntegratorConvergence(
                dummy_cfg, type_results, integrator_type, strategy=strategy
            )
            # Use _make_figure() directly — publish() requires HydraConfig
            fig, axes = plotter._make_figure()
            output_path = input_dir / f"integrator_convergence_{integrator_type}_{strategy}.pdf"
            fig.savefig(output_path)
            logger.info(
                "Convergence plot saved to %s", output_path,
            )

        # Family comparison: all integrator types on one figure per strategy
        from omegaconf import OmegaConf
        from orbitdet.visualization import IntegratorFamilyComparison

        import matplotlib.pyplot as plt

        plt.close("all")
        dummy_cfg = OmegaConf.create({})
        family_plotter = IntegratorFamilyComparison(
            dummy_cfg, type_groups, strategy=strategy
        )
        fig, ax = family_plotter._make_figure()
        output_path = input_dir / f"integrator_family_comparison_{strategy}.pdf"
        fig.savefig(output_path)
        logger.info("Family comparison plot saved to %s", output_path)

        # Per-type error vs time
        for integrator_type, type_results in type_groups.items():
            evt_plotter = IntegratorErrorVsTime(
                dummy_cfg, type_results, integrator_type, strategy=strategy
            )
            fig_evt, ax_evt = evt_plotter._make_figure()
            evt_path = input_dir / f"integrator_error_vs_time_{integrator_type}_{strategy}.pdf"
            fig_evt.savefig(evt_path)
            logger.info("Error-vs-time plot saved to %s", evt_path)
            plt.close(fig_evt)

        # Family error vs time
        evt_family = IntegratorErrorVsTimeFamily(
            dummy_cfg, type_groups, strategy=strategy
        )
        fig_evtf, ax_evtf = evt_family._make_figure()
        evtf_path = input_dir / f"integrator_error_vs_time_family_{strategy}.pdf"
        fig_evtf.savefig(evtf_path)
        logger.info("Error-vs-time family plot saved to %s", evtf_path)
        plt.show()


def analyze_interpolator(input_dir: Path):
    """Analyze interpolator validation results and produce convergence plots."""
    results = _discover_results(input_dir)
    if not results:
        logger.error("No validation_results.npz found under %s", input_dir)
        return

    logger.info("Found %d result files under %s", len(results), input_dir)
    grouped = _group_by_type_and_strategy(results)

    for strategy, type_groups in grouped.items():
        for interpolator_type, type_results in type_groups.items():
            logger.info(
                "Plotting convergence for interpolator '%s' (%s, %d timesteps)",
                interpolator_type,
                strategy,
                len(type_results),
            )
            from omegaconf import OmegaConf

            dummy_cfg = OmegaConf.create({})
            plotter = InterpolatorConvergence(
                dummy_cfg, type_results, interpolator_type
            )
            fig, axes = plotter._make_figure()
            output_path = input_dir / f"interpolator_convergence_{interpolator_type}_{strategy}.pdf"
            fig.savefig(output_path)
            logger.info(
                "Convergence plot saved to %s", output_path,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze integrator/interpolator validation sweep results."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to the sweep output directory containing validation_results.npz files. "
             "If omitted, uses the latest subfolder under results/validate_integrator/ "
             "or results/validate_interpolator/.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["integrator", "interpolator", "auto"],
        default="auto",
        help="Analysis mode. 'auto' detects from metadata.",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        input_dir = Path(args.input_dir)
    else:
        # Auto-detect: look inside each results/validate_* directory for
        # timestamped subdirectories and pick the most recently modified one.
        candidates = []
        for parent in sorted(Path("results").glob("validate_*")):
            if not parent.is_dir():
                continue
            for sub in parent.iterdir():
                if sub.is_dir():
                    candidates.append(sub)
        if not candidates:
            logger.error("No results/validate_*/<timestamp> directories found.")
            sys.exit(1)
        # Pick the most recently modified subdirectory
        input_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        logger.info("Auto-detected latest results directory: %s", input_dir)
        print(f"Reading results from: {input_dir}", flush=True)

    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    if args.mode == "integrator":
        analyze_integrator(input_dir)
    elif args.mode == "interpolator":
        analyze_interpolator(input_dir)
    else:
        # Auto-detect: check if any metadata has interpolator_type
        results = _discover_results(input_dir)
        if not results:
            logger.error("No validation_results.npz found under %s", input_dir)
            sys.exit(1)
        has_interpolator = any(
            r["metadata"].get("interpolator_type") for r in results
        )
        if has_interpolator:
            analyze_interpolator(input_dir)
        else:
            analyze_integrator(input_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()