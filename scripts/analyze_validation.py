"""Analyze integrator/interpolator validation sweep results.

Reads the NPZ files produced by ``validate_integrator.py`` or
``validate_interpolator.py`` from a sweep output directory, groups them by
integrator/interpolator type, and produces convergence plots.

Usage
-----
::

    python scripts/analyze_validation.py --input-dir results/validate_integrator/<run>/
    python scripts/analyze_validation.py --input-dir results/validate_interpolator/<run>/

Every figure draws a configurable horizontal reference line on the error
(y) axis. It defaults to 14 km — one tenth of Gaia's estimated resolution of
Triton (~140 km) — and can be overridden on the command line::

    python scripts/analyze_validation.py --reference-line-value 5000 --reference-line-label "5 km"
    python scripts/analyze_validation.py --no-reference-line
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from orbitdet.visualization import (
    IntegratorConvergence,
    IntegratorErrorVsTime,
    IntegratorErrorVsTimeFamily,
    InterpolatorConvergence,
)

logger = logging.getLogger(__name__)

# Directory holding the per-figure YAML configs (relative to the repo root).
_FIGURES_CONF_DIR = Path(__file__).resolve().parent.parent / "conf" / "figures"


def _load_figure_cfg(name: str, reference_line: dict | None = None):
    """Load a figure config from ``conf/figures/<name>.yaml``.

    Parameters
    ----------
    name : str
        Figure config name without extension (e.g. ``"integrator_convergence"``).
    reference_line : dict | None
        Optional overrides merged into the config's ``reference_line`` section
        (e.g. ``{"value": 5000.0, "label": "5 km"}``). ``None`` values are
        ignored so callers can pass partially-populated dicts.

    Returns
    -------
    DictConfig
        The loaded (and possibly overridden) config, or an empty config if the
        YAML file does not exist.
    """
    cfg_path = _FIGURES_CONF_DIR / f"{name}.yaml"
    if cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
    else:
        logger.warning("Figure config not found: %s", cfg_path)
        cfg = OmegaConf.create({})

    if reference_line:
        overrides = {k: v for k, v in reference_line.items() if v is not None}
        if overrides:
            section = cfg.get(name, OmegaConf.create({}))
            ref = section.get("reference_line", OmegaConf.create({}))
            cfg[name] = OmegaConf.merge(
                section, {"reference_line": OmegaConf.merge(ref, overrides)}
            )

    return cfg


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
        results.append(
            {
                "npz_path": npz_path,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "output_dir": npz_path.parent,
            }
        )
    return results


def _group_by_type_and_strategy(results: list[dict]) -> dict[str, dict[str, dict[float, dict]]]:
    """Group results by (strategy, integrator/interpolator type), then by timestep.

    Applies a buffer slice to each result to exclude boundary regions where
    Runge's phenomenon may inflate error statistics. For interpolator data the
    buffer comes from metadata (or falls back to the interpolator order). For
    integrator data it uses the integrator order.

    Warns if multiple results share the same key (only the last one is kept).

    Returns a nested dict: ``strategy -> type -> timestep -> result_data``.
    """
    from orbitdet.visualization.validation import _integrator_order

    grouped: dict[str, dict[str, dict[float, dict]]] = {}
    for r in results:
        meta = r["metadata"]
        strategy = meta.get("strategy", "richardson")
        type_key = meta.get("interpolator_type") or meta.get("integrator_type", "unknown")
        timestep = meta.get("timestep", 0.0)

        # Load the NPZ data
        data = np.load(r["npz_path"])
        pos_diff = data["pos_diff_norm"]

        # Determine buffer: how many samples to skip at each boundary
        if meta.get("interpolator_type"):
            # Interpolator data: use buffer from metadata, or fall back to order
            buffer = meta.get("buffer", 0)
            if buffer == 0:
                test_order = meta.get("test_lagrange_order", 0)
                ref_order = meta.get("reference_lagrange_order", 8)
                buffer = max(ref_order, test_order) // 2
        else:
            # Integrator data: use the integrator order
            buffer = _integrator_order(type_key)

        # Apply buffer slice if we have enough samples
        if buffer > 0 and len(pos_diff) > 2 * buffer:
            pos_diff = pos_diff[buffer:-buffer]
            epochs = data["epochs"][buffer:-buffer]
            vel_diff = data["vel_diff_norm"][buffer:-buffer]
        else:
            epochs = data["epochs"]
            vel_diff = data["vel_diff_norm"]

        # Warn about duplicate keys
        existing = grouped.get(strategy, {}).get(type_key, {}).get(timestep)
        if existing is not None:
            logger.warning(
                "Overwriting duplicate key strategy=%s type=%s timestep=%s (old: %s, new: %s)",
                strategy,
                type_key,
                timestep,
                existing["output_dir"],
                r["output_dir"],
            )

        result_data = {
            "epochs": epochs,
            "pos_diff_norm": pos_diff,
            "vel_diff_norm": vel_diff,
            "integration_time": float(data.get("integration_time", np.nan)),
            "output_dir": r["output_dir"],
        }

        grouped.setdefault(strategy, {}).setdefault(type_key, {})[timestep] = result_data

    return grouped


def analyze_integrator(input_dir: Path, reference_line: dict | None = None):
    """Analyze integrator validation results and produce convergence plots.

    Parameters
    ----------
    input_dir : Path
        Directory containing the ``validation_results.npz`` files.
    reference_line : dict | None
        Optional overrides for the horizontal reference line drawn on the
        error (y) axis of every figure (keys: ``enabled``, ``value``,
        ``label``, ``color``, ``linestyle``, ``linewidth``).
    """
    results = _discover_results(input_dir)
    if not results:
        logger.error("No validation_results.npz found under %s", input_dir)
        return

    logger.info("Found %d result files under %s", len(results), input_dir)
    grouped = _group_by_type_and_strategy(results)

    import matplotlib.pyplot as plt

    for strategy, type_groups in grouped.items():
        for integrator_type, type_results in type_groups.items():
            logger.info(
                "Plotting convergence for integrator '%s' (%s, %d timesteps)",
                integrator_type,
                strategy,
                len(type_results),
            )
            cfg = _load_figure_cfg("integrator_convergence", reference_line)
            plotter = IntegratorConvergence(cfg, type_results, integrator_type, strategy=strategy)
            # Use _make_figure() directly — publish() requires HydraConfig
            fig, axes = plotter._make_figure()
            output_path = input_dir / f"integrator_convergence_{integrator_type}_{strategy}.pdf"
            fig.savefig(output_path)
            logger.info(
                "Convergence plot saved to %s",
                output_path,
            )
            plt.close(fig)

        # Family comparison: all integrator types on one figure per strategy
        from orbitdet.visualization import IntegratorFamilyComparison

        plt.close("all")
        cfg = _load_figure_cfg("integrator_family_comparison", reference_line)
        family_plotter = IntegratorFamilyComparison(cfg, type_groups, strategy=strategy)
        fig, ax = family_plotter._make_figure()
        output_path = input_dir / f"integrator_family_comparison_{strategy}.pdf"
        fig.savefig(output_path)
        logger.info("Family comparison plot saved to %s", output_path)
        plt.close(fig)

        # Per-type error vs time
        for integrator_type, type_results in type_groups.items():
            cfg = _load_figure_cfg("integrator_error_vs_time", reference_line)
            evt_plotter = IntegratorErrorVsTime(
                cfg, type_results, integrator_type, strategy=strategy
            )
            fig_evt, ax_evt = evt_plotter._make_figure()
            evt_path = input_dir / f"integrator_error_vs_time_{integrator_type}_{strategy}.pdf"
            fig_evt.savefig(evt_path)
            logger.info("Error-vs-time plot saved to %s", evt_path)
            plt.close(fig_evt)

        # Family error vs time
        cfg = _load_figure_cfg("integrator_error_vs_time_family", reference_line)
        evt_family = IntegratorErrorVsTimeFamily(cfg, type_groups, strategy=strategy)
        fig_evtf, ax_evtf = evt_family._make_figure()
        evtf_path = input_dir / f"integrator_error_vs_time_family_{strategy}.pdf"
        fig_evtf.savefig(evtf_path)
        logger.info("Error-vs-time family plot saved to %s", evtf_path)
        plt.close(fig_evtf)


def analyze_interpolator(input_dir: Path, reference_line: dict | None = None):
    """Analyze interpolator validation results and produce convergence plots.

    Parameters
    ----------
    input_dir : Path
        Directory containing the ``validation_results.npz`` files.
    reference_line : dict | None
        Optional overrides for the horizontal reference line drawn on the
        error (y) axis of every figure (keys: ``enabled``, ``value``,
        ``label``, ``color``, ``linestyle``, ``linewidth``).
    """
    results = _discover_results(input_dir)
    if not results:
        logger.error("No validation_results.npz found under %s", input_dir)
        return

    logger.info("Found %d result files under %s", len(results), input_dir)
    grouped = _group_by_type_and_strategy(results)

    import matplotlib.pyplot as plt

    for strategy, type_groups in grouped.items():
        for interpolator_type, type_results in type_groups.items():
            logger.info(
                "Plotting convergence for interpolator '%s' (%s, %d timesteps)",
                interpolator_type,
                strategy,
                len(type_results),
            )
            cfg = _load_figure_cfg("interpolator_convergence", reference_line)
            plotter = InterpolatorConvergence(cfg, type_results, interpolator_type)
            fig, axes = plotter._make_figure()
            output_path = input_dir / f"interpolator_convergence_{interpolator_type}_{strategy}.pdf"
            fig.savefig(output_path)
            logger.info(
                "Convergence plot saved to %s",
                output_path,
            )
            plt.close(fig)

        # Family comparison: all interpolator types on one figure
        from orbitdet.visualization import InterpolatorFamilyComparison

        cfg = _load_figure_cfg("interpolator_family_comparison", reference_line)
        family_plotter = InterpolatorFamilyComparison(cfg, type_groups)
        fig, ax = family_plotter._make_figure()
        output_path = input_dir / "interpolator_family_comparison.pdf"
        fig.savefig(output_path)
        logger.info("Interpolator family comparison plot saved to %s", output_path)
        plt.close(fig)


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
    parser.add_argument(
        "--reference-line-value",
        type=float,
        default=None,
        help="Y-value [m] of the horizontal reference line drawn on the error "
        "(y) axis of every figure. Defaults to the value in the figure "
        "config (14000 m = one tenth of Gaia's Triton resolution).",
    )
    parser.add_argument(
        "--reference-line-label",
        type=str,
        default=None,
        help="Legend label for the horizontal reference line. Defaults to the "
        "label in the figure config.",
    )
    parser.add_argument(
        "--no-reference-line",
        action="store_true",
        help="Disable the horizontal reference line on all figures.",
    )
    args = parser.parse_args()

    reference_line = {
        "value": args.reference_line_value,
        "label": args.reference_line_label,
    }
    if args.no_reference_line:
        reference_line["enabled"] = False

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
        analyze_integrator(input_dir, reference_line)
    elif args.mode == "interpolator":
        analyze_interpolator(input_dir, reference_line)
    else:
        # Auto-detect: check if any metadata has interpolator_type
        results = _discover_results(input_dir)
        if not results:
            logger.error("No validation_results.npz found under %s", input_dir)
            sys.exit(1)
        has_interpolator = any(r["metadata"].get("interpolator_type") for r in results)
        if has_interpolator:
            analyze_interpolator(input_dir, reference_line)
        else:
            analyze_integrator(input_dir, reference_line)
    from matplotlib import pyplot as plt

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
