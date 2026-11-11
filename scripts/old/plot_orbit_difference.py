#!/usr/bin/env python3
"""
Plot the Euclidean distance between two orbits stored in separate .tudat files.

Both files must have been saved via ``SingleArcSimulationResults.save_to_binary``
(or ``SimulationResults.save_binary``) and contain the same propagated body.

Usage:
    python scripts/plot_orbit_difference.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tudatpy.dynamics.propagation import SingleArcSimulationResults

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# ── Configuration ──────────────────────────────────────────────────────────
FILE_A = PROJECT_DIR / "myresult.tudat"
FILE_B = PROJECT_DIR / "results_atanas.tudat"

LABEL_A = "myresult"
LABEL_B = "results_atanas"

OUTPUT_PDF = PROJECT_DIR / "orbit_difference.pdf"
# ───────────────────────────────────────────────────────────────────────────


def load_orbit(path: Path) -> dict[float, np.ndarray]:
    """Load a .tudat file and return its state_history.

    ``save_to_binary`` / ``load_from_binary`` take a path *without* the
    ``.tudat`` extension (it is appended automatically).
    """
    stem = str(path.with_suffix(""))
    print(f"Loading {path.name} (stem={stem}) …")
    results = SingleArcSimulationResults.load_from_binary(stem)
    state_hist = results.state_history
    print(
        f"  → {len(state_hist)} epochs, "
        f"state vector length = {results.propagated_state_vector_length}"
    )
    return state_hist


def main():
    state_a = load_orbit(FILE_A)
    state_b = load_orbit(FILE_B)

    # Find common epochs (within 1 µs tolerance)
    epochs_a = np.array(list(state_a.keys()))
    epochs_b = np.array(list(state_b.keys()))

    # Use set intersection with a tolerance
    # Build a mapping: rounded epoch → exact epoch for both
    def build_epoch_map(epochs, tol=1e-6):
        mapping: dict[float, float] = {}
        for ep in epochs:
            key = round(ep / tol) * tol
            mapping[key] = ep
        return mapping

    map_a = build_epoch_map(epochs_a)
    map_b = build_epoch_map(epochs_b)

    common_keys = sorted(set(map_a.keys()) & set(map_b.keys()))

    if not common_keys:
        print("No overlapping epochs found between the two orbits.")
        return

    print(f"Found {len(common_keys)} common epochs.")

    # Extract position vectors (first 3 components) at common epochs
    positions_a = np.array([state_a[map_a[k]][:3] for k in common_keys])
    positions_b = np.array([state_b[map_b[k]][:3] for k in common_keys])

    # Compute Euclidean distance at each epoch
    distances = np.linalg.norm(positions_a - positions_b, axis=1)

    # Use the exact epoch values for the time axis
    common_epochs = np.array([map_a[k] for k in common_keys])

    # Convert seconds since J2000 to something readable: years since J2000
    seconds_per_year = 365.25 * 86400
    years_since_j2000 = common_epochs / seconds_per_year

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(years_since_j2000, distances, color="tab:blue", linewidth=0.8)

    ax.set_xlabel("Time [years since J2000]")
    ax.set_ylabel("Position difference [m]")
    ax.set_title(
        f"Euclidean distance between {LABEL_A} and {LABEL_B}\n({len(common_keys)} common epochs)"
    )
    ax.grid(True, alpha=0.3)

    # Add a text box with statistics
    stats_text = (
        f"Mean:  {distances.mean():.3e} m\n"
        f"Median: {np.median(distances):.3e} m\n"
        f"Max:   {distances.max():.3e} m\n"
        f"Min:   {distances.min():.3e} m\n"
        f"RMS:   {np.sqrt(np.mean(distances**2)):.3e} m"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF)
    print(f"Plot saved to {OUTPUT_PDF}")

    plt.show()


if __name__ == "__main__":
    main()
