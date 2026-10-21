#!/usr/bin/env python3
"""clean_orphan_runs.py — Remove Hydra output dirs whose Aim run was deleted.

Aim only manages data inside ``results/.aim/``. When a run is removed with
``aim runs rm`` (or via the UI), the corresponding Hydra output directory is
left behind on disk. This script cross-references the two and cleans up the
leftovers.

The link between the two is the ``.aim_run_hash`` file that
:func:`orbitdet.reproducibility.aim.aim_start_run` writes into every output
directory. A directory is an *orphan* when its hash is absent from Aim's
``run_metadata.sqlite``.

Both directory layouts are handled::

    results/<experiment>/<timestamp>[/<subdir>]              # current
    results/<experiment>/<git_commit>/<timestamp>[/<subdir>] # legacy

Usage
-----
::

    cd /path/to/NeptuneOD
    python scripts/clean_orphan_runs.py                 # preview (default)
    python scripts/clean_orphan_runs.py --delete        # delete orphans
    python scripts/clean_orphan_runs.py --trash /tmp/x  # move instead of delete
    python scripts/clean_orphan_runs.py --experiment solve_least_squares

By default the script only reports. ``--delete`` removes the directories
permanently; ``--trash DIR`` moves them to *DIR* instead, which is reversible.

Directories without a ``.aim_run_hash`` file (created before that feature
existed) cannot be classified and are reported separately, never touched.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_orphan_runs")

# Repo root relative to this script (scripts/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results"

# Directories never descended into.
SKIP_DIRS = {".aim", ".git", "__pycache__", ".ruff_cache", ".pytest_cache"}

HASH_FILE = ".aim_run_hash"


# ── Aim metadata ─────────────────────────────────────────────────────


def read_aim_run_hashes(results_root: Path) -> set[str]:
    """Return the set of run hashes present in Aim's metadata database.

    Reads ``run_metadata.sqlite`` directly (read-only) rather than going
    through the Aim SDK, so the script works while the Aim server is running
    and never opens a RocksDB container.
    """
    db_path = results_root / ".aim" / "run_metadata.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Aim metadata database not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {row[0] for row in con.execute("SELECT hash FROM run")}
    finally:
        con.close()


# ── scanning ─────────────────────────────────────────────────────────


def scan_run_dirs(results_root: Path, experiment: str | None = None):
    """Yield ``(dir_path, run_hash)`` for every dir containing a hash file.

    Directories without a ``.aim_run_hash`` file are yielded with a hash of
    ``None`` so the caller can report them separately.
    """
    for dirpath, dirnames, filenames in os.walk(results_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        path = Path(dirpath)
        if path == results_root:
            continue
        if experiment is not None and path.relative_to(results_root).parts[0] != experiment:
            continue

        if HASH_FILE in filenames:
            try:
                run_hash = (path / HASH_FILE).read_text().strip()
            except OSError as exc:
                logger.warning("could not read %s: %s", path / HASH_FILE, exc)
                continue
            yield path, run_hash


def classify(results_root: Path, aim_hashes: set[str], experiment: str | None):
    """Split scanned dirs into orphans, live dirs, and unclassifiable dirs."""
    orphans: list[tuple[Path, str]] = []
    live: list[tuple[Path, str]] = []
    unknown: list[Path] = []

    for path, run_hash in scan_run_dirs(results_root, experiment):
        if not run_hash:
            unknown.append(path)
        elif run_hash in aim_hashes:
            live.append((path, run_hash))
        else:
            orphans.append((path, run_hash))

    return orphans, live, unknown


# ── reporting ────────────────────────────────────────────────────────


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def report(results_root: Path, orphans, live, unknown) -> None:
    logger.info("Aim runs in database:      %d", len(live) + len(orphans))
    logger.info("Output dirs with hash file: %d", len(live) + len(orphans))
    logger.info("  live (run still in Aim):  %d", len(live))
    logger.info("  orphaned (run deleted):   %d", len(orphans))

    if orphans:
        by_exp: dict[str, int] = {}
        for path, _ in orphans:
            exp = path.relative_to(results_root).parts[0]
            by_exp[exp] = by_exp.get(exp, 0) + 1
        logger.info("Orphans by experiment:")
        for name, count in sorted(by_exp.items(), key=lambda kv: -kv[1]):
            logger.info("  %5d  %s", count, name)

        total = sum(_dir_size(p) for p, _ in orphans)
        logger.info("Reclaimable space: %s", _human(total))

    if unknown:
        logger.warning(
            "%d dir(s) have no %s file and cannot be classified (not touched):",
            len(unknown),
            HASH_FILE,
        )
        for path in unknown[:10]:
            logger.warning("  %s", path.relative_to(results_root))
        if len(unknown) > 10:
            logger.warning("  ... and %d more", len(unknown) - 10)


# ── removal ──────────────────────────────────────────────────────────


def remove_orphans(results_root: Path, orphans, *, trash: Path | None) -> int:
    """Delete or move the orphan dirs. Returns the number handled."""
    if trash is not None:
        trash.mkdir(parents=True, exist_ok=True)

    handled = 0
    for path, _ in orphans:
        rel = path.relative_to(results_root)
        try:
            if trash is not None:
                dest = trash / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
            else:
                shutil.rmtree(path)
            handled += 1
        except OSError as exc:
            logger.error("failed to remove %s: %s", rel, exc)

    # Prune directories left empty by the removal (e.g. legacy commit dirs).
    for dirpath, dirnames, filenames in os.walk(results_root, topdown=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        path = Path(dirpath)
        if path == results_root or not path.is_dir():
            continue
        try:
            if not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass

    return handled


# ── main ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove Hydra output dirs whose Aim run was deleted.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Path to the results directory (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Only consider this experiment (top-level dir under results/)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete orphaned directories",
    )
    parser.add_argument(
        "--trash",
        type=Path,
        default=None,
        metavar="DIR",
        help="Move orphaned directories to DIR instead of deleting (reversible)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log warnings and errors",
    )
    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    results_root = args.results.resolve()
    if not results_root.is_dir():
        logger.error("results directory not found: %s", results_root)
        return 1

    if args.delete and args.trash is not None:
        logger.error("--delete and --trash are mutually exclusive")
        return 2

    try:
        aim_hashes = read_aim_run_hashes(results_root)
    except (FileNotFoundError, sqlite3.Error) as exc:
        logger.error("could not read Aim metadata: %s", exc)
        return 1

    orphans, live, unknown = classify(results_root, aim_hashes, args.experiment)
    report(results_root, orphans, live, unknown)

    if not orphans:
        logger.info("Nothing to do.")
        return 0

    if not args.delete and args.trash is None:
        logger.info("Dry run — pass --delete or --trash DIR to remove the orphans.")
        return 0

    handled = remove_orphans(results_root, orphans, trash=args.trash)
    if args.trash is not None:
        logger.info("Moved %d orphaned dir(s) to %s", handled, args.trash)
    else:
        logger.info("Deleted %d orphaned dir(s)", handled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
