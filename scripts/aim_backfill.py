#!/usr/bin/env python3
"""aim_backfill.py — Import existing Hydra run directories into Aim.

Scans ``results/<experiment>/<commit>/<timestamp>/`` for ``config.yaml``
and ``metadata.yaml``, creates an Aim run for each, and logs everything
so it appears in the Aim UI.

Usage
-----
    cd /path/to/NeptuneOD
    /path/to/env/bin/python scripts/aim_backfill.py          # backfill everything
    /path/to/env/bin/python scripts/aim_backfill.py --dry-run # preview only
    /path/to/env/bin/python scripts/aim_backfill.py --run-hash a6a06b0  # single commit
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aim_backfill")

AIM_REPO_DIR = "results"
# Folders at the top of results/ that are not experiment results
SKIP_DIRS = {".aim", ".gitkeep", "__pycache__"}

# Regex for timestamp-based run folders:  YYYY-MM-DD_HH-MM-SS
_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


# ── helpers ──────────────────────────────────────────────────────────


def _is_run_dir(name: str) -> bool:
    """True if *name* looks like a Hydra timestamp output dir."""
    return bool(_RUN_DIR_RE.match(name))


def _experiment_name(results_root: Path, commit_dir: Path) -> str:
    """Derive the experiment name from the directory structure.

    results/<experiment>/<commit>/<timestamp>/
    """
    return commit_dir.parent.name


def _flatten_cfg(cfg, prefix: str = "", sep: str = ".") -> dict:
    """Recursively flatten an OmegaConf dict into a flat key-value dict."""
    result: dict = {}
    for key, value in cfg.items():
        flat = f"{prefix}{sep}{key}" if prefix else key
        if hasattr(value, "items"):
            result.update(_flatten_cfg(value, prefix=flat, sep=sep))
        elif isinstance(value, (list, tuple)):
            result[flat] = ",".join(str(v) for v in value)
        else:
            result[flat] = value
    return result


# ── scanning ─────────────────────────────────────────────────────────


def find_runs(results_root: Path, run_hash_filter: str | None = None):
    """Yield ``(experiment, commit_str, timestamp_dir)`` tuples."""
    for exp_dir in sorted(results_root.iterdir()):
        if exp_dir.name in SKIP_DIRS or not exp_dir.is_dir():
            continue
        for commit_dir in sorted(exp_dir.iterdir()):
            if not commit_dir.is_dir():
                continue
            # commit_dir may be named "4b6b969" or "4b6b969_dirty"
            commit_str = commit_dir.name
            if run_hash_filter and run_hash_filter not in commit_str:
                continue

            for ts_dir in sorted(commit_dir.iterdir()):
                if not ts_dir.is_dir() or not _is_run_dir(ts_dir.name):
                    continue
                yield exp_dir.name, commit_str, ts_dir


# ── backfill single run ──────────────────────────────────────────────


def backfill_run(experiment: str, commit_str: str, ts_dir: Path, *, dry_run: bool) -> bool:
    """Create an Aim run from the Hydra output at *ts_dir*.

    Returns True on success, False on skip/error.
    """
    config_path = ts_dir / "config.yaml"
    metadata_path = ts_dir / "metadata.yaml"
    aim_hash_file = ts_dir / ".aim_run_hash"

    # Skip if already imported
    if aim_hash_file.exists():
        logger.info("  ↳ already imported (hash %s), skipping", aim_hash_file.read_text().strip())
        return True

    if not config_path.exists():
        logger.warning("  ↳ no config.yaml, skipping %s", ts_dir)
        return False

    # ── read metadata ──
    git_commit = commit_str
    seed = 42
    if metadata_path.exists():
        try:
            meta = yaml.safe_load(metadata_path.read_text()) or {}
            git_commit = meta.get("git_commit", commit_str)
            seed = meta.get("seed", 42)
        except Exception:
            pass

    # ── read config ──
    try:
        cfg = OmegaConf.load(str(config_path))
    except Exception as exc:
        logger.warning("  ↳ failed to load config.yaml: %s", exc)
        return False

    if dry_run:
        logger.info(
            "  would import: experiment=%s commit=%s timestamp=%s",
            experiment,
            commit_str,
            ts_dir.name,
        )
        return True

    # ── create Aim run ──
    from aim.sdk import Run

    run = Run(
        repo=AIM_REPO_DIR,
        experiment=experiment,
        capture_terminal_logs=False,
    )

    # Name the run after the script + git info
    run.name = f"{experiment} ({git_commit})"

    # tags
    if "_dirty" in commit_str:
        run.add_tag("dirty")

    # set artifacts URI so log_artifact works later
    run.set_artifacts_uri("file://" + AIM_REPO_DIR)

    # ── log hyperparams ──
    flat_cfg = _flatten_cfg(cfg)
    run["hparams"] = flat_cfg
    run["git_commit"] = git_commit
    run["seed"] = seed
    run["output_dir"] = str(ts_dir)

    # ── log PDF figures as artifacts (attached for download) ──
    for pdf in sorted(ts_dir.glob("*.pdf")):
        try:
            run.log_artifact(str(pdf), pdf.name)
            logger.debug("  attached artifact %s", pdf.name)
        except Exception as exc:
            logger.debug("  could not attach artifact %s: %s", pdf.name, exc)

    # ── attach log files as artifacts ──
    for log_file in sorted(ts_dir.glob("*.log")):
        try:
            run.log_artifact(str(log_file), log_file.name)
            logger.debug("  attached log %s", log_file.name)
        except Exception as exc:
            logger.debug("  could not attach log %s: %s", log_file.name, exc)

    # ── store link file ──
    aim_hash_file.write_text(run.hash)

    # ── close ──
    run.close()
    logger.info("  ✓ %s  (run=%s)", ts_dir.name, run.hash)
    return True


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill existing Hydra results into Aim")
    parser.add_argument("--dry-run", action="store_true", help="Only list what would be imported")
    parser.add_argument(
        "--run-hash",
        default=None,
        help="Only import runs whose commit hash contains this string (e.g. a6a06b0)",
    )
    args = parser.parse_args()

    results_root = Path(AIM_REPO_DIR)
    if not (results_root / ".aim").exists():
        logger.error(
            "No Aim repository found at %s/.aim — run `aim init --repo results` first", results_root
        )
        sys.exit(1)

    runs = list(find_runs(results_root, run_hash_filter=args.run_hash))
    if not runs:
        logger.info("No un-imported runs found.")
        return

    logger.info("Found %d run(s) to backfill%s", len(runs), " (DRY RUN)" if args.dry_run else "")

    ok = skipped = 0
    for experiment, commit_str, ts_dir in runs:
        logger.info("  %s / %s / %s", experiment, commit_str, ts_dir.name)
        if backfill_run(experiment, commit_str, ts_dir, dry_run=args.dry_run):
            ok += 1
        else:
            skipped += 1

    logger.info(
        "Done — %d imported, %d skipped%s", ok, skipped, " (dry run)" if args.dry_run else ""
    )

    if not args.dry_run and ok:
        logger.info("Start the UI with:  aim up")


if __name__ == "__main__":
    main()
