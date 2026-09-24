"""
DelftBlue validation soak (WP7).

Runs a fast orbit determination with the new logging pipeline, then prints
diagnostics so the operator can verify correctness without opening the Aim UI.

Usage (local test):
    python scripts/delftblue_validation.py --config-name=sweep/validation_fast \
        'hydra.run.dir=results/_soak/run' 'fail=false'

Usage (on the cluster via submitit, 2-job sweep):
    sbatch scripts/run_delftblue_validation.sbatch

What this tests:
    1. No freeze under ``srun`` (stdout is a pipe, which is the exact condition
       that deadlocked the old ``FdCapture`` mechanism).
    2. The stdout/stderr split: ``.err`` contains WARNING and above.
    3. The file sinks: ``run.log`` and ``errors.log`` are written.
    4. The Aim handler: records appear in the Aim ``__log_records`` sequence.
    5. ``@enforce_initialization``: a crashed job tags the Aim run and a
       traceback lands in the child's ``.err``.
    6. Per-job submitit folders: each job's logs are isolated.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from orbitdet.reproducibility import RuntimeContext, enforce_initialization, initialize
from orbitdet.reproducibility.logging import get_aim_handler, get_logger, shutdown_logging

log = get_logger(__name__)

#: Markers that the validation script checks.
PASS = "[VALIDATION-PASS]"
FAIL = "[VALIDATION-FAIL]"
CHECK = "[CHECK]"


def _verify_run_dir(run_dir: str) -> list[str]:
    """Check file-sink contents and return a list of diagnostic lines."""
    rd = Path(run_dir)
    findings: list[str] = []

    # Run log exists and has both INFO and WARNING.
    run_log = rd / "run.log"
    if run_log.exists():
        text = run_log.read_text(encoding="utf-8", errors="replace")
        has_info = "VALIDATION-INFO" in text
        has_warning = "VALIDATION-WARNING" in text
        has_error = "VALIDATION-CRASH" in text or "VALIDATION-ERROR" in text
        findings.append(
            f"{CHECK} {run_log.name}: "
            f"INFO={'OK' if has_info else 'MISS'} "
            f"WARNING={'OK' if has_warning else 'MISS'} "
            f"ERROR={'OK' if has_error else '-'}"
        )
    else:
        findings.append(f"{FAIL} {run_log.name} not found at {rd}")

    # Errors log exists and has only ERROR records.
    err_log = rd / "errors.log"
    if err_log.exists():
        text = err_log.read_text(encoding="utf-8", errors="replace")
        has_info = "VALIDATION-INFO" in text
        has_error = "VALIDATION-ERROR" in text or "VALIDATION-CRASH" in text
        findings.append(
            f"{CHECK} {err_log.name}: "
            f"ERROR={'OK' if has_error else 'MISS'} "
            f"INFO-leak={'YES-BAD' if has_info else 'OK'}"
        )
    else:
        findings.append(f"{CHECK} errors.log: not present (expected for success job)")

    return findings


def _verify_submitit_dir(submitit_dir: str) -> list[str]:
    """Check submitit log files and return diagnostics."""
    sd = Path(submitit_dir)
    findings: list[str] = []

    if not sd.exists():
        findings.append(f"{CHECK} submitit dir: none (not a submitit run)")
        return findings

    for job_dir in sorted(sd.iterdir()):
        if not job_dir.is_dir():
            continue

        err = next((f for f in job_dir.iterdir() if f.suffix == ".err"), None)
        out = next((f for f in job_dir.iterdir() if f.suffix == ".out"), None)

        if err and out:
            err_text = err.read_text(encoding="utf-8", errors="replace")
            had_traceback = "Traceback" in err_text
            had_crash_log = "VALIDATION-CRASH" in err_text or "VALIDATION-ERROR" in err_text
            findings.append(
                f"{CHECK} submitit/{job_dir.name}: "
                f"traceback-in-err={'OK' if had_traceback else '-'} "
                f"crash-log-in-err={'OK' if had_crash_log else '-'}"
            )
        elif err:
            findings.append(f"{CHECK} submitit/{job_dir.name}: .err present (no .out)")
        elif out:
            findings.append(f"{CHECK} submitit/{job_dir.name}: .out present (no .err)")
        else:
            findings.append(f"{CHECK} submitit/{job_dir.name}: dir only")

    return findings


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="sweep/validation_fast",
)
@enforce_initialization
def main(cfg: DictConfig) -> None:
    ctx: RuntimeContext = initialize(cfg)

    job_num = HydraConfig.get().job.num
    do_crash = OmegaConf.select(cfg, "fail", default=False)

    log.info("VALIDATION-INFO job=%s", job_num)
    log.warning("VALIDATION-WARNING job=%s", job_num)

    if do_crash:
        log.error("VALIDATION-CRASH job=%s", job_num)
        raise RuntimeError(f"VALIDATION-INTENTIONAL-CRASH job={job_num}")

    log.info("VALIDATION-COMPLETED job=%s", job_num)

    # Capture Aim run hash before shutdown_logging() detaches it.
    handler = get_aim_handler()
    aim_hash = handler.run.hash if (handler is not None and handler.run is not None) else None

    # Drain and shut down the logging pipeline before checking files,
    # otherwise queued records may still be in-flight.
    shutdown_logging()

    # === Diagnostics (only for non-crash jobs) ===
    output_dir = str(ctx.output_dir)
    print(f"\n{PASS} {output_dir}", flush=True)

    for line in _verify_run_dir(output_dir):
        print(line, flush=True)

    # Look for the submitit folder alongside the run directory.
    submitit_candidates = list(Path(output_dir).parent.glob(".submitit"))
    if submitit_candidates:
        for line in _verify_submitit_dir(str(submitit_candidates[0])):
            print(line, flush=True)
    else:
        # Sweep config: submitit_folder is under the sweep dir.
        sweep_dir = Path(output_dir).parent.parent
        sub = sweep_dir / ".submitit"
        if sub.exists():
            for line in _verify_submitit_dir(str(sub)):
                print(line, flush=True)

    # Aim handler diagnostics (hash captured before shutdown).
    if aim_hash:
        print(f"{CHECK} Aim run hash: {aim_hash}", flush=True)
    else:
        print(f"{FAIL} No Aim run attached to handler", flush=True)


if __name__ == "__main__":
    main()
