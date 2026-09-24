#!/bin/bash
#
# DelftBlue validation soak (WP7) — submit via:  sbatch scripts/run_delftblue_validation.sbatch
#
# Runs a 2-job multirun (one success, one intentional crash) using the new
# logging pipeline. After completion the results directory contains diagnostic
# output files that the operator inspects for the acceptance criteria.
#
# Acceptance criteria (logged at the end of each job):
#   1. run.log has INFO+WARNING+ERROR; errors.log has ERROR only.
#   2. .err file has WARNING and ERROR (not just noise).
#   3. Crashed job's .err has the full traceback.
#   4. Each job's logs are in its own .submitit/<job_id>/ folder.
#   5. Aim run hash is printed (can be checked with aim reindex + get_run).
#
# No freeze: the old FdCapture deadlock under srun (pipe-based stdout) is gone.
# If this job hangs, the new logging pipeline still has a regression.
# If it completes, the acceptance diagnostics are the final gate.

#SBATCH --job-name="NeptuneOD-logging-soak"
#SBATCH --partition=compute
#SBATCH --time=00:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3996M
#SBATCH --account=Education-AE-MSc-AE
#SBATCH --output=results/delftblue_validation/%x_%j_%a.out
#SBATCH --error=results/delftblue_validation/%x_%j_%a.err

export HYDRA_FULL_ERROR=1

module load miniconda3
conda activate NeptuneOD
cd ~/NeptuneOD || { echo "FAIL: ~/NeptuneOD not found"; exit 1; }

echo "=== WP7 validation: logging-pipeline soak ==="
echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Python: $(which python) ($(python --version 2>&1))"

# Run the 2-job sweep.
# --multirun launches both jobs via submitit_local (per-job process isolation).
# The config is: one job with fail=false, one with fail=true.
echo "=== Launching multirun ==="
time python scripts/delftblue_validation.py --multirun --config-name sweep/validation_fast
EXIT_CODE=$?

echo "=== Multirun exit code: ${EXIT_CODE} ==="

# Print the result directory tree for quick inspection.
find results -type d -name 'delftblue_validation' -print -quit | while read -r root; do
  echo "=== Result tree ==="
  find "$root" -maxdepth 5 -type f \( -name 'run.log' -o -name 'errors.log' -o \
    -name '*_log.err' -o -name '*_log.out' -o -name '.aim_run_hash' \) | sort
done

# Print the diagnostic CHECK lines from the run logs.
echo "=== Diagnostic output from each run ==="
find results -name 'run.log' -newer /tmp -exec grep -l 'VALIDATION-' {} \; | \
  sort | while read -r log; do
  echo "--- $(dirname "${log}") ---"
  grep '\[VALIDATION-PASS\]\|\[VALIDATION-FAIL\]\|\[CHECK\]' "$log" | \
    sed 's/^.*\[VALIDATION-PASS\]/PASS/;s/^.*\[VALIDATION-FAIL\]/FAIL/;s/^.*\[CHECK\]/CHECK/'
done

echo "=== WP7 validation finished (exit code ${EXIT_CODE}) ==="
echo "If all lines show CHECK ✓ and PASS, the logging pipeline passes the soak."
echo "If any FAIL line appears or the job timed out (> 59 min), investigate."
exit ${EXIT_CODE}