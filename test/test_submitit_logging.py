"""Tests for submitit log placement (WP4).

The submitit launcher writes ``<job_id>_<task>_log.out`` / ``.err`` into
``hydra.launcher.submitit_folder``. That folder must therefore contain a
per-job component, or every job in a sweep writes into one shared directory
where the files can only be told apart by PID.
"""

from pathlib import Path

import pytest
import yaml


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def conf_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "conf"


def test_sweep_submitit_folder_has_per_job_segment(conf_dir):
    """The submitit folder must include ``%j`` so each job gets its own dir."""
    cfg = _load(conf_dir / "sweep" / "params_x_data.yaml")

    folder = cfg["hydra"]["launcher"]["submitit_folder"]
    assert "%j" in folder, (
        "submitit_folder must contain '%j'; without it every job's logs are "
        "flattened into a single directory."
    )
    assert ".submitit" in folder


def test_sweep_output_dir_is_self_contained(conf_dir):
    """Submitit logs must live under the sweep dir, next to the job outputs."""
    cfg = _load(conf_dir / "sweep" / "params_x_data.yaml")

    folder = cfg["hydra"]["launcher"]["submitit_folder"]
    assert folder.startswith("${hydra.sweep.dir}")

    # Per-job subdirs keyed by the parameter override, so a run dir is
    # identifiable from its name alone.
    assert cfg["hydra"]["sweep"]["subdir"] == "${hydra.job.override_dirname}"


def test_sweep_keeps_stderr_separate_from_stdout(conf_dir):
    """``stderr_to_stdout`` must stay false so .err files are meaningful."""
    cfg = _load(conf_dir / "sweep" / "params_x_data.yaml")

    assert cfg["hydra"]["launcher"]["stderr_to_stdout"] is False
