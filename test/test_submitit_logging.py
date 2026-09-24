"""Tests for submitit log placement and job-identity tracking (WP4).

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


def test_job_identity_helper_extracts_hydra_fields():
    """_job_identity must report the Hydra job fields it can resolve."""
    from unittest.mock import patch

    from orbitdet.reproducibility import aim

    class FakeJob:
        name = "solve_least_squares"
        num = 3
        id = "12345"
        override_dirname = "data=classical,parameters=initial_state"

    class FakeHydra:
        job = FakeJob()

    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=FakeHydra()):
        identity = aim._job_identity()

    assert identity["job_name"] == "solve_least_squares"
    assert identity["job_num"] == 3
    assert identity["job_id"] == "12345"
    assert identity["override_dirname"] == "data=classical,parameters=initial_state"


def test_job_identity_omits_unavailable_fields():
    """`hydra.job.id` only exists under submitit; it must be omitted when absent."""
    from unittest.mock import patch

    from orbitdet.reproducibility import aim

    class FakeJob:
        name = "solve_least_squares"
        override_dirname = "seed=1"

    class FakeHydra:
        job = FakeJob()

    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=FakeHydra()):
        identity = aim._job_identity()

    assert "job_id" not in identity
    assert "job_num" not in identity
    assert identity["job_name"] == "solve_least_squares"
    assert identity["override_dirname"] == "seed=1"


def test_job_identity_never_raises_outside_hydra():
    """Job identity is useful metadata, not something that may break a run."""
    from unittest.mock import patch

    from orbitdet.reproducibility import aim

    with patch("hydra.core.hydra_config.HydraConfig.get", side_effect=RuntimeError("no hydra")):
        assert aim._job_identity() == {}


def test_aim_start_run_records_job_identity(tmp_path, monkeypatch):
    """aim_start_run must persist job identity so runs trace back to jobs."""
    from types import SimpleNamespace

    import aim.sdk as aim_sdk

    import orbitdet.reproducibility.aim as aim_module

    stored = {}

    class FakeRun:
        def __init__(self, **kwargs):
            self.hash = "deadbeef"
            self.artifacts_uri = None

        def add_tag(self, tag):
            pass

        def set_artifacts_uri(self, uri):
            self.artifacts_uri = uri

        def __setitem__(self, key, value):
            stored[key] = value

    monkeypatch.setattr(aim_sdk, "Run", FakeRun)
    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig.get",
        staticmethod(
            lambda: SimpleNamespace(
                job=SimpleNamespace(
                    name="demo",
                    num=2,
                    id="999",
                    override_dirname="data=classical",
                )
            )
        ),
    )

    from omegaconf import OmegaConf

    aim_module.aim_start_run(OmegaConf.create({}), "abc123", tmp_path, 42)

    assert stored["job_name"] == "demo"
    assert stored["job_num"] == 2
    assert stored["job_id"] == "999"
    assert stored["override_dirname"] == "data=classical"
