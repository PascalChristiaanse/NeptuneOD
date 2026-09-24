"""Lean tests for the AimStack integration (orbitdet.reproducibility.aim)."""

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from orbitdet.reproducibility import aim

# ---------------------------------------------------------------------------
# _flatten_omegaconf — pure helper, no mocking needed
# ---------------------------------------------------------------------------


def test_flatten_omegaconf_empty():
    assert aim._flatten_omegaconf(OmegaConf.create({})) == {}


def test_flatten_omegaconf_flat():
    cfg = OmegaConf.create({"a": 1, "b": 2.5, "c": "hello"})
    assert aim._flatten_omegaconf(cfg) == {"a": 1, "b": 2.5, "c": "hello"}


def test_flatten_omegaconf_nested():
    cfg = OmegaConf.create({"outer": {"inner": 42, "other": "x"}})
    assert aim._flatten_omegaconf(cfg) == {"outer.inner": 42, "outer.other": "x"}


def test_flatten_omegaconf_with_lists():
    cfg = OmegaConf.create({"nums": [1, 2, 3], "nested": {"vals": [4, 5]}})
    result = aim._flatten_omegaconf(cfg)
    assert result == {"nums": "1,2,3", "nested.vals": "4,5"}


def test_flatten_omegaconf_custom_sep():
    cfg = OmegaConf.create({"a": {"b": 1}})
    assert aim._flatten_omegaconf(cfg, sep="/") == {"a/b": 1}


# ---------------------------------------------------------------------------
# aim_finalize
# ---------------------------------------------------------------------------


def test_aim_finalize_none_is_noop():
    """Should not raise when passed None."""
    aim.aim_finalize(None)


def test_aim_finalize_calls_close():
    run = MagicMock()
    aim.aim_finalize(run)
    run.close.assert_called_once()


def test_aim_finalize_swallows_close_exception():
    run = MagicMock()
    run.close.side_effect = RuntimeError("boom")
    aim.aim_finalize(run)  # should not raise


# ---------------------------------------------------------------------------
# get_aim_run — when no RuntimeContext exists
# ---------------------------------------------------------------------------


def test_get_aim_run_returns_none_when_no_context():
    assert aim.get_aim_run() is None


# ---------------------------------------------------------------------------
# Logging functions — graceful degradation when no run is active
# ---------------------------------------------------------------------------


def test_aim_log_metrics_no_run(caplog):
    aim.aim_log_metrics({"loss": 0.5})
    assert "No active Aim run" in caplog.text


def test_aim_log_figure_no_run(caplog):
    aim.aim_log_figure(MagicMock())
    assert "No active Aim run" in caplog.text


def test_aim_log_artifact_no_run(caplog):
    aim.aim_log_artifact("/nonexistent/file.pdf")
    assert "No active Aim run" in caplog.text


# ---------------------------------------------------------------------------
# aim_log_artifact — file-not-found branch
# ---------------------------------------------------------------------------


def test_aim_log_artifact_skips_missing_file(caplog):
    """When a run *is* active but the file doesn't exist, warn and skip."""
    MagicMock()
    aim.aim_log_artifact("/nonexistent/file.pdf")
    # run.log_artifact should not be called since there's no active run
    assert "No active Aim run" in caplog.text


# ---------------------------------------------------------------------------
# aim_start_run — terminal capture must stay off
# ---------------------------------------------------------------------------


def test_aim_start_run_disables_terminal_capture(tmp_path, monkeypatch):
    """Aim must not patch sys.stdout/sys.stderr; the logging handler covers it."""
    from types import SimpleNamespace

    import aim.sdk as aim_sdk

    import orbitdet.reproducibility.aim as aim_module

    captured_kwargs = {}

    class FakeRun:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.hash = "abc123"
            self.artifacts_uri = None

        def add_tag(self, tag):
            pass

        def set_artifacts_uri(self, uri):
            self.artifacts_uri = uri

        def __setitem__(self, key, value):
            pass

    monkeypatch.setattr(aim_sdk, "Run", FakeRun)
    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig.get",
        staticmethod(lambda: SimpleNamespace(job=SimpleNamespace(name="demo"))),
    )

    cfg = OmegaConf.create({})
    run = aim_module.aim_start_run(cfg, "abc123", tmp_path, 42)

    assert isinstance(run, FakeRun)
    assert captured_kwargs["capture_terminal_logs"] is False
    assert captured_kwargs["experiment"] == "demo"
