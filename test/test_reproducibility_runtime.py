import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from orbitdet.reproducibility import runtime


@pytest.fixture(autouse=True)
def reset_runtime_context():
    original_excepthook = sys.excepthook
    runtime._CONTEXT = None
    yield
    runtime._CONTEXT = None
    sys.excepthook = original_excepthook


def test_enforce_initialization_raises_if_initialize_not_called():
    @runtime.enforce_initialization
    def wrapped():
        return "ok"

    with pytest.raises(RuntimeError, match="not initialized"):
        wrapped()


def test_enforce_initialization_allows_test_mode_initialization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @runtime.enforce_initialization
    def wrapped():
        runtime.initialize_test_mode(seed=123)
        return "ok"

    assert wrapped() == "ok"


def test_initialize_test_mode_installs_console_only_pipeline(tmp_path, monkeypatch):
    """Test mode must configure the production pipeline, console-only (WP6).

    It replaces the old behavior where test mode bypassed logging entirely.
    """
    from orbitdet.reproducibility import logging as od_logging

    captured = {}

    def fake_configure_logging(cfg, *, run_dir, use_queue):
        captured["cfg"] = cfg
        captured["run_dir"] = run_dir
        captured["use_queue"] = use_queue

    monkeypatch.setattr(od_logging, "configure_logging", fake_configure_logging)
    monkeypatch.chdir(tmp_path)

    ctx = runtime.initialize_test_mode(seed=7)

    assert ctx.test_mode is True
    assert captured["cfg"] is None
    assert captured["run_dir"] is None  # console-only: no file sink
    assert captured["use_queue"] is True


# def test_initialize_blocks_dirty_repository(tmp_path, monkeypatch):
#     def fake_check_output(cmd, text=True):
#         if cmd == ["git", "status", "--porcelain"]:
#             return " M scripts/Atanas2026.py\n"
#         if cmd == ["git", "rev-parse", "--short", "HEAD"]:
#             return "abc123\n"
#         if len(cmd) == 3 and cmd[1:] == ["env", "export"] and cmd[0].endswith("conda"):
#             return "name: test\n"
#         raise AssertionError(f"Unexpected command: {cmd}")

#     def raise_dirty():
#         raise RuntimeError(
#             """Repository has uncommitted changes. Commit """
#             """or stash changes before running experiments."""
#         )

#     monkeypatch.setattr(runtime.subprocess, "check_output", fake_check_output)
#     monkeypatch.setattr(runtime, "assert_clean_repo", raise_dirty)
#     monkeypatch.setattr(
#         runtime.HydraConfig,
#         "get",
#         staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))),
#     )

#     with pytest.raises(RuntimeError, match="uncommitted changes"):
#         runtime.initialize(OmegaConf.create({"seed": 42}))


def test_initialize_sets_context_and_writes_metadata(tmp_path, monkeypatch):
    def fake_check_output(cmd, text=True):
        if cmd == ["git", "status", "--porcelain"]:
            return ""
        if cmd == ["git", "rev-parse", "--short", "HEAD"]:
            return "abc123\n"
        if len(cmd) == 3 and cmd[1:] == ["env", "export"] and cmd[0].endswith("conda"):
            return "name: NeptuneOD\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(runtime.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(
        runtime.HydraConfig,
        "get",
        staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))),
    )
    monkeypatch.setattr(runtime, "aim_start_run", MagicMock())

    cfg = OmegaConf.create({"seed": 7, "foo": "bar"})
    ctx = runtime.initialize(cfg)

    assert ctx.git_commit == "abc123"
    assert ctx.seed == 7
    assert ctx.test_mode is False
    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "metadata.yaml").exists()
    assert (tmp_path / "conda_environment.yaml").exists()


def test_save_conda_environment_uses_quiet_fallback_when_conda_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: None)
    check_output = MagicMock(side_effect=AssertionError("should not call subprocess"))
    monkeypatch.setattr(runtime.subprocess, "check_output", check_output)

    runtime.save_conda_environment(tmp_path)

    check_output.assert_not_called()
    content = (tmp_path / "conda_environment.yaml").read_text()
    assert "conda was not found in PATH" in content


def test_setup_logging_installs_pipeline(tmp_path, monkeypatch):
    """setup_logging must delegate to the dictConfig pipeline, not basicConfig.

    The old implementation called ``logging.basicConfig``, which was silently a
    no-op because Hydra had already attached root handlers, so ``cfg.logging.level``
    had no effect.
    """
    from orbitdet.reproducibility import logging as od_logging

    captured = {}

    def fake_configure_logging(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(od_logging, "configure_logging", fake_configure_logging)

    cfg = OmegaConf.create(
        {
            "logging": {
                "level": "INFO",
                "tudatpy_logging_level": "WARNING",
                "muted_loggers": ["matplotlib"],
            }
        }
    )

    runtime.setup_logging(cfg)

    assert captured["cfg"] is cfg


def test_setup_logging_replaces_default_excepthook(monkeypatch):
    monkeypatch.setattr("orbitdet.reproducibility.logging.configure_logging", lambda cfg: None)
    default_hook = sys.excepthook

    runtime.setup_logging(OmegaConf.create({"logging": {"level": "INFO"}}))

    assert sys.excepthook is not default_hook

    # A non-KeyboardInterrupt exception must be routed through the logger.
    logged = {}
    monkeypatch.setattr(
        runtime.logging.getLogger("orbitdet.reproducibility.runtime"),
        "error",
        lambda *a, **k: logged.update(kwargs=k),
    )
    sys.excepthook(ValueError, ValueError("boom"), None)
    assert "exc_info" in logged["kwargs"]


def test_no_fd_level_capture_mechanism_exists():
    """Regression guard for WP1.

    ``FdCapture`` dup2'd fds 1/2 into a pipe and re-injected captured lines into
    the root logger, whose console handler wrote back to the same pipe. When the
    pipe filled (e.g. under Slurm ``srun``, where stdout is a pipe) this
    deadlocked the process. The mechanism must never be reintroduced.
    """
    for name in (
        "FdCapture",
        "_start_native_fd_capture",
        "_stop_native_fd_capture",
        "_NATIVE_FD_CAPTURES",
        "_PYTHON_LOG_LINE_PATTERN",
    ):
        assert not hasattr(runtime, name), f"{name} must not exist in runtime.py"


def test_initialize_returns_existing_context_without_reinitializing(monkeypatch):
    existing = runtime.RuntimeContext(
        git_commit="abc123",
        output_dir=Path("/tmp/unused"),
        seed=99,
        test_mode=False,
    )
    runtime._CONTEXT = existing

    monkeypatch.setattr(
        runtime, "assert_clean_repo", MagicMock(side_effect=AssertionError("should not run"))
    )
    monkeypatch.setattr(
        runtime, "get_git_commit", MagicMock(side_effect=AssertionError("should not run"))
    )
    monkeypatch.setattr(
        runtime.HydraConfig, "get", MagicMock(side_effect=AssertionError("should not run"))
    )
    # Logging is intentionally reconfigured on every call (see
    # test_initialize_configures_logging_even_when_context_is_cached); it must
    # not require the run directory here.
    monkeypatch.setattr(runtime, "setup_logging", MagicMock(return_value=None))

    cfg = OmegaConf.create({"seed": 1})

    assert runtime.initialize(cfg) is existing


def test_initialize_test_mode_returns_existing_context_without_reinitializing(
    monkeypatch, tmp_path
):
    existing = runtime.RuntimeContext(
        git_commit="TEST",
        output_dir=tmp_path,
        seed=7,
        test_mode=True,
    )
    runtime._CONTEXT = existing

    assert runtime.initialize_test_mode(seed=123) is existing


def test_get_context_returns_initialized_context(tmp_path):
    expected = runtime.RuntimeContext(
        git_commit="abc123",
        output_dir=tmp_path,
        seed=7,
        test_mode=False,
    )
    runtime._CONTEXT = expected

    assert runtime.get_context() is expected


def test_get_context_raises_when_uninitialized():
    runtime._CONTEXT = None

    with pytest.raises(RuntimeError, match="not initialized"):
        runtime.get_context()


def test_require_initialized_delegates_to_get_context(monkeypatch):
    sentinel = object()
    mocked_get_context = MagicMock(return_value=sentinel)
    monkeypatch.setattr(runtime, "get_context", mocked_get_context)

    runtime.require_initialized()

    mocked_get_context.assert_called_once_with()


def test_initialize_configures_logging_even_when_context_is_cached(monkeypatch, tmp_path):
    """Logging must be (re)configured for every job, not only the first one.

    Hydra launchers such as joblib can run several jobs in one interpreter. Each
    job has its own output directory, so logging has to be configured before the
    ``_CONTEXT`` cache check - otherwise every job after the first writes into
    the first job's log files, or produces no log file at all.
    """
    configure_calls = []

    monkeypatch.setattr(
        "orbitdet.reproducibility.logging.configure_logging",
        lambda cfg, **kwargs: configure_calls.append(cfg),
    )
    monkeypatch.setattr(runtime, "get_git_commit", MagicMock(return_value="abc123"))
    monkeypatch.setattr(runtime, "save_conda_environment", MagicMock())
    monkeypatch.setattr(runtime, "aim_start_run", MagicMock())
    monkeypatch.setattr(
        runtime.HydraConfig,
        "get",
        staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))),
    )

    runtime._CONTEXT = runtime.RuntimeContext(
        git_commit="cached",
        output_dir=tmp_path,
        seed=1,
        test_mode=False,
    )

    cfg = OmegaConf.create({"seed": 1, "logging": {"level": "INFO"}})
    result = runtime.initialize(cfg)

    # The cached context is still returned...
    assert result.git_commit == "cached"
    # ...but logging was configured for this job regardless.
    assert configure_calls == [cfg]
