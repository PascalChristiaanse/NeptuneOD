import contextlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from orbitdet.reproducibility import runtime


@pytest.fixture(autouse=True)
def reset_runtime_context():
    runtime._stop_native_fd_capture()
    runtime._CONTEXT = None
    yield
    runtime._stop_native_fd_capture()
    runtime._CONTEXT = None


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


def test_initialize_blocks_dirty_repository(monkeypatch):
    def fake_check_output(cmd, text=True):
        if cmd == ["git", "status", "--porcelain"]:
            return " M scripts/Atanas2026.py\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(runtime.subprocess, "check_output", fake_check_output)

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        runtime.initialize(OmegaConf.create({"seed": 42}))


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


def test_initialize_starts_native_fd_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "get_git_commit", MagicMock(return_value="abc123"))
    monkeypatch.setattr(runtime, "save_conda_environment", MagicMock())
    monkeypatch.setattr(runtime, "setup_logging", MagicMock())
    start_capture = MagicMock()
    monkeypatch.setattr(runtime, "_start_native_fd_capture", start_capture)
    monkeypatch.setattr(
        runtime.HydraConfig,
        "get",
        staticmethod(lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))),
    )

    cfg = OmegaConf.create({"seed": 11})

    ctx = runtime.initialize(cfg)

    assert ctx.seed == 11
    start_capture.assert_called_once()


def test_setup_logging_configures_root_and_muted_loggers(monkeypatch):
    basic_config = MagicMock()
    logger_map = {}

    def fake_get_logger(name):
        logger = logger_map.get(name)
        if logger is None:
            logger = MagicMock()
            logger_map[name] = logger
        return logger

    fake_logging = SimpleNamespace(
        basicConfig=basic_config,
        getLogger=fake_get_logger,
        WARNING=runtime.logging.WARNING,
    )
    monkeypatch.setattr(runtime, "logging", fake_logging)

    cfg = OmegaConf.create(
        {
            "logging": {
                "level": "INFO",
                "tudatpy_logging_level": "WARNING",
                "muted_loggers": ["matplotlib", "orbitdet.data.kernel"],
            }
        }
    )

    runtime.setup_logging(cfg)

    basic_config.assert_called_once_with(
        level="INFO",
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    assert logger_map["tudatpy"].setLevel.called
    logger_map["tudatpy"].setLevel.assert_called_once_with("WARNING")
    logger_map["matplotlib"].setLevel.assert_called_once_with(fake_logging.WARNING)
    logger_map["orbitdet.data.kernel"].setLevel.assert_called_once_with(fake_logging.WARNING)


def test_fd_capture_forwards_lines_and_restores_fd():
    logger = MagicMock()
    mirror_read_fd, mirror_write_fd = os.pipe()
    target_fd = os.dup(mirror_write_fd)

    try:
        capture = runtime.FdCapture(target_fd, logger, runtime.logging.INFO)
        capture.start()

        os.write(target_fd, b"first line\nsecond line\n")

        capture.stop()
        os.close(target_fd)
        os.close(mirror_write_fd)

        mirrored = os.read(mirror_read_fd, 4096).decode().replace("\r\n", "\n")

        logger.log.assert_any_call(runtime.logging.INFO, "first line")
        logger.log.assert_any_call(runtime.logging.INFO, "second line")
        assert mirrored == "first line\nsecond line\n"
    finally:
        os.close(mirror_read_fd)
        with contextlib.suppress(OSError):
            os.close(mirror_write_fd)
        with contextlib.suppress(OSError):
            os.close(target_fd)


def test_fd_capture_ignores_python_formatted_log_lines():
    logger = MagicMock()
    mirror_read_fd, mirror_write_fd = os.pipe()
    target_fd = os.dup(mirror_write_fd)

    try:
        capture = runtime.FdCapture(target_fd, logger, runtime.logging.INFO)
        capture.start()

        os.write(
            target_fd,
            b"[2026-05-21 15:57:18,581][orbitdet.reproducibility.runtime][INFO] - already logged\n",
        )

        capture.stop()
        os.close(target_fd)
        os.close(mirror_write_fd)

        mirrored = os.read(mirror_read_fd, 4096).decode()

        logger.log.assert_not_called()
        assert "already logged" in mirrored
    finally:
        os.close(mirror_read_fd)
        with contextlib.suppress(OSError):
            os.close(mirror_write_fd)
        with contextlib.suppress(OSError):
            os.close(target_fd)


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
