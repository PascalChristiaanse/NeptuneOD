from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from orbitdet.reproducibility import runtime


@pytest.fixture(autouse=True)
def reset_runtime_context():
    runtime._CONTEXT = None
    yield
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
        if cmd == ["conda", "env", "export"]:
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
