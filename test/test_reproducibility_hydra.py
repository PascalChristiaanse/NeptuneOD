from orbitdet.reproducibility import hydra


def test_git_commit_strips_trailing_newline(monkeypatch) -> None:
    def fake_check_output(cmd, text=True):
        if cmd == ["git", "rev-parse", "--short", "HEAD"]:
            return "abc123\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(hydra.subprocess, "check_output", fake_check_output)

    assert hydra._git_commit() == "abc123"


def test_git_commit_safe_appends_dirty_suffix_when_repo_is_dirty(monkeypatch) -> None:
    def fake_check_output(cmd, text=True):
        if cmd == ["git", "status", "--porcelain"]:
            return " M scripts/Atanas2026.py\n"
        if cmd == ["git", "rev-parse", "--short", "HEAD"]:
            return "abc123\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(hydra.subprocess, "check_output", fake_check_output)

    assert hydra._git_commit_safe() == "abc123_dirty"


def test_git_commit_safe_returns_commit_when_repo_is_clean(monkeypatch) -> None:
    def fake_check_output(cmd, text=True):
        if cmd == ["git", "status", "--porcelain"]:
            return ""
        if cmd == ["git", "rev-parse", "--short", "HEAD"]:
            return "abc123\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(hydra.subprocess, "check_output", fake_check_output)

    assert hydra._git_commit_safe() == "abc123"


def test_register_resolvers_registers_git_commit(monkeypatch) -> None:
    registered = {}

    def fake_register_new_resolver(name, resolver):
        registered[name] = resolver

    monkeypatch.setattr(hydra.OmegaConf, "register_new_resolver", fake_register_new_resolver)

    hydra.register_resolvers()

    assert "git_commit" in registered
    assert registered["git_commit"] is hydra._git_commit_safe
