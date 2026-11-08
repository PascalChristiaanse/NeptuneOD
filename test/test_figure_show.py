"""Tests for the interactive-show guard in the figure scripts."""

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# ``scripts/`` is not on the pytest pythonpath (only ``src`` is), so the shared
# figure helper is loaded from its file path.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("_figure_common", _SCRIPTS / "_figure_common.py")
_figure_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_common)

running_under_snakemake = _figure_common.running_under_snakemake
show_or_serve = _figure_common.show_or_serve


class _FakeStdout:
    """Stand-in for sys.stdout with a controllable isatty()."""

    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


@pytest.fixture
def no_snakemake_env(monkeypatch):
    """Remove any SNAKEMAKE_* variables so the tty signal is what is tested."""
    for key in list(__import__("os").environ):
        if key.startswith("SNAKEMAKE_"):
            monkeypatch.delenv(key, raising=False)


# -- running_under_snakemake ------------------------------------------------


def test_tty_stdout_is_interactive(monkeypatch, no_snakemake_env):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=True))
    assert running_under_snakemake() is False


def test_redirected_stdout_is_non_interactive(monkeypatch, no_snakemake_env):
    """Every Snakemake rule redirects stdout to its log file."""
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=False))
    assert running_under_snakemake() is True


def test_snakemake_env_var_is_detected(monkeypatch):
    monkeypatch.setenv("SNAKEMAKE_TEST_MARKER", "1")
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=True))
    assert running_under_snakemake() is True


# -- show_or_serve ----------------------------------------------------------


@pytest.fixture
def show_calls(monkeypatch):
    """Record plt.show() calls instead of blocking on a window."""
    calls = []
    monkeypatch.setattr(_figure_common.plt, "show", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(_figure_common.plt, "get_backend", lambda: "QtAgg")
    return calls


def test_does_not_show_when_non_interactive(monkeypatch, no_snakemake_env, show_calls):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=False))
    show_or_serve()
    assert show_calls == []


def test_shows_when_interactive(monkeypatch, no_snakemake_env, show_calls):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=True))
    show_or_serve()
    assert show_calls == [{"block": True}]


def test_figure_show_false_suppresses_even_when_interactive(
    monkeypatch, no_snakemake_env, show_calls
):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=True))
    show_or_serve(OmegaConf.create({"figure_show": False}))
    assert show_calls == []


def test_figure_show_true_forces_a_show_when_non_interactive(
    monkeypatch, no_snakemake_env, show_calls
):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=False))
    show_or_serve(OmegaConf.create({"figure_show": True}))
    assert show_calls == [{"block": True}]


def test_webagg_backend_never_calls_show(monkeypatch, no_snakemake_env, show_calls):
    monkeypatch.setattr(sys, "stdout", _FakeStdout(is_tty=True))
    monkeypatch.setattr(_figure_common.plt, "get_backend", lambda: "WebAgg")
    show_or_serve()
    assert show_calls == []
