"""Tests for :func:`orbitdet.reproducibility.hydra.hydra_dependencies`.

The helper answers one question: which configuration files does a Hydra
composition actually read? The tests build small config trees on disk and assert
on the returned file list, because that list is what Snakemake uses to decide
whether a rule is stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orbitdet.reproducibility.hydra import (
    HydraDependencyError,
    UnsupportedConfigSourceError,
    hydra_dependencies,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    """The config tree from the acceptance criteria."""
    root = tmp_path / "config"
    _write(
        root / "plotting.yaml",
        "defaults:\n"
        "  - dataset: experiment\n"
        "  - visualization: publication\n"
        "  - style: journal\n"
        "  - _self_\n"
        "\n"
        "title: test\n",
    )
    _write(root / "dataset" / "experiment.yaml", "name: experiment\n")
    _write(root / "visualization" / "publication.yaml", "dpi: 300\n")
    _write(root / "style" / "journal.yaml", "font: serif\n")
    return root


def test_returns_every_selected_config_file(config_tree: Path) -> None:
    """All four files -- the top-level one and its three defaults -- are returned."""
    files = hydra_dependencies(config_tree / "plotting.yaml")

    assert files == sorted(
        [
            str((config_tree / "plotting.yaml").resolve()),
            str((config_tree / "dataset" / "experiment.yaml").resolve()),
            str((config_tree / "visualization" / "publication.yaml").resolve()),
            str((config_tree / "style" / "journal.yaml").resolve()),
        ]
    )


def test_returns_absolute_paths(config_tree: Path) -> None:
    """Snakemake inputs must be usable regardless of the working directory."""
    for path in hydra_dependencies(config_tree / "plotting.yaml"):
        assert Path(path).is_absolute()
        assert Path(path).is_file()


def test_accepts_a_relative_path(config_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented usage passes a path relative to the working directory."""
    monkeypatch.chdir(config_tree.parent)

    files = hydra_dependencies("config/plotting.yaml")

    assert str((config_tree / "style" / "journal.yaml").resolve()) in files


def test_deduplicates_shared_configs(tmp_path: Path) -> None:
    """A config selected twice appears once."""
    root = tmp_path / "config"
    _write(
        root / "top.yaml",
        "defaults:\n"
        "  - /shared@a\n"
        "  - /shared@b\n"
        "  - _self_\n",
    )
    _write(root / "shared.yaml", "value: 1\n")

    files = hydra_dependencies(root / "top.yaml")

    assert files.count(str((root / "shared.yaml").resolve())) == 1


def test_includes_nested_defaults(tmp_path: Path) -> None:
    """A config pulled in by a nested defaults list is a dependency too."""
    root = tmp_path / "config"
    _write(
        root / "top.yaml",
        "defaults:\n  - middle: one\n  - _self_\n",
    )
    _write(
        root / "middle" / "one.yaml",
        "defaults:\n  - /leaf\n  - _self_\n",
    )
    _write(root / "leaf.yaml", "value: 1\n")

    files = hydra_dependencies(root / "top.yaml")

    assert str((root / "leaf.yaml").resolve()) in files
    assert str((root / "middle" / "one.yaml").resolve()) in files


def test_changing_the_defaults_list_changes_the_result(config_tree: Path) -> None:
    """Selecting a different config must update the dependency list.

    This is the property that keeps a rule from going stale after a defaults
    edit: the new file appears and the old one disappears.
    """
    _write(config_tree / "style" / "modern.yaml", "font: sans\n")
    top = config_tree / "plotting.yaml"

    before = hydra_dependencies(top)
    top.write_text(
        top.read_text(encoding="utf-8").replace("style: journal", "style: modern"),
        encoding="utf-8",
    )
    after = hydra_dependencies(top)

    assert str((config_tree / "style" / "journal.yaml").resolve()) in before
    assert str((config_tree / "style" / "modern.yaml").resolve()) not in before
    assert str((config_tree / "style" / "modern.yaml").resolve()) in after
    assert str((config_tree / "style" / "journal.yaml").resolve()) not in after


def test_plain_config_without_defaults(tmp_path: Path) -> None:
    """A config with no defaults list still depends on itself."""
    root = tmp_path / "config"
    _write(root / "plain.yaml", "a: 1\nb: 2\n")

    assert hydra_dependencies(root / "plain.yaml") == [str((root / "plain.yaml").resolve())]


def test_finds_config_root_above_the_file(tmp_path: Path) -> None:
    """A config inside a group resolves its absolute defaults from the root.

    ``conf/experiment/x.yaml`` refers to ``/figures/y``, which Hydra resolves
    against the config root, not the file's own directory.
    """
    root = tmp_path / "conf"
    _write(
        root / "experiment" / "run.yaml",
        "defaults:\n  - /figures/plot\n  - _self_\n",
    )
    _write(root / "figures" / "plot.yaml", "dpi: 100\n")

    files = hydra_dependencies(root / "experiment" / "run.yaml")

    assert str((root / "figures" / "plot.yaml").resolve()) in files
    assert str((root / "experiment" / "run.yaml").resolve()) in files


def test_honours_hydra_searchpath(tmp_path: Path) -> None:
    """Configs found through ``hydra.searchpath`` are dependencies."""
    root = tmp_path / "config"
    extra = tmp_path / "extra"
    _write(
        root / "top.yaml",
        "defaults:\n  - extra_group: thing\n  - _self_\n"
        "\n"
        "hydra:\n"
        "  searchpath:\n"
        f"    - file://{extra}\n",
    )
    _write(extra / "extra_group" / "thing.yaml", "x: 1\n")

    files = hydra_dependencies(root / "top.yaml")

    assert str((extra / "extra_group" / "thing.yaml").resolve()) in files


def test_overrides_select_a_different_config(tmp_path: Path) -> None:
    """An override can change which config is composed."""
    root = tmp_path / "config"
    _write(root / "top.yaml", "defaults:\n  - style: journal\n  - _self_\n")
    _write(root / "style" / "journal.yaml", "font: serif\n")
    _write(root / "style" / "modern.yaml", "font: sans\n")

    files = hydra_dependencies(root / "top.yaml", overrides=["style=modern"])

    assert str((root / "style" / "modern.yaml").resolve()) in files
    assert str((root / "style" / "journal.yaml").resolve()) not in files


def test_excludes_hydra_internal_configs(config_tree: Path) -> None:
    """Hydra's own configs ship with the package and are not project files."""
    for path in hydra_dependencies(config_tree / "plotting.yaml"):
        assert "site-packages" not in path


def test_missing_config_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        hydra_dependencies(tmp_path / "does_not_exist.yaml")


def test_unresolvable_defaults_raise(tmp_path: Path) -> None:
    """A defaults entry with no matching file fails loudly, not silently."""
    root = tmp_path / "config"
    _write(root / "top.yaml", "defaults:\n  - style: missing\n  - _self_\n")

    with pytest.raises(HydraDependencyError):
        hydra_dependencies(root / "top.yaml")


def test_configstore_source_is_reported_as_unsupported(tmp_path: Path) -> None:
    """A ``structured://`` config has no file, so it cannot be tracked.

    Silently dropping it would leave the rule stale, so the helper raises.
    """
    from dataclasses import dataclass

    from hydra.core.config_store import ConfigStore

    @dataclass
    class Registered:
        value: int = 1

    ConfigStore.instance().store(name="test_registered_config", node=Registered)

    root = tmp_path / "config"
    _write(root / "top.yaml", "defaults:\n  - test_registered_config\n  - _self_\n")

    with pytest.raises(UnsupportedConfigSourceError) as excinfo:
        hydra_dependencies(root / "top.yaml")

    assert excinfo.value.config_name == "test_registered_config"
    assert excinfo.value.scheme == "structured"


def test_repeated_calls_are_stable(config_tree: Path) -> None:
    """The helper must be safe to call once per rule at parse time."""
    first = hydra_dependencies(config_tree / "plotting.yaml")
    second = hydra_dependencies(config_tree / "plotting.yaml")

    assert first == second


def test_real_repository_config_composes() -> None:
    """The repository's own experiment configs resolve to real files."""
    repo_root = Path(__file__).resolve().parents[1]
    config = repo_root / "conf" / "experiment" / "fit_to_ephemeris.yaml"
    if not config.is_file():  # pragma: no cover - config may be renamed
        pytest.skip("experiment config not present")

    files = hydra_dependencies(config)

    assert files
    assert all(Path(path).is_file() for path in files)
    assert str(config.resolve()) in files
