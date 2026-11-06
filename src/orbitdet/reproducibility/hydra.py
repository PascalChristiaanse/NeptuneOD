"""Hydra integration: OmegaConf resolvers and configuration dependency discovery.

Snakemake's ``code`` rerun trigger tracks the rule definition and the file named
by ``script:``, but it does **not** follow Hydra ``defaults`` lists. A rule that
reads ``config/plotting.yaml`` therefore stays "up to date" when
``config/style/journal.yaml`` -- selected by that file's ``defaults`` list --
changes. 

:func:`hydra_dependencies` closes that gap::

    rule plot_results:
        input:
            data="results/{sample}/statistics.parquet",
            config=hydra_dependencies("config/plotting.yaml")
        output:
            "results/{sample}/plot.png"
        script:
            "scripts/plot.py"

"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.config_search_path_impl import ConfigSearchPathImpl
from hydra._internal.defaults_list import create_defaults_list
from hydra.core.plugins import Plugins
from hydra.plugins.config_source import ConfigSource
from omegaconf import OmegaConf

__all__ = [
    "register_resolvers",
    "hydra_dependencies",
    "HydraDependencyError",
    "UnsupportedConfigSourceError",
]

#: Prefix of Hydra's own configuration names, which are never project files.
_HYDRA_PREFIX = "hydra/"


# ---------------------------------------------------------------------------
# OmegaConf resolvers
# ---------------------------------------------------------------------------


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def _git_commit_safe():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()

    commit = _git_commit()
    return f"{commit}_dirty" if dirty else commit


def register_resolvers():
    OmegaConf.register_new_resolver("git_commit", _git_commit_safe)


# ---------------------------------------------------------------------------
# Configuration dependency discovery
# ---------------------------------------------------------------------------


class HydraDependencyError(RuntimeError):
    """Raised when a Hydra configuration's dependencies cannot be determined."""


class UnsupportedConfigSourceError(HydraDependencyError):
    """Raised when a selected config has no file on disk to depend on.

    Attributes:
        config_name: The Hydra config name that could not be resolved to a file.
        scheme: The source scheme that was found (e.g. ``"pkg"``).
    """

    def __init__(self, config_name: str, scheme: str) -> None:
        self.config_name = config_name
        self.scheme = scheme
        super().__init__(
            f"Config '{config_name}' is provided by a '{scheme}://' source, which has no "
            "file on disk. Snakemake cannot track it as an input. Register the "
            "configuration as a YAML file, or declare the affected files explicitly "
            "in the rule's input."
        )


def _ensure_plugins() -> None:
    """Register Hydra's config-source plugins.

    ``ConfigLoaderImpl`` resolves source schemes through
    :class:`~hydra.core.plugins.Plugins`. The registry is normally populated as a
    side effect of ``hydra.initialize*``; calling it directly keeps this function
    usable from a plain Snakemake ``input:`` expression, where no Hydra context
    exists. The call is idempotent.
    """
    Plugins.instance()


def _config_name(config_path: Path, config_root: Path) -> str:
    """Hydra config name for *config_path* relative to *config_root*.

    Hydra names configs without their extension and with ``/`` separators, e.g.
    ``experiment/fit_to_ephemeris``.
    """
    relative = config_path.relative_to(config_root)
    return relative.with_suffix("").as_posix()


def _search_path(config_root: Path) -> ConfigSearchPathImpl:
    """Build the config search path Hydra would use for *config_root*.

    Mirrors what ``hydra.initialize_config_dir`` installs: Hydra's own configs,
    then the project directory, then the structured-config schema source.
    """
    search_path = ConfigSearchPathImpl()
    search_path.append(provider="hydra", path="pkg://hydra.conf")
    search_path.append(provider="main", path=f"file://{config_root}")
    search_path.append(provider="schema", path="structured://")
    return search_path


def _compose_defaults(
    config_root: Path,
    config_name: str,
    overrides: Sequence[str],
):
    """Compose *config_name* and return ``(defaults_list, repository)``.

    The repository is returned alongside the defaults list because it is the one
    that already accounts for ``hydra.searchpath``: ``compute_defaults_list``
    builds a caching repository internally, and resolving names against the
    loader's own repository would miss search-path entries.
    """
    loader = ConfigLoaderImpl(config_search_path=_search_path(config_root))
    parsed_overrides, repository = loader._parse_overrides_and_create_caching_repo(
        config_name, list(overrides)
    )
    defaults_list = create_defaults_list(
        repo=repository,
        config_name=config_name,
        overrides_list=parsed_overrides,
        prepend_hydra=True,
        skip_missing=False,
    )
    return defaults_list, repository


def _source_for(config_name: str, repository) -> ConfigSource | None:
    """Return the config source providing *config_name*, or ``None``."""
    for source in repository.get_sources():
        if source.is_config(config_name):
            return source
    return None


def _file_for(config_name: str, source: ConfigSource) -> str:
    """Absolute path of the file *config_name* resolves to within *source*.

    Raises:
        UnsupportedConfigSourceError: If *source* is not a ``file://`` source.
    """
    if source.scheme() != "file":
        raise UnsupportedConfigSourceError(config_name, source.scheme())

    filename = ConfigSource._normalize_file_name(config_name)
    return os.path.realpath(os.path.join(source.path, filename))


def _candidate_roots(config_path: Path) -> Iterable[Path]:
    """Directories to try as the Hydra config root, nearest first.

    A top-level config may live at the root (``config/plotting.yaml``) or inside
    a group (``conf/experiment/fit_to_ephemeris.yaml``). Hydra resolves group
    references such as ``/figures/residuals`` against the *config root*, not the
    file's own directory, so the root is found by trying each ancestor until the
    composition succeeds.
    """
    yield config_path.parent
    yield from config_path.parent.parents


def hydra_dependencies(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
) -> list[str]:
    """Return all configuration source files used by a Hydra configuration.

    Includes files selected through nested ``defaults`` lists, so editing any of
    them invalidates a Snakemake rule that declares this list as an input.

    Parameters
    ----------
    config_path:
        Path to the top-level Hydra YAML configuration, relative to the current
        working directory (or absolute).
    overrides:
        Optional Hydra overrides applied before composition, e.g.
        ``["experiment=fit_to_ephemeris"]``. Use this when the caller selects a
        config group that the top-level file does not default to.

    Returns
    -------
    list[str]
        Sorted, de-duplicated absolute paths of every configuration file the
        composition reads. Suitable for a Snakemake ``input:`` list.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    HydraDependencyError
        If the configuration cannot be composed, or if a selected config has no
        file on disk (see :class:`UnsupportedConfigSourceError`).

    Examples
    --------
    >>> hydra_dependencies("config/plotting.yaml")
    ['/repo/config/dataset/experiment.yaml',
     '/repo/config/plotting.yaml',
     '/repo/config/style/journal.yaml',
     '/repo/config/visualization/publication.yaml']
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Hydra configuration not found: {config_path}")

    _ensure_plugins()

    errors: list[str] = []
    for config_root in _candidate_roots(path):
        try:
            config_name = _config_name(path, config_root)
        except ValueError:  # pragma: no cover - ancestors always contain the file
            continue

        try:
            defaults_list, repository = _compose_defaults(
                config_root, config_name, overrides or ()
            )
        except Exception as exc:  # Hydra raises several unrelated types
            errors.append(f"  root={config_root}: {type(exc).__name__}: {exc}")
            continue

        files: set[str] = set()
        for default in defaults_list.defaults:
            name = default.config_path
            if name is None or name.startswith(_HYDRA_PREFIX):
                continue

            source = _source_for(name, repository)
            if source is None:
                raise HydraDependencyError(
                    f"Config '{name}' is in the composed defaults list but no source "
                    f"provides it (searched from {config_root})."
                )
            files.add(_file_for(name, source))

        return sorted(files)

    raise HydraDependencyError(
        f"Could not compose Hydra configuration '{config_path}' from any candidate "
        "config root. Tried:\n" + "\n".join(errors)
    )
