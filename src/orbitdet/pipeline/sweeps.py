"""Enumerating Hydra sweep configurations for the Snakemake workflow.

Background
----------
Sweeps in this project are plain Hydra multirun configs under ``conf/sweep/``.
A sweep file is an experiment config plus a ``hydra.sweeper.params`` block::

    hydra:
      sweeper:
        params:
          estimation.parameters_to_estimate.neptune_GM: true,false
          estimation.parameters_to_estimate.triton_GM: true,false

Hydra takes the Cartesian product of those value lists. Running one therefore
means running that many jobs, which is exactly the shape Snakemake is good at --
but only if the combination list is *enumerable before the run starts*. Hydra
enumerates internally at launch time; the workflow needs the list up front to
declare its targets.

Two override styles appear in this repository and both reduce to the same thing:

*config-group override* -- ``parameters: initial_state,state_plus_gravity``
    Swaps ``conf/parameters/<name>.yaml``. Override string: ``parameters=<name>``.
*dotted-key override* -- ``estimation.parameters_to_estimate.neptune_GM: true,false``
    Patches one nested key. Override string:
    ``estimation.parameters_to_estimate.neptune_GM=<value>``.

So enumeration is uniform: the Cartesian product of the value lists, each
rendered as a ``key=value`` override string that
:func:`orbitdet.pipeline.config.load_config` accepts directly.

Why the raw YAML is read rather than composed
---------------------------------------------
Composing a sweep config through Hydra would resolve the launcher, and these
files pin ``submitit_local`` -- an optional dependency the workflow does not
need (Snakemake owns the parallelism). Reading ``hydra.sweeper.params`` from the
file avoids that dependency entirely and keeps enumeration a pure function of
the YAML.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import yaml

#: Repo root and Hydra config directory.
#:
#: Computed here rather than imported from :mod:`orbitdet.pipeline.config`.
#: That module imports Hydra, and this one is imported by the *Snakefile* while
#: it builds the DAG -- pulling Hydra (and its plugin machinery) into every
#: ``snakemake --dry-run`` for no benefit. The enumerator only needs the
#: directory, so the one line of duplication buys a much lighter parse.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONF_DIR = REPO_ROOT / "conf"

logger = logging.getLogger(__name__)

#: Filename of the manifest every sweep writes, mapping combo id -> overrides.
MANIFEST_NAME = "sweep_manifest.json"


class SweepCombo:
    """One point of the sweep's Cartesian product.

    A plain class rather than a :func:`dataclasses.dataclass` on purpose: the
    Snakefile loads this module *by path* (to enumerate combinations before any
    stage process exists), and ``@dataclass`` needs to resolve its own module to
    look up annotations, which fails for a module loaded that way.

    Attributes
    ----------
    index:
        Position in the enumeration, zero-based. Used as the directory name, so
        it is zero-padded for readability and to keep a stable sort order.
    overrides:
        Hydra-style override strings, e.g.
        ``("estimation.parameters_to_estimate.neptune_GM=true",)``.
    """

    __slots__ = ("index", "overrides")

    def __init__(self, index: int, overrides: tuple[str, ...]):
        self.index = index
        self.overrides = overrides

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SweepCombo):
            return NotImplemented
        return (self.index, self.overrides) == (other.index, other.overrides)

    def __hash__(self) -> int:
        return hash((self.index, self.overrides))

    def __repr__(self) -> str:
        return f"SweepCombo(index={self.index}, overrides={self.overrides!r})"

    @property
    def combo_id(self) -> str:
        """Directory name for this combination, e.g. ``"00"``.

        Two digits is enough for any sweep in this repository (the largest is
        36 combinations); wider indices simply render wider. Zero-padding keeps
        ``ls`` and glob order the same as enumeration order.
        """
        return f"{self.index:02d}"

    @property
    def label(self) -> str:
        """Short human-readable label for figure axes.

        Combines the overridden keys' last path component, e.g.
        ``"neptune_GM=true,triton_GM=false"``. Long for a bar chart with 32
        combinations, so figures fall back to :attr:`combo_id` and publish the
        full mapping in the manifest.
        """
        parts = []
        for override in self.overrides:
            key, _, value = override.partition("=")
            parts.append(f"{key.rsplit('.', 1)[-1]}={value}")
        return ",".join(parts)


def sweep_config_path(sweep: str) -> Path:
    """Path of the sweep config for *sweep*, e.g. ``conf/sweep/fit_to_ephemeris.yaml``."""
    return CONF_DIR / "sweep" / f"{sweep}.yaml"


def load_sweep_params(sweep: str) -> dict[str, list[str]]:
    """Read ``hydra.sweeper.params`` from a sweep config.

    Values are normalised to lists of strings so the enumeration does not care
    whether the YAML wrote a scalar or a sequence.

    Raises
    ------
    FileNotFoundError
        If the sweep config does not exist.
    ValueError
        If the config declares no sweep parameters. A sweep with nothing to vary
        would silently produce one job, which is almost certainly a mistake.
    """
    path = sweep_config_path(sweep)
    if not path.exists():
        raise FileNotFoundError(f"No sweep config at {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    params = (raw.get("hydra") or {}).get("sweeper", {}).get("params") or {}
    if not params:
        raise ValueError(
            f"Sweep '{sweep}' declares no hydra.sweeper.params in {path}; "
            "there is nothing to sweep over."
        )

    normalised: dict[str, list[str]] = {}
    for key, value in params.items():
        normalised[str(key)] = _split_values(value)
    return normalised


def _split_values(value: object) -> list[str]:
    """Normalise a sweep parameter's values to a list of strings.

    Hydra accepts three spellings, and all three appear in this repository::

        key: true,false          # compact comma-separated string (most common)
        key: [true, false]       # YAML sequence
        key: single_value        # scalar, i.e. no variation

    The compact form is *not* a YAML list -- it is a plain string that Hydra's
    sweeper splits on commas. Reading it with ``yaml.safe_load`` yields
    ``"true,false"``, so splitting is required here; treating it as a single
    value silently collapses the sweep to one combination.
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value)
    return [part.strip() for part in text.split(",")] if "," in text else [text.strip()]


def enumerate_combos(sweep: str) -> list[SweepCombo]:
    """List every combination of a sweep, in Hydra's product order.

    The order is the Cartesian product with the *last* declared parameter
    varying fastest, which is what Hydra's own sweeper produces. Keeping that
    order matters because the combination index is used as the directory name,
    so a stable order keeps existing results addressable.
    """
    params = load_sweep_params(sweep)

    keys = list(params)
    value_lists = [params[key] for key in keys]

    combos: list[SweepCombo] = []
    for index, values in enumerate(itertools.product(*value_lists)):
        overrides = tuple(f"{key}={value}" for key, value in zip(keys, values, strict=True))
        combos.append(SweepCombo(index=index, overrides=overrides))

    logger.info("Sweep %r: %d combination(s) over %d parameter(s)", sweep, len(combos), len(keys))
    return combos


def write_manifest(sweep: str, path: str | Path, combos: list[SweepCombo] | None = None) -> Path:
    """Write the combo-id -> overrides mapping next to a sweep's results.

    Without this the directory names are opaque indices. The manifest makes a
    result directory self-describing and lets the comparison figure annotate its
    bars with the parameters they actually varied.
    """
    combos = combos if combos is not None else enumerate_combos(sweep)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sweep": sweep,
        "params": load_sweep_params(sweep),
        "combos": [
            {
                "id": combo.combo_id,
                "index": combo.index,
                "overrides": list(combo.overrides),
                # Parsed form of the overrides, so a reader does not have to
                # split strings to get at the varied values.
                "values": {
                    override.partition("=")[0]: override.partition("=")[2]
                    for override in combo.overrides
                },
            }
            for combo in combos
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Sweep manifest written to %s", path)
    return path


def load_manifest(path: str | Path) -> dict:
    """Load a sweep manifest written by :func:`write_manifest`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No sweep manifest at {path}")
    return json.loads(path.read_text())


def combo_override_dict(combo: SweepCombo) -> dict[str, str]:
    """Overrides as a plain ``{key: value}`` mapping.

    Convenience for the telemetry stage, which records what each combination
    varied as columns rather than as override strings.
    """
    return dict(override.split("=", 1) for override in combo.overrides)


def sweep_output_dir(sweep: str, combo_id: str | None = None) -> Path:
    """Result directory for a sweep, or for one of its combinations.

    Sweeps live under a separate ``sweeps/`` root rather than inside
    ``results/<experiment>/``. That is deliberate: a Snakemake wildcard cannot
    span a ``/``, so ``results/{experiment}/...`` and
    ``results/{experiment}/{combo}/...`` are *different* patterns and two rules
    would claim the same file. A separate root removes the overlap entirely.
    """
    base = Path("sweeps") / sweep
    return base if combo_id is None else base / combo_id


# Re-exported for the Snakefile, which needs the config directory to resolve
# sweep files.
__all__ = [
    "MANIFEST_NAME",
    "SweepCombo",
    "combo_override_dict",
    "enumerate_combos",
    "load_manifest",
    "load_sweep_params",
    "sweep_config_path",
    "sweep_output_dir",
    "write_manifest",
]
