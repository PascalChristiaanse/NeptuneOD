#!/usr/bin/env python
"""Generate Mermaid sequence diagrams for selected OrbitDet entrypoints.

The diagrams are produced by *tracing live execution* rather than by parsing
the source: each scenario below calls a real entrypoint with a small, fully
offline input, and a :func:`sys.setprofile` hook records every call that stays
inside the :mod:`orbitdet` package.  The recorded call graph is then emitted as
Mermaid ``sequenceDiagram`` markup, which the docs build renders through
``sphinxcontrib-mermaid``.

Why trace instead of parse
--------------------------
Static call graphs cannot see through the registry indirection this codebase
leans on (``register_weight_strategy`` / ``register_outlier_strategy`` dispatch
through a dict of classes) or through ``from_config`` factories that pick a
strategy at runtime.  A trace shows what *actually* runs for a given input,
which is exactly what a sequence diagram is for.

Adding a scenario
-----------------
Append an entry to :data:`SCENARIOS`.  The value is a ``(title, factory)``
pair; the factory performs setup and returns a zero-argument callable that
invokes the entrypoint under study.  The callable must run with no network
access and no filesystem writes.

Choosing which diagrams to build and where they go
--------------------------------------------------
Selection and placement live in ``docs/diagrams.yaml``, not in this script.
Each entry names a scenario and the documented object it belongs to::

    diagrams:
      - name: sim_integrator_settings
        target: orbitdet.simulation.integration.get_integrator_settings

The ``target`` is the fully-qualified dotted path of the object as autodoc
knows it.  The docs build's ``sequence_diagrams`` extension (see
``docs/_ext/sequence_diagrams.py``) injects the generated diagram into that
object's docstring, so it renders on the module's API page right where the
method is documented.  Pass ``--manifest PATH`` to use a different file, or
``--only NAME...`` to override the manifest for a one-off build.

Usage
-----
::

    python scripts/gen_sequence_diagrams.py            # build from the manifest
    python scripts/gen_sequence_diagrams.py --list     # list scenarios
    python scripts/gen_sequence_diagrams.py --only grouping weighting
    python scripts/gen_sequence_diagrams.py --stdout grouping
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from types import FrameType

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROJECT_ROOT = "orbitdet"
OUT_DIR = ROOT / "docs" / "diagrams"
MANIFEST = ROOT / "docs" / "diagrams.yaml"

# A call edge is keyed by ``((caller_module, caller_qualname),
# (callee_module, callee_qualname))`` and maps to the number of times it fired.
Edge = tuple[tuple[str, str], tuple[str, str]]


def _is_project(module: str) -> bool:
    """Return True for modules that belong to the ``orbitdet`` package."""
    return module == PROJECT_ROOT or module.startswith(PROJECT_ROOT + ".")


class _Tracer:
    """A :func:`sys.setprofile` hook that records intra-project call edges.

    Only frames whose module is inside :mod:`orbitdet` are recorded, so the
    diagrams stay focused on project code even though the scenarios call into
    ``tudatpy``, ``numpy`` and ``omegaconf``.  Private helpers (leading
    underscore) are skipped by default to keep the diagrams readable; their
    public callees are attributed to the nearest recorded ancestor.
    """

    def __init__(self, max_depth: int = 12, include_private: bool = False):
        self.max_depth = max_depth
        self.include_private = include_private
        # Stack of (id(frame), module, qualname) for the frames we recorded.
        self._stack: list[tuple[int, str, str]] = []
        self.edges: OrderedDict[Edge, int] = OrderedDict()
        self.roots: list[tuple[str, str]] = []

    def _record(self, frame: FrameType) -> tuple[str, str] | None:
        """Return ``(module, qualname)`` if this frame should be recorded."""
        module = frame.f_globals.get("__name__", "")
        if not _is_project(module):
            return None
        qualname = frame.f_code.co_qualname
        # ``<module>`` frames are import-time execution; the warm-up run in
        # :func:`_trace` already paid that cost, so skip them here.
        if qualname == "<module>":
            return None
        # Comprehensions, lambdas and nested closures add noise without
        # conveying structure; drop them and let their public caller carry the
        # edge.
        if "<" in qualname:
            return None
        if not self.include_private and qualname.rsplit(".", 1)[-1].startswith("_"):
            return None
        return module, qualname

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call":
            record = self._record(frame)
            if record is None or len(self._stack) >= self.max_depth:
                return
            module, qualname = record
            if self._stack:
                caller = (self._stack[-1][1], self._stack[-1][2])
                edge = (caller, (module, qualname))
                self.edges[edge] = self.edges.get(edge, 0) + 1
            else:
                self.roots.append((module, qualname))
            self._stack.append((id(frame), module, qualname))
        elif event == "return":
            # Pop only if this frame is the one we pushed; skipped frames
            # (private or past the depth limit) never entered the stack.
            if self._stack and self._stack[-1][0] == id(frame):
                self._stack.pop()


def _to_mermaid(title: str, roots: list[tuple[str, str]], edges: OrderedDict[Edge, int]) -> str:
    """Render a recorded call graph as Mermaid ``sequenceDiagram`` markup."""
    modules: list[str] = []

    def _add(module: str) -> None:
        if module not in modules:
            modules.append(module)

    for (caller_module, _), (callee_module, _) in edges:
        _add(caller_module)
        _add(callee_module)
    for module, _ in roots:
        _add(module)

    if not modules:
        return "sequenceDiagram\n    Note over M0: (no orbitdet calls recorded)"

    alias = {module: f"M{index}" for index, module in enumerate(modules)}
    lines = ["sequenceDiagram", "    autonumber"]
    for module in modules:
        lines.append(f"    participant {alias[module]} as {module}")
    lines.append(f"    Note over {alias[modules[0]]}: {title}")
    for ((caller_module, _), (callee_module, callee_qualname)), count in edges.items():
        label = callee_qualname if count == 1 else f"{callee_qualname} (x{count})"
        lines.append(f"    {alias[caller_module]}->>{alias[callee_module]}: {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
# Each scenario is a factory: calling it performs any setup (building inputs,
# importing modules) and returns a zero-argument callable that invokes the
# entrypoint under study.  Only that returned callable is traced, so the
# diagram shows the entrypoint's own call tree rather than the scaffolding.
# Keep scenarios deterministic and side-effect free: they run on every build.


def _scenario_grouping() -> Callable[[], None]:
    """Partition a synthetic set into sections and timeframes."""
    import numpy as np
    from omegaconf import OmegaConf

    from orbitdet.observations.grouping import build_group_list

    config = OmegaConf.create(
        {
            "levels": [
                {
                    "type": "section",
                    "sections": [
                        {"name": "pre_encounter", "start": -1e30, "end": 0.0},
                        {"name": "post_encounter", "start": 0.0, "end": 1e30},
                    ],
                },
                {"type": "timeframe", "gap_threshold_hours": 8.0},
            ]
        }
    )
    times = np.array([-100.0, -50.0, 10.0, 20.0])
    return lambda: build_group_list(times, "673", config)


def _scenario_weighting() -> Callable[[], None]:
    """Build the weighting engine from a Hydra config node."""
    from omegaconf import OmegaConf

    from orbitdet.observations.weighting.engine import WeightEngine

    config = OmegaConf.create(
        {
            "strategy": "id_v2",
            "min_sigma_arcsec": 0.01,
            "grouping": {"levels": [{"type": "timeframe", "gap_threshold_hours": 8.0}]},
        }
    )
    return lambda: WeightEngine.from_config(config)


def _scenario_outlier_rejection() -> Callable[[], None]:
    """Build the outlier-rejection engine from a Hydra config node."""
    from omegaconf import OmegaConf

    from orbitdet.observations.outlier_rejection.engine import OutlierEngine

    config = OmegaConf.create(
        {
            "strategies": [
                {"type": "residual_threshold", "threshold_arcsec": 1.5},
                {"type": "epoch_filter", "epochs": [0.0]},
            ]
        }
    )
    return lambda: OutlierEngine.from_config(config)


def _scenario_apriori() -> Callable[[], None]:
    """Build the inverse a priori covariance from a config and parameter set."""
    from omegaconf import OmegaConf
    from tudatpy.dynamics import parameters_setup as param_setup

    from orbitdet.estimation.apriori import get_apriori_covariance_matrix
    from orbitdet.estimation.estimators import get_additional_parameter_identifiers

    state_id = (
        param_setup.EstimatableParameterTypes.initial_body_state_type,
        ("Triton", ""),
    )
    config = OmegaConf.create(
        {
            "estimation": {
                "parameters_to_estimate": {"initial_state": True, "neptune_GM": True},
                "apriori": {"position": 100.0e3, "velocity": 1.0},
            }
        }
    )
    identifiers = get_additional_parameter_identifiers(config)

    class _FakeParameterSet:
        """Minimal stand-in exposing only what the builder reads."""

        def __init__(self, blocks):
            self._blocks = []
            start = 0
            for identifier, size in blocks:
                self._blocks.append((identifier, start, size))
                start += size
            self.parameter_set_size = start

        def get_parameter_identifiers(self):
            return [identifier for identifier, _, _ in self._blocks]

        def indices_for_parameter_type(self, identifier):
            return [(start, size) for ident, start, size in self._blocks if ident == identifier]

    fake = _FakeParameterSet([(state_id, 6), (identifiers["neptune_GM"], 1)])
    return lambda: get_apriori_covariance_matrix(config, fake)


# --- simulation module -----------------------------------------------------
# The simulation entrypoints take a ``RuntimeContext`` and a system of bodies.
# The scenarios below build the same lightweight stand-ins the unit tests use
# (a MagicMock context and, where the entrypoint only reads config, a mocked
# system of bodies) so the traces stay offline and deterministic.


def _simulation_config():
    """A minimal simulation config mirroring the unit-test fixture."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "global_frame_origin": "Neptune",
            "global_frame_orientation": "J2000",
            "bodies_to_create": {
                "Neptune": {"gravity": "central", "rotation_model": {"type": "IAU2015"}},
                "Triton": {
                    "ephemeris": {
                        "type": "interpolated_spice",
                        "interpolator_cadance": 300,
                        "coverage_buffer": 86400,
                    }
                },
            },
            "bodies_to_propagate": {"Triton": {"central_body": "Neptune"}},
            "propagation": {"set_integrated_result": True},
            "integrator": {"type": "RKF78", "fixed_step_size": 10.0},
            "start_epoch": 1234567890.0,
            "end_epoch": 1234567900.0,
            "use_default_body_settings": True,
        }
    )


def _simulation_context():
    """A MagicMock RuntimeContext with the attributes the entrypoints read."""
    from unittest.mock import MagicMock

    from orbitdet.reproducibility.runtime import RuntimeContext

    ctx = MagicMock(spec=RuntimeContext)
    ctx.initial_epoch = 1234567890.0
    ctx.start_epoch = 1234567890.0
    ctx.end_epoch = 1234567900.0
    return ctx


def _scenario_sim_environment() -> Callable[[], None]:
    """Build the environment (system of bodies) from a config."""
    from unittest.mock import MagicMock, patch

    from tudatpy.dynamics import environment as env
    from tudatpy.dynamics import environment_setup as env_setup

    from orbitdet.simulation.environment import get_environment

    config = _simulation_config()
    ctx = _simulation_context()

    def run() -> None:
        body_settings = MagicMock(spec=env_setup.BodyListSettings)
        body_settings.get = MagicMock(return_value=MagicMock())
        bodies = MagicMock(spec=env.SystemOfBodies)
        bodies.list_of_bodies = MagicMock(return_value=[])
        with (
            patch("orbitdet.simulation.environment.spice") as spice,
            patch(
                "orbitdet.simulation.environment.env_setup.get_default_body_settings",
                return_value=body_settings,
            ),
            patch(
                "orbitdet.simulation.environment.env_setup.create_system_of_bodies",
                return_value=bodies,
            ),
        ):
            spice.get_body_gravitational_parameter.return_value = 6.836529e15
            get_environment(config, ctx)

    return run


def _scenario_sim_dynamical_model() -> Callable[[], None]:
    """Build the acceleration model from a config."""
    from unittest.mock import MagicMock, patch

    from tudatpy.dynamics import environment as env

    from orbitdet.simulation.dynamics import get_dynamical_model

    config = _simulation_config()
    ctx = _simulation_context()
    bodies = MagicMock(spec=env.SystemOfBodies)

    def run() -> None:
        with patch(
            "orbitdet.simulation.dynamics.prop_setup.create_acceleration_models",
            return_value={"Triton": {"Neptune": []}},
        ):
            get_dynamical_model(config, ctx, bodies)

    return run


def _scenario_sim_integrator_settings() -> Callable[[], None]:
    """Build integrator settings from a config."""
    from orbitdet.simulation.integration import get_integrator_settings

    config = _simulation_config()
    ctx = _simulation_context()
    return lambda: get_integrator_settings(config, ctx)


def _scenario_sim_propagator_settings() -> Callable[[], None]:
    """Build propagator settings from a config."""
    from unittest.mock import MagicMock, patch

    from orbitdet.simulation.propagation import get_propagator_settings

    config = _simulation_config()
    ctx = _simulation_context()
    integrator_settings = MagicMock()
    # ``translational`` returns a settings object whose ``processing_settings``
    # is read-only on the real TudatPy object; mock it as the unit tests do.
    settings = MagicMock()
    settings.processing_settings = MagicMock()

    def run() -> None:
        with (
            patch(
                "orbitdet.simulation.propagation.spice.get_body_cartesian_state_at_epoch",
                return_value=[100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
            ),
            patch(
                "orbitdet.simulation.propagation.prop_setup.propagator.translational",
                return_value=settings,
            ),
        ):
            get_propagator_settings(
                config,
                ctx,
                MagicMock(),
                integrator_settings,
                [],
                initial_state=[100000.0, 0.0, 0.0, 0.0, 5000.0, 0.0],
            )

    return run


SCENARIOS: dict[str, tuple[str, Callable[[], Callable[[], None]], bool]] = {
    "grouping": ("Grouping: build_group_list", _scenario_grouping, False),
    "weighting": ("Weighting: WeightEngine.from_config", _scenario_weighting, False),
    "outlier_rejection": (
        "Outlier rejection: OutlierEngine.from_config",
        _scenario_outlier_rejection,
        False,
    ),
    "apriori": ("A priori covariance: get_apriori_covariance_matrix", _scenario_apriori, False),
    # ``get_environment`` and ``get_dynamical_model`` are thin orchestrators
    # whose substance is in their private helpers, so their diagrams include
    # private frames by default; the others stay public-only for readability.
    "sim_environment": ("Simulation: get_environment", _scenario_sim_environment, True),
    "sim_dynamical_model": (
        "Simulation: get_dynamical_model",
        _scenario_sim_dynamical_model,
        True,
    ),
    "sim_integrator_settings": (
        "Simulation: get_integrator_settings",
        _scenario_sim_integrator_settings,
        False,
    ),
    "sim_propagator_settings": (
        "Simulation: get_propagator_settings",
        _scenario_sim_propagator_settings,
        False,
    ),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _trace(
    scenario: Callable[[], Callable[[], None]],
    max_depth: int,
    include_private: bool,
) -> _Tracer:
    """Run a scenario's entrypoint under the tracer and return the graph.

    The scenario factory is called first to perform setup (imports, building
    inputs) outside the trace; only the returned entrypoint callable is traced.
    """
    entrypoint = scenario()  # setup, untraced
    tracer = _Tracer(max_depth=max_depth, include_private=include_private)
    sys.setprofile(tracer)
    try:
        entrypoint()
    finally:
        sys.setprofile(None)
    return tracer


def _load_manifest(path: Path) -> list[tuple[str, str, str | None]]:
    """Read the diagram manifest.

    Returns a list of ``(name, target, title_override)`` in listed order, where
    ``target`` is the dotted path of the documented object the diagram is
    injected into.  Raises :class:`ValueError` with a readable message on any
    structural problem so a typo fails the docs build loudly rather than
    silently dropping a diagram.
    """
    if not path.exists():
        raise ValueError(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("diagrams")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty 'diagrams' list")

    manifest: list[tuple[str, str, str | None]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "name" not in entry or "target" not in entry:
            raise ValueError(f"{path}: entry {index} must be a mapping with 'name' and 'target'")
        name = str(entry["name"])
        if name not in SCENARIOS:
            available = ", ".join(SCENARIOS)
            raise ValueError(f"{path}: unknown scenario {name!r} (available: {available})")
        target = str(entry["target"])
        title = entry.get("title")
        manifest.append((name, target, str(title) if title is not None else None))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="NAME", help="generate only these scenarios")
    parser.add_argument("--list", action="store_true", help="list scenario names and exit")
    parser.add_argument("--stdout", metavar="NAME", help="print one diagram to stdout")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="diagram manifest (default: docs/diagrams.yaml)",
    )
    parser.add_argument("--max-depth", type=int, default=12, help="max recorded call depth")
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="also record private (underscore-prefixed) helpers",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, (title, _, _) in SCENARIOS.items():
            print(f"{name:24s} {title}")
        return 0

    if args.stdout:
        if args.stdout not in SCENARIOS:
            parser.error(f"unknown scenario {args.stdout!r}")
        title, scenario, private = SCENARIOS[args.stdout]
        tracer = _trace(scenario, args.max_depth, args.include_private or private)
        print(_to_mermaid(title, tracer.roots, tracer.edges))
        return 0

    try:
        manifest = _load_manifest(args.manifest)
    except ValueError as error:
        parser.error(str(error))

    # ``--only`` overrides the manifest's selection but keeps its target/title
    # assignments for the scenarios it names.
    if args.only:
        unknown = [name for name in args.only if name not in SCENARIOS]
        if unknown:
            parser.error(f"unknown scenario(s): {', '.join(unknown)}")
        by_name = {name: (target, title) for name, target, title in manifest}
        manifest = [(name, *by_name.get(name, ("", None))) for name in args.only]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every .rst here was generated by an older version of this script; the
    # diagrams are now injected into docstrings by the Sphinx extension, so
    # clear any stale page files left behind.
    for stale in OUT_DIR.glob("*.rst"):
        stale.unlink()

    for name, _target, _override in manifest:
        title, scenario, private = SCENARIOS[name]
        tracer = _trace(scenario, args.max_depth, args.include_private or private)
        mermaid = _to_mermaid(title, tracer.roots, tracer.edges)
        (OUT_DIR / f"{name}.mmd").write_text(mermaid + "\n", encoding="utf-8")
        print(f"wrote {OUT_DIR.relative_to(ROOT)}/{name}.mmd ({len(tracer.edges)} edges)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
