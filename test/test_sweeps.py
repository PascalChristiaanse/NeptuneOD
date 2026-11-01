"""Tests for sweep enumeration, telemetry aggregation, and comparison figures.

The sweep support has one property that must not regress: the enumerator's
ordering has to match Hydra's own, because the combination index is used as the
directory name. If the order changed, existing result directories would silently
point at different parameter combinations.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))

import depgraph  # noqa: E402,F401  (keeps the workflow import root consistent)

from orbitdet.pipeline import sweep_telemetry, sweeps  # noqa: E402

# ---------------------------------------------------------------------------
# Parsing hydra.sweeper.params
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_dir(tmp_path, monkeypatch):
    """Point the enumerator at a temporary conf/ tree."""
    conf = tmp_path / "conf" / "sweep"
    conf.mkdir(parents=True)
    monkeypatch.setattr(sweeps, "CONF_DIR", tmp_path / "conf")
    return conf


def _write_sweep(sweep_dir: Path, name: str, params: dict) -> None:
    import yaml

    (sweep_dir / f"{name}.yaml").write_text(
        yaml.safe_dump({"hydra": {"sweeper": {"params": params}}})
    )


def test_compact_comma_syntax_is_split(sweep_dir):
    """Hydra's ``true,false`` is a string, not a YAML list.

    Reading it without splitting collapses the sweep to a single combination,
    which is the bug this guards: the first version of the enumerator treated
    ``"true,false"`` as one value and produced 1 combination instead of 32.
    """
    _write_sweep(sweep_dir, "s", {"a.b": "true,false"})

    assert sweeps.load_sweep_params("s") == {"a.b": ["true", "false"]}


def test_yaml_sequence_syntax_is_accepted(sweep_dir):
    _write_sweep(sweep_dir, "s", {"a": ["x", "y", "z"]})

    assert sweeps.load_sweep_params("s") == {"a": ["x", "y", "z"]}


def test_scalar_value_yields_one_option(sweep_dir):
    _write_sweep(sweep_dir, "s", {"a": "only"})

    assert sweeps.load_sweep_params("s") == {"a": ["only"]}


def test_missing_sweep_file_raises(sweep_dir):
    with pytest.raises(FileNotFoundError, match="No sweep config"):
        sweeps.load_sweep_params("nope")


def test_sweep_without_params_raises(sweep_dir):
    """A sweep that varies nothing would silently produce one job."""
    _write_sweep(sweep_dir, "empty", {})

    with pytest.raises(ValueError, match="declares no hydra.sweeper.params"):
        sweeps.load_sweep_params("empty")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def test_enumeration_is_the_cartesian_product(sweep_dir):
    _write_sweep(sweep_dir, "s", {"a": "1,2", "b": "x,y,z"})

    combos = sweeps.enumerate_combos("s")

    assert len(combos) == 6


def test_enumeration_order_matches_hydra(sweep_dir):
    """Last declared key varies fastest, as Hydra's own sweeper does.

    Verified against ``BasicSweeper.split_arguments`` in Hydra 1.3: it yields
    (1,x), (1,y), (2,x), (2,y) for two 2-valued keys.
    """
    _write_sweep(sweep_dir, "s", {"a": "1,2", "b": "x,y"})

    combos = sweeps.enumerate_combos("s")

    assert [c.overrides for c in combos] == [
        ("a=1", "b=x"),
        ("a=1", "b=y"),
        ("a=2", "b=x"),
        ("a=2", "b=y"),
    ]


def test_combo_ids_are_zero_padded_and_sequential(sweep_dir):
    _write_sweep(sweep_dir, "s", {"a": "1,2,3"})

    combos = sweeps.enumerate_combos("s")

    assert [c.combo_id for c in combos] == ["00", "01", "02"]


def test_combo_id_widens_past_two_digits(sweep_dir):
    """A sweep with >100 combinations must not collide on directory names."""
    _write_sweep(sweep_dir, "s", {"a": ",".join(str(i) for i in range(120))})

    combos = sweeps.enumerate_combos("s")

    assert combos[99].combo_id == "99"
    assert combos[100].combo_id == "100"
    assert len({c.combo_id for c in combos}) == 120


def test_real_fit_to_ephemeris_sweep_has_32_combinations():
    """The committed sweep is 5 booleans, i.e. 32 combinations."""
    combos = sweeps.enumerate_combos("fit_to_ephemeris")

    assert len(combos) == 32
    assert combos[0].combo_id == "00"
    assert combos[-1].combo_id == "31"
    # First and last are the extremes: all-true and all-false.
    assert all(o.endswith("=true") for o in combos[0].overrides)
    assert all(o.endswith("=false") for o in combos[-1].overrides)


def test_real_sweeps_all_enumerate():
    """Every committed sweep config must enumerate without error."""
    sweep_dir = Path("conf/sweep")
    for path in sorted(sweep_dir.glob("*.yaml")):
        combos = sweeps.enumerate_combos(path.stem)
        assert combos, f"{path.name} enumerated zero combinations"


def test_labels_use_short_key_names(sweep_dir):
    _write_sweep(sweep_dir, "s", {"estimation.parameters_to_estimate.neptune_GM": "true,false"})

    combos = sweeps.enumerate_combos("s")

    assert combos[0].label == "neptune_GM=true"
    assert combos[1].label == "neptune_GM=false"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trips(sweep_dir, tmp_path):
    _write_sweep(sweep_dir, "s", {"a": "1,2", "b": "x,y"})
    path = tmp_path / "manifest.json"

    sweeps.write_manifest("s", path)
    loaded = sweeps.load_manifest(path)

    assert loaded["sweep"] == "s"
    assert len(loaded["combos"]) == 4
    assert loaded["combos"][0]["id"] == "00"
    assert loaded["combos"][0]["values"] == {"a": "1", "b": "x"}


def test_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No sweep manifest"):
        sweeps.load_manifest(tmp_path / "absent.json")


def test_combo_override_dict_parses_kv(sweep_dir):
    _write_sweep(sweep_dir, "s", {"a": "1,2"})

    combo = sweeps.enumerate_combos("s")[1]

    assert sweeps.combo_override_dict(combo) == {"a": "2"}


# ---------------------------------------------------------------------------
# Shared-stage guard
# ---------------------------------------------------------------------------


def test_guard_refuses_overrides_that_reach_a_shared_stage():
    """Combinations share the base experiment's preprocessing, so an override
    that could change it must be refused rather than silently ignored."""
    from orbitdet.pipeline.sweep_resolve import _refuse_shared_stage_overrides

    for bad in ("data=classical", "datasets.x=y", "integrator.type=RK4", "weighting=x"):
        with pytest.raises(ValueError, match="shared stage"):
            _refuse_shared_stage_overrides([bad])


def test_guard_allows_estimation_only_overrides():
    from orbitdet.pipeline.sweep_resolve import _refuse_shared_stage_overrides

    # Must not raise.
    _refuse_shared_stage_overrides(
        [
            "estimation.parameters_to_estimate.neptune_GM=false",
            "estimation.max_iterations=3",
        ]
    )


def test_every_committed_sweep_passes_the_shared_stage_guard():
    """A committed sweep whose overrides reach a shared stage would fail at run
    time; catching it here keeps the configs honest.

    ``params_x_data`` and friends *do* vary ``data``/``parameters``, so they are
    excluded: they are not yet wired into this workflow, and wiring them would
    require per-combination preprocessing.
    """
    from orbitdet.pipeline.sweep_resolve import _refuse_shared_stage_overrides

    for combo in sweeps.enumerate_combos("fit_to_ephemeris"):
        _refuse_shared_stage_overrides(list(combo.overrides))


def test_sweep_covariance_uses_a_per_combination_trajectory():
    """The covariance stage must not borrow the base experiment's trajectory.

    ``run_covariance`` defines its RSW frame from a propagated trajectory, and
    the frame has to belong to the *same* fit as the covariance. An earlier
    version of the sweep rules pointed ``sweep_covariance`` at
    ``results/<experiment>/propagation/postfit_propagation_result.tudat``, so
    each combination was rotated into a frame from a different estimate -- a
    16-parameter state while the combination had estimated 6.

    This guards the fix by asserting the rule reads the combination's own
    propagation output.
    """
    rules_text = (Path(__file__).resolve().parents[1] / "workflow/rules/sweeps.smk").read_text()

    start = rules_text.index("rule sweep_covariance:")
    end = rules_text.index("rule ", start + len("rule sweep_covariance:"))
    block = rules_text[start:end]

    assert "sweeps/{sweep}/{combo}/propagation/postfit_propagation_result.tudat" in block, (
        "sweep_covariance must consume the combination's own post-fit trajectory"
    )
    assert "sweep_postfit_propagation" in rules_text, (
        "the per-combination propagation rule must exist"
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_telemetry_aggregates_estimation_and_covariance(tmp_path, monkeypatch):
    monkeypatch.setattr(sweeps, "CONF_DIR", Path("conf"))
    base = tmp_path / "sweeps" / "fit_to_ephemeris"
    _write_json(
        base / "manifest.json",
        {
            "sweep": "fit_to_ephemeris",
            "combos": [
                {
                    "id": "00",
                    "index": 0,
                    "overrides": ["estimation.parameters_to_estimate.neptune_GM=true"],
                    "values": {"estimation.parameters_to_estimate.neptune_GM": "true"},
                },
                {
                    "id": "01",
                    "index": 1,
                    "overrides": ["estimation.parameters_to_estimate.neptune_GM=false"],
                    "values": {"estimation.parameters_to_estimate.neptune_GM": "false"},
                },
            ],
        },
    )
    for combo, rms in (("00", 1.0), ("01", 2.0)):
        _write_json(
            base / combo / "estimation" / "estimation_summary.json",
            {"residuals_rms": rms, "num_parameters": 7, "covariance_is_finite": rms < 1.5},
        )
        _write_json(
            base / combo / "covariance" / "covariance_summary.json",
            {"formal_errors_are_finite": rms < 1.5, "max_position_sigma_m": rms * 10},
        )

    monkeypatch.setattr(sweep_telemetry, "sweep_output_dir", lambda sweep, combo_id=None: base)

    out = sweep_telemetry.collect_telemetry("fit_to_ephemeris", tmp_path / "telemetry.csv")
    rows = list(csv.DictReader(out.open()))

    assert len(rows) == 2
    assert rows[0]["combo"] == "00"
    assert float(rows[1]["residuals_rms"]) == 2.0
    # The varied parameter becomes its own column, keyed by its *full* dotted
    # path. Short names would collide: a sweep over both
    # `...iau_rotation_model_pole` and `...iau_rotation_model_pole_rate` shares
    # the prefix, so the last component alone is not unique.
    assert rows[0]["estimation.parameters_to_estimate.neptune_GM"] == "true"


def test_telemetry_columns_stay_unique_for_similar_key_names(tmp_path, monkeypatch):
    """Two varied keys whose short names share a prefix must not collide."""
    base = tmp_path / "sweeps" / "fit_to_ephemeris"
    manifest_combos = []
    for index, key in enumerate(
        {
            "estimation.parameters_to_estimate.iau_rotation_model_pole",
            "estimation.parameters_to_estimate.iau_rotation_model_pole_rate",
        }
    ):
        combo_id = f"{index:02d}"
        manifest_combos.append(
            {
                "id": combo_id,
                "index": index,
                "overrides": [f"{key}=true"],
                "values": {key: "true"},
            }
        )
        _write_json(
            base / combo_id / "estimation" / "estimation_summary.json",
            {"residuals_rms": 1.0},
        )

    _write_json(base / "manifest.json", {"sweep": "fit_to_ephemeris", "combos": manifest_combos})
    monkeypatch.setattr(sweep_telemetry, "sweep_output_dir", lambda sweep, combo_id=None: base)

    out = sweep_telemetry.collect_telemetry("fit_to_ephemeris", tmp_path / "t.csv")
    header = out.read_text().splitlines()[0].split(",")

    assert "estimation.parameters_to_estimate.iau_rotation_model_pole" in header
    assert "estimation.parameters_to_estimate.iau_rotation_model_pole_rate" in header
    assert len(header) == len(set(header))


def test_telemetry_skips_combination_without_summary(tmp_path, monkeypatch):
    """A partially-complete sweep must still produce a usable table."""
    base = tmp_path / "sweeps" / "s"
    _write_json(
        base / "manifest.json",
        {
            "sweep": "s",
            "combos": [
                {"id": "00", "index": 0, "overrides": [], "values": {}},
                {"id": "01", "index": 1, "overrides": [], "values": {}},
            ],
        },
    )
    _write_json(base / "00" / "estimation" / "estimation_summary.json", {"residuals_rms": 1.0})

    monkeypatch.setattr(sweep_telemetry, "sweep_output_dir", lambda sweep, combo_id=None: base)

    out = sweep_telemetry.collect_telemetry("s", tmp_path / "t.csv")
    rows = list(csv.DictReader(out.open()))

    assert [r["combo"] for r in rows] == ["00"]


def test_telemetry_writes_header_when_no_summaries(tmp_path, monkeypatch):
    base = tmp_path / "sweeps" / "s"
    _write_json(base / "manifest.json", {"sweep": "s", "combos": []})

    monkeypatch.setattr(sweep_telemetry, "sweep_output_dir", lambda sweep, combo_id=None: base)

    out = sweep_telemetry.collect_telemetry("s", tmp_path / "t.csv")

    header = out.read_text().strip()
    assert header.startswith("combo,overrides,")


def test_telemetry_columns_are_disjoint_and_ordered():
    """The three column groups must not overlap, or a value is written twice."""
    groups = [
        sweep_telemetry.MANIFEST_COLUMNS,
        sweep_telemetry.ESTIMATION_COLUMNS,
        sweep_telemetry.COVARIANCE_COLUMNS,
    ]
    flat = [c for g in groups for c in g]

    assert len(flat) == len(set(flat)), "column groups overlap"
    assert tuple(flat) == sweep_telemetry.BASE_COLUMNS


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------


def test_metrics_registry_covers_the_charted_columns():
    from orbitdet.pipeline.sweep_figures import METRICS

    # Every charted metric must be a telemetry column, or the figure reads "".
    for metric in METRICS:
        assert metric in sweep_telemetry.BASE_COLUMNS, metric


def test_unknown_metric_raises(tmp_path):
    from omegaconf import OmegaConf

    from orbitdet.pipeline.sweep_figures import SweepComparison

    with pytest.raises(ValueError, match="Unknown metric"):
        SweepComparison(OmegaConf.create({}), tmp_path / "t.csv", "s", "not_a_metric")


def test_comparison_figure_renders(tmp_path):
    """Render a real chart, since the plotting path is where sorting and the
    colour ramp can fail."""
    from omegaconf import OmegaConf

    from orbitdet.pipeline.sweep_figures import SweepComparison

    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "combo,overrides,residuals_rms,num_parameters\n"
        "00,a=1,3.0,7\n"
        "01,a=2,1.0,9\n"
        "02,a=3,2.0,8\n"
    )

    plotter = SweepComparison(OmegaConf.create({}), telemetry, "s", "residuals_rms")
    plotter.output_dir = tmp_path
    plotter.plot()

    assert (tmp_path / "sweep_comparison_residuals_rms.pdf").exists()


def test_comparison_figure_tolerates_non_finite_values(tmp_path):
    """A degenerate metric must still render, not crash the whole sweep."""
    from omegaconf import OmegaConf

    from orbitdet.pipeline.sweep_figures import SweepComparison

    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "combo,overrides,residuals_rms,num_parameters\n"
        "00,a=1,nan,7\n"
        "01,a=2,1.0,9\n"
    )

    plotter = SweepComparison(OmegaConf.create({}), telemetry, "s", "residuals_rms")
    plotter.output_dir = tmp_path
    plotter.plot()

    assert (tmp_path / "sweep_comparison_residuals_rms.pdf").exists()


def test_comparison_figure_raises_on_empty_telemetry(tmp_path):
    from omegaconf import OmegaConf

    from orbitdet.pipeline.sweep_figures import SweepComparison

    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text("combo,overrides,residuals_rms\n")

    plotter = SweepComparison(OmegaConf.create({}), telemetry, "s", "residuals_rms")
    plotter.output_dir = tmp_path

    with pytest.raises(RuntimeError, match="no rows"):
        plotter.plot()
