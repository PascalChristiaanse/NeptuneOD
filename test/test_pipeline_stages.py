"""Tests for the file-based pipeline stages.

These cover the parts of the pipeline that can be tested without a full tudatpy
environment: config slicing, artifact path handling, the residual CSV schema,
and the report assembly. The stages that need SPICE kernels and a real
observation collection are exercised end-to-end by the Snakemake workflow
instead, which is the only place their inputs exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from orbitdet.pipeline import artifacts, report, resolve

# ---------------------------------------------------------------------------
# Config slicing
# ---------------------------------------------------------------------------


def test_stage_keys_covers_every_stage_the_workflow_uses():
    """The Snakefile's resolve rules and STAGE_KEYS must agree.

    A stage present in one but not the other produces a rule whose config slice
    is never written, which fails at run time rather than at parse time.
    """
    expected = {
        "kernels",
        "observations",
        "outlier_rejection",
        "weighting",
        "dynamics",
        "prefit",
        "estimation",
        "covariance",
        "figures",
    }

    assert set(resolve.STAGE_KEYS) == expected


def test_slice_keeps_only_requested_keys():
    cfg = OmegaConf.create({"a": 1, "b": {"c": 2}, "d": 3})

    sliced = resolve._slice(cfg, ["a", "b"])

    assert set(sliced.keys()) == {"a", "b"}
    assert sliced.b.c == 2


def test_slice_ignores_absent_keys():
    """A stage must not fail because an optional config block is missing."""
    cfg = OmegaConf.create({"a": 1})

    sliced = resolve._slice(cfg, ["a", "missing"])

    assert set(sliced.keys()) == {"a"}


def test_figures_slice_excludes_compute_keys():
    """The figures slice must not carry compute config, or styling edits
    would invalidate the observation build."""
    keys = resolve.STAGE_KEYS["figures"]

    assert "datasets" not in keys
    assert "estimation" not in keys
    assert "weighting" not in keys
    assert "figures" in keys


def test_prefit_slice_includes_datasets():
    """``prefit`` needs ``datasets`` even though it loads the collection from
    disk: the environment build resolves a tabulated ephemeris through
    ``cfg.datasets.<source_dataset>``.

    This is a regression test. Omitting ``datasets`` here fails at run time with
    "Voyager dataset configuration ... is required to build a tabulated
    ephemeris", which is exactly what happened for ``classic_triton_state``.
    """
    assert "datasets" in resolve.STAGE_KEYS["prefit"]


def test_every_environment_building_stage_includes_datasets():
    """Any stage that builds an environment needs the dataset config.

    ``figures`` is the only stage that never builds one, so it is the only
    stage exempt from this.
    """
    for stage in ("observations", "prefit", "estimation", "covariance", "weighting"):
        assert "datasets" in resolve.STAGE_KEYS[stage], f"{stage} is missing datasets"

    assert "datasets" not in resolve.STAGE_KEYS["figures"]


def test_estimation_slice_includes_dynamics_and_estimation():
    keys = resolve.STAGE_KEYS["estimation"]

    assert "estimation" in keys
    assert "bodies_to_propagate" in keys
    assert "integrator" in keys


def test_fit_to_ephemeris_has_no_weighting_or_outliers():
    """The simulated experiment disables both subsystems.

    The DAG is unconditional, so both stages still run and forward the
    collection unchanged; the weight-table stage must therefore tolerate an
    empty weights CSV rather than fail.
    """
    from orbitdet.pipeline.config import load_config

    cfg = load_config("fit_to_ephemeris")

    assert "weighting" not in cfg
    assert "outlier_rejection" not in cfg


def test_every_stage_slice_includes_context_keys():
    """Every stage builds a context, which reads the epochs and seed."""
    for stage, keys in resolve.STAGE_KEYS.items():
        if stage == "figures":
            # Figures never build a context: they are pure functions of a file.
            continue
        for required in ("seed", "start_date", "end_date", "initial_epoch"):
            assert required in keys, f"{stage} is missing {required}"


def test_resolve_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match="Unknown stage"):
        resolve.run_resolve_config("fit_to_ephemeris", tmp_path, only="nonsense")


# ---------------------------------------------------------------------------
# Artifact path handling
# ---------------------------------------------------------------------------


def test_tudat_stem_strips_the_suffix():
    """``save_to_binary`` appends ``.tudat`` unconditionally, so passing a path
    that already ends in it would write ``x.tudat.tudat``."""
    assert artifacts.tudat_stem("results/obs.tudat") == "results/obs"
    assert artifacts.tudat_stem("results/obs") == "results/obs"


def test_tudat_stem_preserves_other_suffixes():
    assert artifacts.tudat_stem("results/obs.bin") == "results/obs.bin"


def test_load_metadata_returns_empty_when_absent(tmp_path):
    assert artifacts.load_metadata(tmp_path / "missing.json") == {}


def test_save_and_load_metadata_round_trip(tmp_path):
    payload = {"set_a": [{"dataset_id": "nm0008", "index": 0}]}
    path = tmp_path / "dataset_metadata.json"

    artifacts.save_metadata(payload, path)

    assert artifacts.load_metadata(path) == payload


def test_load_collection_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No serialized observation collection"):
        artifacts.load_collection(tmp_path / "absent.tudat")


def test_load_propagation_results_tries_concrete_classes_first():
    """``load_from_binary`` is bound per concrete class and rejects a type
    mismatch, so the helper must offer the concrete classes.

    This is a regression test. A dynamics propagation writes a
    ``SingleArcSimulationResults``, and loading through the abstract
    ``SimulationResults`` fails with "The file contains
    'SingleArcSimulationResults<...>', but the selected load method expects
    'SimulationResults<...>'".
    """
    from tudatpy.dynamics.propagation import SingleArcSimulationResults

    from orbitdet.pipeline.artifacts import load_propagation_results  # noqa: F401

    # The concrete class must be reachable from the module the helper uses.
    assert SingleArcSimulationResults.__name__ == "SingleArcSimulationResults"


def test_concrete_load_class_rejects_abstract_loader():
    """Documents the tudatpy behaviour the helper works around.

    If tudatpy ever gains a polymorphic loader this test will fail, which is the
    right signal: the workaround could then be simplified.
    """
    from tudatpy.dynamics.propagation import SimulationResults, SingleArcSimulationResults

    # The abstract class must not be a superclass that accepts the concrete
    # file's root type.
    assert SimulationResults is not SingleArcSimulationResults
    assert issubclass(SingleArcSimulationResults, SimulationResults)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Report captions
# ---------------------------------------------------------------------------


def _rule_files() -> list[Path]:
    return sorted((Path(__file__).resolve().parents[1] / "workflow" / "rules").glob("*.smk"))


def test_every_caption_file_exists():
    """Every caption named by ``_caption(...)`` must exist under ``report/``.

    A missing caption is a hard failure, but only *during* ``--report``: the
    pipeline runs to completion and then the report dies with a bare
    ``FileNotFoundError`` naming a path under ``workflow/rules/``. That made
    the failure look like a Snakemake path bug rather than a missing file.

    This test is a regression guard for exactly that: the sweep comparison
    rules referenced ``report/sweeps/sweep_comparison.rst`` for a while before
    the file existed (and ``report/sweeps/`` was an empty directory), so no
    sweep could ever be reported.
    """
    pattern = re.compile(r'_caption\(\s*"([^"]+)"\s*(?:,\s*"([^"]+)")?\s*\)')
    repo = Path(__file__).resolve().parents[1]

    referenced: set[str] = set()
    for rule_file in _rule_files():
        for line in rule_file.read_text().splitlines():
            # Only real call sites, so a docstring or comment showing the
            # helper's signature does not register as a referenced caption.
            if "caption=_caption(" not in line:
                continue
            match = pattern.search(line)
            assert match, f"could not parse caption on {rule_file.name}: {line.strip()}"
            referenced.add("/".join(part for part in match.groups() if part))

    assert referenced, "no captions found; did the _caption helper get renamed?"

    missing = sorted(name for name in referenced if not (repo / "report" / name).exists())
    assert not missing, f"caption files referenced but missing: {missing}"


def test_workflow_report_description_exists():
    """The global ``report:`` directive must name an existing file."""
    snakefile = Path(__file__).resolve().parents[1] / "workflow" / "Snakefile"
    text = snakefile.read_text()

    match = re.search(r'^report:\s*_caption\("([^"]+)"\)', text, re.M)
    assert match, "the Snakefile should declare a workflow-level report description"

    assert (Path(__file__).resolve().parents[1] / "report" / match.group(1)).exists()


def test_caption_paths_are_built_from_the_helper():
    """Captions must go through ``_caption``, never a bare relative path.

    Snakemake resolves a caption relative to the *including* file, so a literal
    ``"report/figures/x.rst"`` inside ``workflow/rules/`` resolves to
    ``workflow/rules/report/figures/x.rst``. An absolute path avoids the
    rule; the helper is what produces one.
    """
    for rule_file in _rule_files():
        for lineno, line in enumerate(rule_file.read_text().splitlines(), 1):
            if "caption=" not in line or line.lstrip().startswith("#"):
                continue
            assert "caption=_caption(" in line, (
                f"{rule_file.name}:{lineno} passes a raw caption path; "
                "use _caption(...) so it resolves to the repository's report/ tree"
            )

# ---------------------------------------------------------------------------
# Weight tables
# ---------------------------------------------------------------------------


def test_empty_weight_tables_are_parseable(tmp_path):
    """An empty summary must have headers, not be a 0-byte file.

    Weighting can legitimately be off (``fit_to_ephemeris`` declares no
    ``weighting:`` block), and the rule is unconditional, so the stage writes an
    empty table. Writing ``""`` looked like "headers only" but is actually an
    unparseable file: ``pd.read_csv`` raises ``EmptyDataError: No columns to
    parse from file``, so anything downstream that reads the table back —
    including a reader opening the CSV straight out of the HTML report — hit an
    error rather than an empty result.
    """
    from orbitdet.pipeline import tables
    from orbitdet.visualization.weight_summary_table import (
        GROUP_COLUMNS,
        SOURCE_COLUMNS,
    )

    weights = tmp_path / "observation_weights.csv"
    weights.write_text("source_name,ra_residual_arcsec\n")  # headers only

    written = tables.run_weight_tables(OmegaConf.create({}), weights, tmp_path)
    assert written

    for name, expected in (("source", SOURCE_COLUMNS), ("group", GROUP_COLUMNS)):
        df = pd.read_csv(tmp_path / f"weight_summary_{name}.csv")
        assert df.empty
        assert list(df.columns) == expected


def test_populated_and_empty_tables_share_a_schema(tmp_path):
    """The empty and populated paths must produce the same columns.

    They are built by different code (a real ``groupby`` versus a bare
    constructor), so nothing but the shared ``*_COLUMNS`` constants keeps them
    from drifting apart.
    """
    from orbitdet.visualization.weight_summary_table import (
        GROUP_COLUMNS,
        SOURCE_COLUMNS,
        WeightSummaryTable,
    )

    weights = pd.DataFrame(
        {
            "source_name": ["a", "a", "b"],
            "dataset_id": ["ds1", "ds1", "ds2"],
            "observatory_code": ["250", "250", "C51"],
            "group_id": [0, 0, 1],
            "ra_residual_arcsec": [0.1, -0.2, 0.3],
            "dec_residual_arcsec": [0.05, -0.1, 0.2],
            "ra_type": ["RA", "RA", "RA"],
            "dec_type": ["DEC", "DEC", "DEC"],
        }
    )

    builder = WeightSummaryTable(OmegaConf.create({}), weights)
    assert list(builder.build_source().columns) == SOURCE_COLUMNS
    assert list(builder.build_group().columns) == GROUP_COLUMNS

    empty = WeightSummaryTable(OmegaConf.create({}), weights.iloc[0:0])
    assert list(empty.build_source().columns) == SOURCE_COLUMNS
    assert list(empty.build_group().columns) == GROUP_COLUMNS


def test_report_includes_estimation_metrics(tmp_path):
    results = tmp_path / "results"
    _write(
        results / "estimation" / "estimation_summary.json",
        {
            "num_observations": 1234,
            "num_iterations": 5,
            "best_iteration": 4,
            "residuals_rms": 1.5e-6,
            "postfit_rms_best_iteration": 1.2e-6,
            "residuals_max_abs": 9.9e-6,
            "estimation_duration_s": 42.0,
        },
    )

    out = report.run_report("demo", results, tmp_path / "report.md")
    text = out.read_text()

    assert "# demo" in text
    assert "1234" in text
    assert "1.5e-06" in text


def test_report_notes_disabled_outlier_rejection(tmp_path):
    results = tmp_path / "results"
    _write(
        results / "preprocessing" / "outlier_rejection_metadata.json",
        {"enabled": False, "n_accepted": 10, "n_rejected": 0, "n_total_observations": 10},
    )

    text = report.run_report("demo", results, tmp_path / "report.md").read_text()

    assert "Disabled for this experiment." in text


def test_report_includes_outlier_counts_when_enabled(tmp_path):
    results = tmp_path / "results"
    _write(
        results / "preprocessing" / "outlier_rejection_metadata.json",
        {"enabled": True, "n_accepted": 90, "n_rejected": 10, "n_total_observations": 100},
    )

    text = report.run_report("demo", results, tmp_path / "report.md").read_text()

    assert "Accepted: 90" in text
    assert "Rejected: 10" in text


def test_report_includes_covariance_metrics(tmp_path):
    results = tmp_path / "results"
    _write(
        results / "covariance" / "covariance_summary.json",
        {
            "num_epochs": 100,
            "max_position_sigma_m": 3.5,
            "max_velocity_sigma_m_s": 1.1e-5,
            "median_position_sigma_m": 0.4,
            "formal_errors_are_finite": True,
        },
    )

    text = report.run_report("demo", results, tmp_path / "report.md").read_text()

    assert "Peak position sigma: 3.5 m" in text
    assert "Peak velocity sigma: 1.1e-05 m/s" in text


def test_report_states_degenerate_covariance_instead_of_nan(tmp_path):
    """A singular normal matrix inverts to NaN rather than raising.

    The report must say the formal errors are unavailable, not print NaN as
    though it were a measurement. This is a regression test: an early version
    emitted "Peak position sigma: nan m".
    """
    results = tmp_path / "results"
    _write(
        results / "covariance" / "covariance_summary.json",
        {
            "num_epochs": 1401,
            "max_position_sigma_m": float("nan"),
            "max_velocity_sigma_m_s": float("nan"),
            "median_position_sigma_m": float("nan"),
            "formal_errors_are_finite": False,
        },
    )

    text = report.run_report("demo", results, tmp_path / "report.md").read_text()

    assert "Not available" in text
    assert "degenerate" in text
    assert "1401" in text
    # The *word* NaN appears in the explanation; a NaN **value** must not.
    assert "sigma: nan" not in text.lower()
    assert "nan m" not in text.lower()


def test_report_flags_degenerate_estimation_covariance(tmp_path):
    results = tmp_path / "results"
    _write(
        results / "estimation" / "estimation_summary.json",
        {
            "num_observations": 10,
            "num_iterations": 2,
            "best_iteration": 1,
            "residuals_rms": 1.0,
            "postfit_rms_best_iteration": 1.0,
            "residuals_max_abs": 2.0,
            "estimation_duration_s": 1.0,
            "covariance_is_finite": False,
        },
    )

    text = report.run_report("demo", results, tmp_path / "report.md").read_text()

    assert "normal matrix is singular" in text


def test_report_handles_no_summaries(tmp_path):
    """A report must still be produced when the pipeline has not run."""
    text = report.run_report("demo", tmp_path / "empty", tmp_path / "report.md").read_text()

    assert "No stage summaries were found" in text


def test_report_survives_corrupt_summary(tmp_path):
    """A malformed JSON must not break the report; it is omitted instead."""
    results = tmp_path / "results"
    path = results / "estimation" / "estimation_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")

    text = report.run_report("demo", results, tmp_path / "report.md").read_text()

    assert "No stage summaries were found" in text


# ---------------------------------------------------------------------------
# Figure dispatch
# ---------------------------------------------------------------------------


def test_collection_figure_registry_covers_every_snakefile_figure():
    """Every figure rule must resolve to a registry entry, or it fails at run
    time rather than at parse time.

    This is a regression test: the registry was originally keyed on plotter
    class names (``normalized_residuals``) while the Snakefile asked for the
    config keys (``prefit_normalized_residuals``), so every prefixed figure
    raised ``Unknown collection figure``.
    """
    from orbitdet.pipeline.figures import COLLECTION_FIGURES  # noqa: F811

    # The names the experiment files declare, which double as config keys.
    expected = {
        "prefit_residuals_spice",
        "prefit_residuals_spice_scan",
        "prefit_normalized_residuals",
        "prefit_normalized_residuals_scan",
        "prefit_residuals_prop",
        "prefit_residuals_prop_scan",
        "postfit_residuals",
        "postfit_residuals_scan",
        "postfit_normalized_residuals",
        "postfit_normalized_residuals_scan",
        "residual_qq",
        "residual_scatter",
        "residual_histogram",
    }

    assert expected <= set(COLLECTION_FIGURES)


def test_scan_figures_use_the_gaia_filter():
    """Scan-frame residuals need the archived ``position_angle_scan``."""
    from orbitdet.pipeline import figures as fig_mod
    from orbitdet.pipeline.figures import COLLECTION_FIGURES

    for name in (
        "prefit_residuals_spice_scan",
        "prefit_normalized_residuals_scan",
        "prefit_residuals_prop_scan",
        "postfit_residuals_scan",
        "postfit_normalized_residuals_scan",
    ):
        assert COLLECTION_FIGURES[name].select is fig_mod._gaia_only


def test_normalized_figures_exclude_gaia():
    """A Gaia residual is not meaningful without the scan rotation, so the
    normalized (non-scan) figure must exclude Gaia rather than include it."""
    from orbitdet.pipeline import figures as fig_mod
    from orbitdet.pipeline.figures import COLLECTION_FIGURES

    for name in ("prefit_normalized_residuals", "postfit_normalized_residuals"):
        assert COLLECTION_FIGURES[name].select is fig_mod._non_gaia


def test_distribution_figures_use_every_set():
    from orbitdet.pipeline import figures as fig_mod
    from orbitdet.pipeline.figures import COLLECTION_FIGURES

    for name in ("residual_qq", "residual_scatter", "residual_histogram"):
        assert COLLECTION_FIGURES[name].select is fig_mod._all_sets


def test_estimation_plotter_registry():
    from orbitdet.pipeline.estimation_figures import ESTIMATION_PLOTTERS

    assert set(ESTIMATION_PLOTTERS) == {
        "parameter_correlation_heatmap",
        "parameter_history_per_iteration",
        "residual_rms_per_iteration",
    }


def test_unknown_collection_figure_raises(tmp_path):
    from orbitdet.pipeline.figures import run_collection_figure

    cfg = OmegaConf.create({"figures": {}})

    with pytest.raises(ValueError, match="Unknown collection figure"):
        run_collection_figure(
            cfg,
            observations_path=tmp_path / "x.tudat",
            metadata_path=None,
            output_dir=tmp_path,
            figure_name="not_a_figure",
        )


def test_unknown_estimation_figure_raises(tmp_path):
    from orbitdet.pipeline.estimation_figures import run_estimation_figure

    cfg = OmegaConf.create({"figures": {}})

    with pytest.raises(ValueError, match="Unknown estimation figure"):
        run_estimation_figure(cfg, tmp_path / "x.tudat", tmp_path, "not_a_figure")
