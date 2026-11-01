"""Shared rules for sweep experiments.

A sweep runs one experiment across a grid of Hydra overrides. This file adds the
rules that turn a sweep config into that grid of runs, plus the comparison
outputs on top of them.

Layout
------
Sweeps live under their own ``sweeps/`` root rather than inside
``results/<experiment>/``::

    sweeps/<sweep>/
    ├── manifest.json              combo id -> overrides (the enumeration)
    ├── config/figures.yaml        styling for the comparison charts
    ├── <combo>/                   one directory per combination
    │   ├── config/                resolved slices, overrides applied
    │   ├── estimation/            estimation_output.tudat + summary
    │   └── covariance/            propagated formal errors
    ├── sweep_telemetry.csv        every combination's metrics in one table
    └── figures/sweep_comparison_*.pdf

Why a separate root
-------------------
A Snakemake wildcard cannot span a ``/``, so ``results/{experiment}/x`` and
``results/{experiment}/{combo}/x`` are distinct patterns that two rules would
both claim, producing an ambiguity error. A separate root removes the overlap.
It also keeps a 32-combination sweep from burying the single-run outputs.

Why the combinations share preprocessing
----------------------------------------
The ``fit_to_ephemeris`` sweep varies only
``estimation.parameters_to_estimate.*``, which nothing before the estimation
stage reads. Re-running the observation build and pre-fit propagation for each
of 32 combinations would multiply the most expensive work in the pipeline for no
gain, so combinations depend on the base experiment's weighted collection and
resolve only their own estimation-facing config.

That sharing is only valid while the sweep varies estimation-only settings, so
``run_resolve_sweep_config`` refuses an override that could reach a shared stage
rather than silently reusing a result computed with different settings.

Reading the sweep map
---------------------
``_experiment_for`` maps a sweep to the experiment whose preprocessing it
reuses, from ``workflow/config.yaml``. Adding a sweep therefore never means
editing a rule below.
"""

import importlib.util
import sys

sys.path.insert(0, workflow.basedir)
import depgraph  # noqa: E402

# ``sweeps.py`` is loaded by path rather than as ``orbitdet.pipeline.sweeps``:
# importing the package pulls in Hydra through ``pipeline/__init__``, and the
# Snakefile needs to enumerate combinations while building the DAG, before any
# stage process exists.
SWEEPS_MODULE_PATH = "src/orbitdet/pipeline/sweeps.py"

_spec = importlib.util.spec_from_file_location("orbitdet_sweeps_standalone", SWEEPS_MODULE_PATH)
_sweeps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sweeps)


def _experiment_for(sweep):
    """Base experiment whose preprocessing a sweep reuses."""
    declared = config.get("sweeps", {})
    if sweep not in declared:
        raise KeyError(
            f"Sweep {sweep!r} is not declared under 'sweeps:' in workflow/config.yaml. "
            "Add it there so the rules know which experiment's preprocessing it reuses."
        )
    return declared[sweep]["experiment"]


def sweep_combos(wildcards):
    """Combination ids of a sweep, in manifest order."""
    return [combo.combo_id for combo in _sweeps.enumerate_combos(wildcards.sweep)]


def sweep_overrides(wildcards):
    """Hydra override strings for one combination, read from the enumeration."""
    for combo in _sweeps.enumerate_combos(wildcards.sweep):
        if combo.combo_id == wildcards.combo:
            return " ".join(combo.overrides)
    raise KeyError(f"Sweep {wildcards.sweep!r} has no combination {wildcards.combo!r}")


def sweep_estimation_summaries(wildcards):
    """Every combination's estimation summary, to drive the telemetry rule."""
    return [
        f"sweeps/{wildcards.sweep}/{combo}/estimation/estimation_summary.json"
        for combo in sweep_combos(wildcards)
    ]


def sweep_covariance_summaries(wildcards):
    """Every combination's covariance summary.

    Also an input to the telemetry rule, for two reasons: it makes
    ``formal_errors_are_finite`` available per combination, and it keeps
    ``sweep_covariance`` reachable from the sweep target. Without this input the
    covariance rule would be orphaned -- declared but never run.
    """
    return [
        f"sweeps/{wildcards.sweep}/{combo}/covariance/covariance_summary.json"
        for combo in sweep_combos(wildcards)
    ]


def sweep_shared(path_suffix):
    """Input function for an artefact shared with the base experiment."""

    def _resolve(wildcards):
        return f"results/{_experiment_for(wildcards.sweep)}/{path_suffix}"

    return _resolve


#: Metrics charted by the comparison figures. Kept here so the rule list and the
#: reporting narrative stay in step.
SWEEP_METRICS = [
    "residuals_rms",
    "postfit_rms_best_iteration",
    "estimation_duration_s",
    "num_iterations",
    "num_parameters",
]


rule sweep_manifest:
    """Enumerate the sweep's combinations into a manifest.

    Everything downstream reads this, so the combination list is an artefact
    rather than a convention: editing ``conf/sweep/<sweep>.yaml`` regenerates
    the manifest, which invalidates the whole sweep.
    """
    input:
        sweep_cfg="conf/sweep/{sweep}.yaml",
        code=_src("orbitdet.pipeline.run_sweep_manifest"),
    output:
        "sweeps/{sweep}/manifest.json",
    log:
        "sweeps/{sweep}/manifest.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_sweep_manifest "
        "--sweep {wildcards.sweep} "
        "--output {output} "
        "2> {log}"


rule resolve_sweep_estimation_config:
    """Resolve one combination's estimation config slice.

    Separate rules per stage, for the same reason as the single-run
    ``resolve_<stage>_config`` rules: one rule emitting every slice would mark
    them all updated and rerun the whole sweep whenever any one changed.
    """
    input:
        manifest="sweeps/{sweep}/manifest.json",
        code=_src("orbitdet.pipeline.run_resolve_sweep_config"),
    output:
        "sweeps/{sweep}/{combo}/config/estimation.yaml",
    params:
        experiment=lambda wildcards: _experiment_for(wildcards.sweep),
        outdir="sweeps/{sweep}/{combo}/config",
        overrides=sweep_overrides,
    log:
        "sweeps/{sweep}/{combo}/config/resolve_estimation.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_resolve_sweep_config "
        "--experiment {params.experiment} "
        "--output-dir {params.outdir} "
        "--only estimation "
        "--overrides {params.overrides} "
        "2> {log}"


rule resolve_sweep_covariance_config:
    """Resolve one combination's covariance config slice."""
    input:
        manifest="sweeps/{sweep}/manifest.json",
        code=_src("orbitdet.pipeline.run_resolve_sweep_config"),
    output:
        "sweeps/{sweep}/{combo}/config/covariance.yaml",
    params:
        experiment=lambda wildcards: _experiment_for(wildcards.sweep),
        outdir="sweeps/{sweep}/{combo}/config",
        overrides=sweep_overrides,
    log:
        "sweeps/{sweep}/{combo}/config/resolve_covariance.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_resolve_sweep_config "
        "--experiment {params.experiment} "
        "--output-dir {params.outdir} "
        "--only covariance "
        "--overrides {params.overrides} "
        "2> {log}"


rule sweep_estimation:
    """Run the estimation for one combination.

    Depends on the base experiment's weighted collection: the sweep varies only
    estimation settings, so the preprocessing is shared.
    """
    input:
        cfg="sweeps/{sweep}/{combo}/config/estimation.yaml",
        obs=sweep_shared("preprocessing/weighted.tudat"),
        code=_src("orbitdet.pipeline.run_estimation"),
    output:
        out="sweeps/{sweep}/{combo}/estimation/estimation_output.tudat",
        summary="sweeps/{sweep}/{combo}/estimation/estimation_summary.json",
        log="sweeps/{sweep}/{combo}/estimation/estimation_log.log",
    params:
        outdir="sweeps/{sweep}/{combo}/estimation",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_estimation "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--observations {input.obs} "


rule sweep_postfit_propagation:
    """Propagate one combination's estimated state.

    Needed because the covariance stage defines its RSW frame from a trajectory,
    and each combination estimates a *different* state. Reusing the base
    experiment's post-fit trajectory would rotate this combination's covariance
    into a frame belonging to a different fit -- which is exactly the kind of
    silent mismatch this file's sharing rule exists to prevent.

    Separate from ``sweep_estimation`` rather than folded into it so the
    propagation re-runs only when this combination's estimate changes.
    """
    input:
        cfg="sweeps/{sweep}/{combo}/config/estimation.yaml",
        obs=sweep_shared("preprocessing/weighted.tudat"),
        meta=sweep_shared("dataset_metadata.json"),
        estimation="sweeps/{sweep}/{combo}/estimation/estimation_summary.json",
        code=_src("orbitdet.pipeline.run_propagation"),
    output:
        results="sweeps/{sweep}/{combo}/propagation/postfit_propagation_result.tudat",
        obs="sweeps/{sweep}/{combo}/propagation/postfit_observations_with_residuals.tudat",
        csv="sweeps/{sweep}/{combo}/propagation/postfit_residuals.csv",
    params:
        outdir="sweeps/{sweep}/{combo}/propagation",
        estimation_dir="sweeps/{sweep}/{combo}/estimation",
    log:
        "sweeps/{sweep}/{combo}/propagation/postfit_propagation.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_propagation "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--observations {input.obs} "
        "--metadata {input.meta} "
        "--tag postfit "
        "--estimation-dir {params.estimation_dir} "
        "2> {log}"


rule sweep_covariance:
    """Propagate one combination's covariance over the full arc.

    Present so a combination's metrics reflect whether its covariance is usable:
    for ``fit_to_ephemeris`` different parameter subsets have different
    normal-matrix rank, so this is a genuine per-combination result rather than
    a repeated calculation.
    """
    input:
        cfg="sweeps/{sweep}/{combo}/config/covariance.yaml",
        obs=sweep_shared("preprocessing/weighted.tudat"),
        estimation="sweeps/{sweep}/{combo}/estimation/estimation_output.tudat",
        states="sweeps/{sweep}/{combo}/propagation/postfit_propagation_result.tudat",
        code=_src("orbitdet.pipeline.run_covariance"),
    output:
        npz="sweeps/{sweep}/{combo}/covariance/propagated_formal_errors.npz",
        summary="sweeps/{sweep}/{combo}/covariance/covariance_summary.json",
    params:
        outdir="sweeps/{sweep}/{combo}/covariance",
    log:
        "sweeps/{sweep}/{combo}/covariance/covariance.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_covariance "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--observations {input.obs} "
        "--estimation {input.estimation} "
        "--states {input.states} "
        "2> {log}"


rule sweep_telemetry:
    """Aggregate every combination's metrics into one table.

    Depends on all combinations' estimation *and* covariance summaries, so it is
    the sweep's barrier: it runs only once the whole grid is done, and re-runs
    if any one changes.
    """
    input:
        manifest="sweeps/{sweep}/manifest.json",
        summaries=sweep_estimation_summaries,
        covariances=sweep_covariance_summaries,
        code=_src("orbitdet.pipeline.run_sweep_telemetry"),
    output:
        report(
            "sweeps/{sweep}/sweep_telemetry.csv",
            caption=_caption("tables", "sweep_telemetry.rst"),
            category="Sweep comparison",
        ),
    log:
        "sweeps/{sweep}/sweep_telemetry.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_sweep_telemetry "
        "--sweep {wildcards.sweep} "
        "--output {output} "
        "2> {log}"


rule sweep_figures_config:
    """Resolve the figure-styling slice used by the comparison charts.

    The base experiment's ``figures`` slice, written under the sweep root so the
    comparison figures are self-contained: their styling comes from the same
    ``conf/figures/`` tree as every other figure.
    """
    input:
        conf=CONF_FIGURES,
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "sweeps/{sweep}/config/figures.yaml",
    params:
        experiment=lambda wildcards: _experiment_for(wildcards.sweep),
        outdir="sweeps/{sweep}/config",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {params.experiment} "
        "--output-dir {params.outdir} "
        "--only figures "


rule sweep_comparison_figure:
    """Render one comparison bar chart per metric.

    ``{metric}`` is a wildcard rather than a loop over ``SWEEP_METRICS`` so each
    chart is its own job: editing one metric's styling re-renders that chart
    alone.
    """
    input:
        cfg="sweeps/{sweep}/config/figures.yaml",
        telemetry="sweeps/{sweep}/sweep_telemetry.csv",
        code=_src("orbitdet.pipeline.run_sweep_figure"),
    output:
        report(
            "sweeps/{sweep}/figures/sweep_comparison_{metric}.pdf",
            caption=_caption("sweeps", "sweep_comparison.rst"),
            category="Sweep comparison",
        ),
    params:
        outdir="sweeps/{sweep}/figures",
    log:
        "sweeps/{sweep}/figures/sweep_comparison_{metric}.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_sweep_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--telemetry {input.telemetry} "
        "--sweep {wildcards.sweep} "
        "--metric {wildcards.metric} "
        "2> {log}"
