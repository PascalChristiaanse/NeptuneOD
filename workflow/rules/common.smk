"""Shared Snakemake rules for every experiment.

This module holds the pipeline's stage rules once, so each experiment's
``.smk`` file only declares which targets it wants and which figures it draws.
That keeps analyses methodologically independent while sharing the execution
machinery, which is the arrangement ``docs/snakemake_design.md`` recommends.

Two classes of input are declared explicitly because Snakemake cannot see them
on its own:

``code=_src(...)``
    The ``code`` rerun trigger tracks the rule definition and any file named by
    ``script:``, but it does NOT follow Python imports. ``workflow/depgraph.py``
    parses the package's import statements and returns the transitive closure
    for each stage's entry module, so a rule depends on exactly the files it can
    import -- the static equivalent of CMake's compiler-generated depfiles.
    Declaring the whole package instead would make every rule depend on every
    file, so editing a plotter would invalidate the kernel download.

``conf=...``
    Snakemake only tracks its own ``configfile:``. The Hydra tree in ``conf/``
    is read inside the stage processes, so without this a change to
    ``conf/weighting/router.yaml`` would silently leave stale outputs in place.
    The ``resolve_<stage>_config`` rules turn that tree into tracked per-stage
    artifacts, and each stage rule depends on its own slice.

Output layout, per experiment::

    results/<experiment>/
    ├── config/<stage>.yaml        resolved, tracked config slices
    ├── kernels.done
    ├── observations.tudat
    ├── dataset_metadata.json
    ├── prefit_ephemeris/
    │   ├── observations_with_residuals.tudat
    │   └── residuals.csv
    ├── preprocessing/
    │   ├── outliers_applied.tudat
    │   ├── outlier_rejection_metadata.json
    │   ├── weighted.tudat
    │   └── observation_weights.csv
    ├── propagation/{prefit,postfit}_*
    ├── estimation/
    ├── covariance/
    ├── figures/
    └── tables/
"""

import glob
import os
import sys

sys.path.insert(0, workflow.basedir)
import depgraph  # noqa: E402

#: The graph builder is itself a dependency: changing how dependencies are
#: computed must invalidate everything.
DEPGRAPH = "workflow/depgraph.py"

#: The caption/description files handed to ``report(...)``.
#:
#: Snakemake resolves a caption path relative to the *including* file -- so
#: ``caption="report/x.rst"`` written inside ``workflow/rules/`` resolves to
#: ``workflow/rules/report/x.rst``, not to the repository's ``report/`` tree.
#: An absolute path sidesteps that rule entirely, which is why this is built
#: from ``workflow.basedir`` rather than written as a literal relative path.
#:
#: ``workflow.basedir`` is the directory of the main Snakefile -- ``workflow/``
#: when the workflow is launched with ``-s workflow/Snakefile`` -- so ``report/``
#: is its sibling.
REPORT_DIR = os.path.normpath(
    os.path.join(workflow.basedir, os.pardir, "report")
)


def _caption(*parts):
    """Absolute path to a caption file under the ``report/`` tree.

    Args are joined as path components, so ``_caption("figures", "x.rst")``
    yields ``<repo>/report/figures/x.rst``. Absolute is deliberate: see
    :data:`REPORT_DIR`.
    """
    return os.path.join(REPORT_DIR, *parts)

#: Interpreter used for every stage command.
#:
#: ``sys.executable`` rather than a bare ``python``: the stages need tudatpy and
#: orbitdet, which live in the environment Snakemake was launched from. A bare
#: ``python`` resolves through PATH and can silently pick a different
#: environment (the conda base env, for instance), where the import fails.
PYTHON = sys.executable


def _src(*entries):
    """Files a stage can import, directly or indirectly."""
    return [DEPGRAPH] + depgraph.stage_sources(entries)


# ---------------------------------------------------------------------------
# Config tracking
# ---------------------------------------------------------------------------
# Snakemake only knows about its own configfile: and the files named in rules.
# The Hydra tree in conf/ is read *inside* the stage processes, so without the
# resolve rules a change there is invisible and outputs go stale silently.
#
# The tree is split by concern so a change invalidates only the stages it can
# affect: editing conf/figures/residuals.yaml must not rebuild the observation
# collection.
#
# NOTE: ``recursive=True`` is required. Without it ``conf/kernels/**/*.yaml``
# matches nothing, because ``**`` only spans directories when recursion is on.
def _conf(*subdirs):
    """Collect every YAML under the given ``conf/`` subdirectories.

    ``conf/experiment`` is deliberately excluded: it holds one file per
    experiment, and globbing it would make every experiment depend on every
    other experiment's config. The experiment's own file is added per-wildcard
    by :func:`_conf_for` instead.
    """
    files = []
    for subdir in subdirs:
        files.extend(glob.glob(f"conf/{subdir}/**/*.yaml", recursive=True))
    return sorted(files)


#: Keys read by ``build_context`` (epochs, seed) and ``setup_logging``.
CONF_CONTEXT = _conf("logging")
#: ``KernelManager`` reads the kernel and data configuration.
CONF_KERNELS = CONF_CONTEXT + _conf("kernels")
#: Everything the observation build and the environment need.
CONF_OBSERVATIONS = CONF_KERNELS + _conf("collection", "datasets", "data", "dynamics", "parameters")
#: Outlier rejection and its (optional) own grouping.
CONF_OUTLIERS = CONF_OBSERVATIONS + _conf("outlier_rejection")
#: Weighting and the shared grouping.
CONF_WEIGHTING = CONF_OBSERVATIONS + _conf("weighting")
#: Estimation settings and the a priori covariance.
CONF_ESTIMATION = CONF_OBSERVATIONS + _conf("parameters")
#: Figure styling only. This is the slice that must NOT invalidate compute.
CONF_FIGURES = CONF_CONTEXT + _conf("figures")

#: Map each stage to the conf globs that can affect it. The stage -> *keys*
#: mapping lives in ``orbitdet.pipeline.resolve.STAGE_KEYS``; this only says
#: which directories can reach each stage.
STAGE_CONF = {
    "kernels": CONF_KERNELS,
    "observations": CONF_OBSERVATIONS,
    "outlier_rejection": CONF_OUTLIERS,
    "weighting": CONF_WEIGHTING,
    "dynamics": CONF_OBSERVATIONS,
    "prefit": CONF_OBSERVATIONS,
    "estimation": CONF_ESTIMATION,
    "covariance": CONF_ESTIMATION,
    "figures": CONF_FIGURES,
}


def _conf_for(wildcards, stage):
    """Conf files that can affect *stage* for the experiment in *wildcards*.

    An input function rather than a static list so the experiment's own config
    is included without pulling in every other experiment's. Editing
    ``conf/experiment/wa_fit_pl.yaml`` must not invalidate this experiment.
    """
    return [f"conf/experiment/{wildcards.experiment}.yaml"] + STAGE_CONF[stage]


def _run(module, *extra, log=None):
    """Compose a stage command with its config and output directory.

    Keeping this in one helper means every rule's shell command has the same
    shape, so a change to the CLI convention is a one-line edit.
    """
    parts = [
        f"{PYTHON} -m orbitdet.pipeline.{module}",
        "--config {input.cfg}",
        "--output-dir {params.outdir}",
        *extra,
    ]
    command = " ".join(parts)
    if log is not None:
        command += f" 2> {log}"
    return command


# ---------------------------------------------------------------------------
# Resolve rules: one per config slice
# ---------------------------------------------------------------------------
# Written out explicitly rather than generated in a loop: Snakemake's ``rule``
# syntax is not valid Python, so rules cannot be produced with ``exec`` or a
# comprehension. One rule per slice is essential -- a single rule emitting every
# slice would mark them all as updated, so editing a figure title would
# invalidate the observation build and the estimation.


rule resolve_kernels_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "kernels"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/kernels.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only kernels"

rule resolve_observations_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "observations"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/observations.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only observations"

rule resolve_outlier_rejection_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "outlier_rejection"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/outlier_rejection.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only outlier_rejection"

rule resolve_weighting_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "weighting"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/weighting.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only weighting"

rule resolve_dynamics_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "dynamics"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/dynamics.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only dynamics"

rule resolve_prefit_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "prefit"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/prefit.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only prefit"

rule resolve_estimation_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "estimation"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/estimation.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only estimation"

rule resolve_covariance_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "covariance"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/covariance.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only covariance"

rule resolve_figures_config:
    input:
        conf=lambda wildcards: _conf_for(wildcards, "figures"),
        code=_src("orbitdet.pipeline.run_resolve_config"),
    output:
        "results/{experiment}/config/figures.yaml",
    params:
        outdir="results/{experiment}/config",
    shell:
        "{PYTHON} -m orbitdet.pipeline.run_resolve_config "
        "--experiment {wildcards.experiment} --output-dir {params.outdir} "
        "--only figures"

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


rule kernels:
    """Download and furnish SPICE kernels; writes a stamp on success."""
    input:
        cfg="results/{experiment}/config/kernels.yaml",
        code=_src("orbitdet.pipeline.run_kernels"),
    output:
        touch("results/{experiment}/kernels.done"),
    params:
        outdir="results/{experiment}",
    shell:
        _run("run_kernels")


rule observations:
    """Build the observation collection and its metadata sidecar."""
    input:
        cfg="results/{experiment}/config/observations.yaml",
        stamp="results/{experiment}/kernels.done",
        code=_src("orbitdet.pipeline.run_observations"),
    output:
        obs="results/{experiment}/observations.tudat",
        meta="results/{experiment}/dataset_metadata.json",
    params:
        outdir="results/{experiment}",
    shell:
        _run("run_observations")


rule ephemeris_residuals:
    """Residuals against the SPICE ephemeris.

    This is the reference outlier rejection and weighting compare against, so it
    must run before either.
    """
    input:
        cfg="results/{experiment}/config/prefit.yaml",
        obs="results/{experiment}/observations.tudat",
        meta="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_ephemeris_residuals"),
    output:
        obs="results/{experiment}/prefit_ephemeris/observations_with_residuals.tudat",
        csv="results/{experiment}/prefit_ephemeris/residuals.csv",
    params:
        outdir="results/{experiment}/prefit_ephemeris",
    log:
        "results/{experiment}/prefit_ephemeris/ephemeris_residuals.log",
    shell:
        _run(
            "run_ephemeris_residuals",
            "--observations {input.obs}",
            "--metadata {input.meta}",
            log="{log}",
        )


rule outlier_rejection:
    """Filter outliers; forwards the collection unchanged when disabled."""
    input:
        cfg="results/{experiment}/config/outlier_rejection.yaml",
        obs="results/{experiment}/prefit_ephemeris/observations_with_residuals.tudat",
        meta="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_outliers"),
    output:
        obs="results/{experiment}/preprocessing/outliers_applied.tudat",
        report="results/{experiment}/preprocessing/outlier_rejection_metadata.json",
    params:
        outdir="results/{experiment}/preprocessing",
    log:
        "results/{experiment}/preprocessing/outlier_rejection.log",
    shell:
        _run(
            "run_outliers",
            "--observations {input.obs}",
            "--metadata {input.meta}",
            log="{log}",
        )


rule weighting:
    """Assign observation weights and write the per-observation table."""
    input:
        cfg="results/{experiment}/config/weighting.yaml",
        obs="results/{experiment}/preprocessing/outliers_applied.tudat",
        meta="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_weighting"),
    output:
        obs="results/{experiment}/preprocessing/weighted.tudat",
        weights="results/{experiment}/preprocessing/observation_weights.csv",
    params:
        outdir="results/{experiment}/preprocessing",
    log:
        "results/{experiment}/preprocessing/weighting.log",
    shell:
        _run(
            "run_weighting",
            "--observations {input.obs}",
            "--metadata {input.meta}",
            log="{log}",
        )


rule prefit_propagation:
    """Propagate from the configured initial state and residualise against it."""
    input:
        cfg="results/{experiment}/config/dynamics.yaml",
        obs="results/{experiment}/preprocessing/weighted.tudat",
        meta="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_propagation"),
    output:
        results="results/{experiment}/propagation/prefit_propagation_result.tudat",
        obs="results/{experiment}/propagation/prefit_observations_with_residuals.tudat",
        csv="results/{experiment}/propagation/prefit_residuals.csv",
    params:
        outdir="results/{experiment}/propagation",
    log:
        "results/{experiment}/propagation/prefit_propagation.log",
    shell:
        _run(
            "run_propagation",
            "--observations {input.obs}",
            "--metadata {input.meta}",
            "--tag prefit",
            log="{log}",
        )


rule estimation:
    """Run the least-squares fit.

    Saves the covariance-propagation payload, which is what allows the
    covariance stage to run as a separate process from the file alone.
    """
    input:
        cfg="results/{experiment}/config/estimation.yaml",
        obs="results/{experiment}/preprocessing/weighted.tudat",
        code=_src("orbitdet.pipeline.run_estimation"),
    output:
        out="results/{experiment}/estimation/estimation_output.tudat",
        summary="results/{experiment}/estimation/estimation_summary.json",
        log="results/{experiment}/estimation/estimation_log.log",
    params:
        outdir="results/{experiment}/estimation",
    shell:
        _run("run_estimation", "--observations {input.obs}")


rule postfit_propagation:
    """Propagate from the estimated parameters and residualise against it."""
    input:
        cfg="results/{experiment}/config/dynamics.yaml",
        obs="results/{experiment}/preprocessing/weighted.tudat",
        meta="results/{experiment}/dataset_metadata.json",
        estimation="results/{experiment}/estimation/estimation_summary.json",
        code=_src("orbitdet.pipeline.run_propagation"),
    output:
        results="results/{experiment}/propagation/postfit_propagation_result.tudat",
        obs="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        csv="results/{experiment}/propagation/postfit_residuals.csv",
    params:
        outdir="results/{experiment}/propagation",
        estimation_dir="results/{experiment}/estimation",
    log:
        "results/{experiment}/propagation/postfit_propagation.log",
    shell:
        _run(
            "run_propagation",
            "--observations {input.obs}",
            "--metadata {input.meta}",
            "--tag postfit",
            "--estimation-dir {params.estimation_dir}",
            log="{log}",
        )


rule covariance:
    """Propagate the estimated covariance over the full arc.

    Runs in a fresh process, rebuilding the state-transition interface from the
    payload stored in the estimation output.
    """
    input:
        cfg="results/{experiment}/config/covariance.yaml",
        obs="results/{experiment}/preprocessing/weighted.tudat",
        estimation="results/{experiment}/estimation/estimation_output.tudat",
        states="results/{experiment}/propagation/postfit_propagation_result.tudat",
        code=_src("orbitdet.pipeline.run_covariance"),
    output:
        npz="results/{experiment}/covariance/propagated_formal_errors.npz",
        summary="results/{experiment}/covariance/covariance_summary.json",
    params:
        outdir="results/{experiment}/covariance",
    log:
        "results/{experiment}/covariance/covariance.log",
    shell:
        _run(
            "run_covariance",
            "--observations {input.obs}",
            "--estimation {input.estimation}",
            "--states {input.states}",
            log="{log}",
        )


rule weight_tables:
    """Build the per-source and per-group weight summary tables."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        weights="results/{experiment}/preprocessing/observation_weights.csv",
        code=_src("orbitdet.pipeline.run_tables"),
    output:
        csv=report(
            "results/{experiment}/tables/weight_summary_source.csv",
            caption=_caption("tables", "weight_summary_source.rst"),
            category="Tables",
        ),
        tex="results/{experiment}/tables/weight_summary_source.tex",
        xlsx="results/{experiment}/tables/weight_summary.xlsx",
    params:
        outdir="results/{experiment}/tables",
    log:
        "results/{experiment}/tables/weight_tables.log",
    shell:
        _run("run_tables", "--weights {input.weights}", log="{log}")


rule report:
    """Assemble the experiment's summary report.

    Collects the JSON summaries each stage writes into one Markdown document.
    Snakemake's ``--report`` renders it alongside the figures, whose captions
    come from the ``report()`` directives on the figure rules.

    This rule is what makes the report a *derived artifact* rather than a
    side effect of running the pipeline: it re-runs when any summary changes,
    and it is listed in the experiment's target so a plain ``snakemake``
    produces it.
    """
    input:
        estimation="results/{experiment}/estimation/estimation_summary.json",
        covariance="results/{experiment}/covariance/covariance_summary.json",
        outliers="results/{experiment}/preprocessing/outlier_rejection_metadata.json",
        code=_src("orbitdet.pipeline.run_report"),
    output:
        report(
            "results/{experiment}/report.md",
            caption=_caption("report.rst"),
            category="Summary",
        ),
    params:
        results_dir="results/{experiment}",
    log:
        "results/{experiment}/report.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_report "
        "--experiment {wildcards.experiment} "
        "--results-dir {params.results_dir} "
        "--output {output} "
        "2> {log}"
