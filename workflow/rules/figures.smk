"""Shared figure rules.

Figures are the report's content, so they are also where config isolation
matters most: a change to figure styling must re-render the figure and nothing
else. Every rule here depends only on the ``figures`` config slice plus the
artifact it draws from.

One CLI (``run_figure``) covers every family, dispatched by ``--kind``. Keeping
a single entry point means each rule keeps the same shape rather than needing a
different invocation style per family.

Rules are written out explicitly rather than generated in a loop: Snakemake's
``rule`` syntax is not valid Python, so rules cannot be produced with ``exec``
or a comprehension.

Each figure also declares a ``report(...)`` caption file, which is what makes
``snakemake --report`` lay the PDF out with a description rather than as a bare
file link.
"""


rule figure_prefit_residuals_spice:
    """Residuals wrt the SPICE ephemeris: the reference the outlier and
    # weighting stages use."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/prefit_ephemeris/observations_with_residuals.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_residuals_spice.pdf",
            caption=_caption("figures", "prefit_residuals_spice.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_residuals_spice",
        kind="residuals",
    log:
        "results/{experiment}/figures/prefit_residuals_spice.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "2> {log}"

rule figure_prefit_residuals_spice_scan:
    """Scan-frame residuals wrt the SPICE ephemeris (Gaia only)."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/prefit_ephemeris/observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_residuals_spice_scan.pdf",
            caption=_caption("figures", "prefit_residuals_spice_scan.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_residuals_spice_scan",
        kind="residuals",
    log:
        "results/{experiment}/figures/prefit_residuals_spice_scan.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_prefit_normalized_residuals:
    """Normalized residuals wrt the SPICE ephemeris."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/prefit_ephemeris/observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_normalized_residuals.pdf",
            caption=_caption("figures", "prefit_normalized_residuals.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_normalized_residuals",
        kind="residuals",
    log:
        "results/{experiment}/figures/prefit_normalized_residuals.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_prefit_normalized_residuals_scan:
    """Normalized scan-frame residuals wrt the SPICE ephemeris (Gaia only)."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/prefit_ephemeris/observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_normalized_residuals_scan.pdf",
            caption=_caption("figures", "prefit_normalized_residuals_scan.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_normalized_residuals_scan",
        kind="residuals",
    log:
        "results/{experiment}/figures/prefit_normalized_residuals_scan.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_prefit_residuals_prop:
    """Residuals wrt the pre-fit propagated trajectory."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/prefit_observations_with_residuals.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_residuals_prop.pdf",
            caption=_caption("figures", "prefit_residuals_prop.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_residuals_prop",
        kind="residuals",
    log:
        "results/{experiment}/figures/prefit_residuals_prop.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "2> {log}"

rule figure_prefit_residuals_prop_scan:
    """Scan-frame residuals wrt the pre-fit propagated trajectory (Gaia only)."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/prefit_observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_residuals_prop_scan.pdf",
            caption=_caption("figures", "prefit_residuals_prop_scan.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_residuals_prop_scan",
        kind="residuals",
    log:
        "results/{experiment}/figures/prefit_residuals_prop_scan.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_postfit_residuals:
    """Residuals wrt the post-fit propagated trajectory."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/postfit_residuals.pdf",
            caption=_caption("figures", "postfit_residuals.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="postfit_residuals",
        kind="residuals",
    log:
        "results/{experiment}/figures/postfit_residuals.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "2> {log}"

rule figure_postfit_residuals_scan:
    """Scan-frame residuals wrt the post-fit trajectory (Gaia only)."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/postfit_residuals_scan.pdf",
            caption=_caption("figures", "postfit_residuals_scan.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="postfit_residuals_scan",
        kind="residuals",
    log:
        "results/{experiment}/figures/postfit_residuals_scan.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_postfit_normalized_residuals:
    """Normalized residuals wrt the post-fit propagated trajectory."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/postfit_normalized_residuals.pdf",
            caption=_caption("figures", "postfit_normalized_residuals.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="postfit_normalized_residuals",
        kind="residuals",
    log:
        "results/{experiment}/figures/postfit_normalized_residuals.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_postfit_normalized_residuals_scan:
    """Normalized scan-frame residuals wrt the post-fit trajectory (Gaia only)."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        metadata="results/{experiment}/dataset_metadata.json",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/postfit_normalized_residuals_scan.pdf",
            caption=_caption("figures", "postfit_normalized_residuals_scan.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="postfit_normalized_residuals_scan",
        kind="residuals",
    log:
        "results/{experiment}/figures/postfit_normalized_residuals_scan.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "--metadata {input.metadata} "
        "2> {log}"

rule figure_residual_qq:
    """QQ plot of the post-fit residuals."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/residual_qq.pdf",
            caption=_caption("figures", "residual_qq.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="residual_qq",
        kind="residuals",
    log:
        "results/{experiment}/figures/residual_qq.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "2> {log}"

rule figure_residual_scatter:
    """Residual scatter of the post-fit residuals."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/residual_scatter.pdf",
            caption=_caption("figures", "residual_scatter.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="residual_scatter",
        kind="residuals",
    log:
        "results/{experiment}/figures/residual_scatter.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "2> {log}"

rule figure_residual_histogram:
    """Histogram of the post-fit residuals."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        observations="results/{experiment}/propagation/postfit_observations_with_residuals.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/residual_histogram.pdf",
            caption=_caption("figures", "residual_histogram.rst"),
            category="residuals",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="residual_histogram",
        kind="residuals",
    log:
        "results/{experiment}/figures/residual_histogram.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--observations {input.observations} "
        "2> {log}"

rule figure_parameter_correlation_heatmap:
    """Correlation matrix of the estimated parameters."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        estimation="results/{experiment}/estimation/estimation_output.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/parameter_correlation_heatmap.pdf",
            caption=_caption("figures", "parameter_correlation_heatmap.rst"),
            category="estimation",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="parameter_correlation_heatmap",
        kind="estimation",
    log:
        "results/{experiment}/figures/parameter_correlation_heatmap.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--estimation {input.estimation} "
        "2> {log}"

rule figure_parameter_history_per_iteration:
    """Parameter evolution across iterations."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        estimation="results/{experiment}/estimation/estimation_output.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/parameter_history_per_iteration.pdf",
            caption=_caption("figures", "parameter_history_per_iteration.rst"),
            category="estimation",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="parameter_history_per_iteration",
        kind="estimation",
    log:
        "results/{experiment}/figures/parameter_history_per_iteration.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--estimation {input.estimation} "
        "2> {log}"

rule figure_state_history_3d:
    """3D view of the post-fit trajectory."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        states="results/{experiment}/propagation/postfit_propagation_result.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/state_history_3d.pdf",
            caption=_caption("figures", "state_history_3d.rst"),
            category="trajectory",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="state_history_3d",
        kind="trajectory",
    log:
        "results/{experiment}/figures/state_history_3d.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--states {input.states} "
        "2> {log}"

rule figure_prefit_rsw:
    """RSW offset between the ephemeris and the pre-fit trajectory."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        kernels_cfg="results/{experiment}/config/kernels.yaml",
        states="results/{experiment}/propagation/prefit_propagation_result.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/prefit_rsw.pdf",
            caption=_caption("figures", "prefit_rsw.rst"),
            category="rsw",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="prefit_rsw",
        kind="rsw",
    log:
        "results/{experiment}/figures/prefit_rsw.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--states {input.states} "
        "--kernels-config {input.kernels_cfg} "
        "2> {log}"

rule figure_postfit_rsw:
    """RSW offset between the ephemeris and the post-fit trajectory."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        kernels_cfg="results/{experiment}/config/kernels.yaml",
        states="results/{experiment}/propagation/postfit_propagation_result.tudat",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/postfit_rsw.pdf",
            caption=_caption("figures", "postfit_rsw.rst"),
            category="rsw",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="postfit_rsw",
        kind="rsw",
    log:
        "results/{experiment}/figures/postfit_rsw.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--states {input.states} "
        "--kernels-config {input.kernels_cfg} "
        "2> {log}"

rule figure_propagated_formal_errors_cartesian:
    """Propagated formal errors in Cartesian components."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        formal_errors="results/{experiment}/covariance/propagated_formal_errors.npz",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/propagated_formal_errors_cartesian.pdf",
            caption=_caption("figures", "propagated_formal_errors_cartesian.rst"),
            category="formal_errors",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="propagated_formal_errors_cartesian",
        kind="formal_errors",
    log:
        "results/{experiment}/figures/propagated_formal_errors_cartesian.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--formal-errors {input.formal_errors} "
        "2> {log}"

rule figure_propagated_formal_errors_rsw:
    """Propagated formal errors in RSW components."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        formal_errors="results/{experiment}/covariance/propagated_formal_errors.npz",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/propagated_formal_errors_rsw.pdf",
            caption=_caption("figures", "propagated_formal_errors_rsw.rst"),
            category="formal_errors",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="propagated_formal_errors_rsw",
        kind="formal_errors",
    log:
        "results/{experiment}/figures/propagated_formal_errors_rsw.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--formal-errors {input.formal_errors} "
        "2> {log}"

rule figure_RSW_distance_with_uncertainty:
    """RSW offset with formal-error envelopes."""
    input:
        cfg="results/{experiment}/config/figures.yaml",
        formal_errors="results/{experiment}/covariance/propagated_formal_errors.npz",
        code=_src("orbitdet.pipeline.run_figure"),
    output:
        report(
            "results/{experiment}/figures/RSW_distance_with_uncertainty.pdf",
            caption=_caption("figures", "RSW_distance_with_uncertainty.rst"),
            category="rsw_uncertainty",
        ),
    params:
        outdir="results/{experiment}/figures",
        name="RSW_distance_with_uncertainty",
        kind="rsw_uncertainty",
    log:
        "results/{experiment}/figures/RSW_distance_with_uncertainty.log",
    shell:
        f"{PYTHON} -m orbitdet.pipeline.run_figure "
        "--config {input.cfg} "
        "--output-dir {params.outdir} "
        "--kind {params.kind} "
        "--name {params.name} "
        "--formal-errors {input.formal_errors} "
        "2> {log}"

#: Every figure rule defined above, keyed by name. Experiment files select
#: from this so a typo in a figure name fails at parse time rather than
#: silently producing nothing.
FIGURE_OUTPUTS = {
    "prefit_residuals_spice": "results/{experiment}/figures/prefit_residuals_spice.pdf",
    "prefit_residuals_spice_scan": "results/{experiment}/figures/prefit_residuals_spice_scan.pdf",
    "prefit_normalized_residuals": "results/{experiment}/figures/prefit_normalized_residuals.pdf",
    "prefit_normalized_residuals_scan": "results/{experiment}/figures/prefit_normalized_residuals_scan.pdf",
    "prefit_residuals_prop": "results/{experiment}/figures/prefit_residuals_prop.pdf",
    "prefit_residuals_prop_scan": "results/{experiment}/figures/prefit_residuals_prop_scan.pdf",
    "postfit_residuals": "results/{experiment}/figures/postfit_residuals.pdf",
    "postfit_residuals_scan": "results/{experiment}/figures/postfit_residuals_scan.pdf",
    "postfit_normalized_residuals": "results/{experiment}/figures/postfit_normalized_residuals.pdf",
    "postfit_normalized_residuals_scan": "results/{experiment}/figures/postfit_normalized_residuals_scan.pdf",
    "residual_qq": "results/{experiment}/figures/residual_qq.pdf",
    "residual_scatter": "results/{experiment}/figures/residual_scatter.pdf",
    "residual_histogram": "results/{experiment}/figures/residual_histogram.pdf",
    "parameter_correlation_heatmap": "results/{experiment}/figures/parameter_correlation_heatmap.pdf",
    "parameter_history_per_iteration": "results/{experiment}/figures/parameter_history_per_iteration.pdf",
    "state_history_3d": "results/{experiment}/figures/state_history_3d.pdf",
    "prefit_rsw": "results/{experiment}/figures/prefit_rsw.pdf",
    "postfit_rsw": "results/{experiment}/figures/postfit_rsw.pdf",
    "propagated_formal_errors_cartesian": "results/{experiment}/figures/propagated_formal_errors_cartesian.pdf",
    "propagated_formal_errors_rsw": "results/{experiment}/figures/propagated_formal_errors_rsw.pdf",
    "RSW_distance_with_uncertainty": "results/{experiment}/figures/RSW_distance_with_uncertainty.pdf",
}
