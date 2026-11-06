"""Chapter 4: Investigation 1 - Repeating Atanas' 'Parameter Estimation with Simulated
Observations'"""

from orbitdet.reproducibility import hydra_dependencies, python_dependencies

STAGE = f"{RESULTS_ROOT}/chapter_3"


rule prefit_residuals:
    """Build the observation collection and compute its pre-fit residuals."""

    input:
        "scripts/prefit_residuals.py",
        python_dependencies("scripts/prefit_residuals.py"),
        hydra_dependencies("conf/prefit_residuals.yaml"),

    output:
        observations=f"{STAGE}/observations.tudat",
        standard_frame=f"{STAGE}/standard_frame.parquet",

    log:
        f"{STAGE}/generation.log",

    shell:
        """
        {PYTHON} scripts/prefit_residuals.py \
            hydra.run.dir={STAGE} \
            > {log} 2>&1
        """


rule prefit_residuals_spice:
    """Pre-fit residuals versus epoch, grouped by source set."""

    input:
        f"{STAGE}/observations.tudat",
        f"{STAGE}/standard_frame.parquet",
        python_dependencies("scripts/plot_figure.py"),
        hydra_dependencies("conf/plot_figure.yaml"),
        "conf/figures/prefit_residuals_spice.yaml",
        "scripts/plot_figure.py",

    output:
        f"{STAGE}/prefit_residuals_spice.pdf",

    log:
        f"{STAGE}/prefit_residuals_spice.log",

    shell:
        """
        {PYTHON} scripts/plot_figure.py \
            hydra.run.dir={STAGE} \
            figure=prefit_residuals_spice \
            figure_config=conf/figures/prefit_residuals_spice.yaml \
            observations={STAGE}/observations.tudat \
            standard_frame={STAGE}/standard_frame.parquet \
            > {log} 2>&1
        """


rule residual_scatter:
    """RA versus Dec residuals, grouped by observatory."""

    input:
        f"{STAGE}/observations.tudat",
        f"{STAGE}/standard_frame.parquet",
        "scripts/plot_figure.py",
        python_dependencies("scripts/plot_figure.py"),
        hydra_dependencies("conf/plot_figure.yaml"),
        "conf/figures/residual_scatter.yaml",

    output:
        f"{STAGE}/residual_scatter.pdf",

    log:
        f"{STAGE}/residual_scatter.log",

    shell:
        """
        {PYTHON} scripts/plot_figure.py \
            hydra.run.dir={STAGE} \
            figure=residual_scatter \
            figure_config=conf/figures/residual_scatter.yaml \
            observations={STAGE}/observations.tudat \
            standard_frame={STAGE}/standard_frame.parquet \
            > {log} 2>&1
        """


rule residual_histogram:
    """Residual histograms, grouped by observable type."""

    input:
        f"{STAGE}/observations.tudat",
        f"{STAGE}/standard_frame.parquet",
        "scripts/plot_figure.py",
        python_dependencies("scripts/plot_figure.py"),
        hydra_dependencies("conf/plot_figure.yaml"),
        "conf/figures/residual_histogram.yaml",

    output:
        f"{STAGE}/residual_histogram.pdf",

    log:
        f"{STAGE}/residual_histogram.log",

    shell:
        """
        {PYTHON} scripts/plot_figure.py \
            hydra.run.dir={STAGE} \
            figure=residual_histogram \
            figure_config=conf/figures/residual_histogram.yaml \
            observations={STAGE}/observations.tudat \
            standard_frame={STAGE}/standard_frame.parquet \
            > {log} 2>&1
        """


rule all_chapter_3:
    """Everything chapter 3 needs: the artifacts and every figure."""

    input:
        f"{STAGE}/observations.tudat",
        f"{STAGE}/standard_frame.parquet",
        f"{STAGE}/prefit_residuals_spice.pdf",
        f"{STAGE}/residual_scatter.pdf",
        f"{STAGE}/residual_histogram.pdf",
