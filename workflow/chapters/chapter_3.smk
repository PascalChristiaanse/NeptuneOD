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
        python_dependencies("scripts/plot_residuals_figure.py"),
        hydra_dependencies("conf/plot_residuals_figure.yaml"),
        "conf/figures/prefit_residuals_spice.yaml",
        "scripts/plot_residuals_figure.py",

    output:
        f"{STAGE}/prefit_residuals_spice.pdf",

    log:
        f"{STAGE}/prefit_residuals_spice.log",

    shell:
        """
        {PYTHON} scripts/plot_residuals_figure.py \
            hydra.run.dir={STAGE} \
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
        "scripts/plot_residuals_figure.py",
        python_dependencies("scripts/plot_residuals_figure.py"),
        hydra_dependencies("conf/plot_residuals_figure.yaml"),
        "conf/figures/residual_scatter.yaml",

    output:
        f"{STAGE}/residual_scatter.pdf",

    log:
        f"{STAGE}/residual_scatter.log",

    shell:
        """
        {PYTHON} scripts/plot_residuals_figure.py \
            hydra.run.dir={STAGE} \
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
        "scripts/plot_residuals_figure.py",
        python_dependencies("scripts/plot_residuals_figure.py"),
        hydra_dependencies("conf/plot_residuals_figure.yaml"),
        "conf/figures/residual_histogram.yaml",

    output:
        f"{STAGE}/residual_histogram.pdf",

    log:
        f"{STAGE}/residual_histogram.log",

    shell:
        """
        {PYTHON} scripts/plot_residuals_figure.py \
            hydra.run.dir={STAGE} \
            figure_config=conf/figures/residual_histogram.yaml \
            observations={STAGE}/observations.tudat \
            standard_frame={STAGE}/standard_frame.parquet \
            > {log} 2>&1
        """


POLE_MODELS = [
    "Pole_Model_Jacobson2009",
    "IAU2015",
]
rule propagate_trajectory_pole_sweep:
    input:
        python_dependencies("scripts/propagate_trajectory.py"),
        hydra_dependencies("conf/chapter_3/prefit_propagation.yaml"),

    output:
        f"{STAGE}/pole_sweep/{{pole_model}}/propagation_result.tudat",

    wildcard_constraints:
        pole_model="|".join(POLE_MODELS),

    log:
        f"{STAGE}/pole_sweep/{{pole_model}}/propagate.log",

    shell:
        """
        {PYTHON} scripts/propagate_trajectory.py \
            --config-name chapter_3/prefit_propagation \
            hydra.run.dir={STAGE}/pole_sweep/{wildcards.pole_model} \
            bodies_to_create.Neptune.rotation_model.type={wildcards.pole_model} \
            > {log} 2>&1
        """

rule rsw_prop_two_poles:
    """RSW separation for every pole model, overlaid on one figure."""

    input:
        expand(
            f"{STAGE}/pole_sweep/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/plot_simulation_figure.yaml"),
        "conf/chapter_3/fig_rsw_prop_two_poles.yaml",

    output:
        f"{STAGE}/rsw_prop_two_poles.pdf",

    params:
        # The results are passed in POLE_MODELS order, which is also the order
        # the figure config's ``labels`` must list them in.
        results=",".join(
            expand(
                f"{STAGE}/pole_sweep/{{pole_model}}/propagation_result.tudat",
                pole_model=POLE_MODELS,
            )
        ),

    log:
        f"{STAGE}/rsw_prop_two_poles.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={STAGE} \
            figure_config=conf/chapter_3/fig_rsw_prop_two_poles.yaml \
            +simulation_results=[{params.results}] \
            > {log} 2>&1
        """


rule trajectory_rsw_difference:
    """RSW difference between the two pole models (Jacobson minus IAU 2015)."""

    input:
        expand(
            f"{STAGE}/pole_sweep/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        "scripts/plot_simulation_figure.py",
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/plot_simulation_figure.yaml"),
        "conf/figures/trajectory_rsw_difference.yaml",

    output:
        f"{STAGE}/trajectory_rsw_difference.pdf",

    log:
        f"{STAGE}/trajectory_rsw_difference.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={STAGE} \
            figure_config=conf/figures/trajectory_rsw_difference.yaml \
            simulation_result={STAGE}/pole_sweep/Pole_Model_Jacobson2009/propagation_result.tudat \
            +simulation_results=[{STAGE}/pole_sweep/IAU2015/propagation_result.tudat] \
            > {log} 2>&1
        """


rule all_chapter_3:
    """Everything chapter 3 needs: the artifacts, every figure, and the sweep."""

    input:
        f"{STAGE}/observations.tudat",
        f"{STAGE}/standard_frame.parquet",
        f"{STAGE}/prefit_residuals_spice.pdf",
        f"{STAGE}/residual_scatter.pdf",
        f"{STAGE}/residual_histogram.pdf",
        # One propagation per pole model.  expand() consumes the wildcard here,
        # so the sweep needs no separate target rule.
        expand(
            f"{STAGE}/pole_sweep/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        f"{STAGE}/rsw_prop_two_poles.pdf",
        f"{STAGE}/trajectory_rsw_difference.pdf",
