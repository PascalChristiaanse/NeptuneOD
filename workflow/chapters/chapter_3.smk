"""Chapter 4: Investigation 1 - Repeating Atanas' 'Parameter Estimation with Simulated
Observations'"""

from orbitdet.reproducibility import hydra_dependencies, python_dependencies

STAGE = f"{RESULTS_ROOT}/chapter_3"


POLE_MODELS = [
    "IAU2015",
    "Pole_Model_Jacobson2009",
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
        hydra_dependencies("conf/chapter_3/fig_rsw_prop_two_poles.yaml"),

    output:
        f"{STAGE}/rsw_prop_two_poles.pdf",

    params:
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


rule all_chapter_3:
    """Everything chapter 3 needs: the artifacts, every figure, and the sweep."""

    input:
        # Paragraph 1
        expand(
            f"{STAGE}/pole_sweep/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        f"{STAGE}/rsw_prop_two_poles.pdf",
        f"{STAGE}/trajectory_rsw_difference.pdf",
