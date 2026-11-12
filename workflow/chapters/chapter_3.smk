"""Chapter 4: Investigation 1 - Repeating Atanas' 'Parameter Estimation with Simulated
Observations'"""

from orbitdet.reproducibility import hydra_dependencies, python_dependencies

STAGE = f"{RESULTS_ROOT}/chapter_3"

POLE_FOLDER = f"{STAGE}/pole_sweep"
POLE_MODELS = [
    "IAU2015",
    "Pole_Model_Jacobson2009",
]
rule propagate_trajectory_pole_sweep:
    input:
        python_dependencies("scripts/propagate_trajectory.py"),
        hydra_dependencies("conf/chapter_3/prefit_propagation.yaml"),

    output:
        f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",

    wildcard_constraints:
        pole_model="|".join(POLE_MODELS),

    log:
        f"{POLE_FOLDER}/{{pole_model}}/propagate.log",

    shell:
        """
        {PYTHON} scripts/propagate_trajectory.py \
            --config-name chapter_3/prefit_propagation \
            hydra.run.dir={POLE_FOLDER}/{wildcards.pole_model} \
            bodies_to_create.Neptune.rotation_model.type={wildcards.pole_model} \
            > {log} 2>&1
        """

rule rsw_prop_two_poles:
    """RSW separation for every pole model, overlaid on one figure."""

    input:
        expand(
            f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_rsw_prop_two_poles.yaml"),

    output:
        f"{POLE_FOLDER}/rsw_prop_two_poles.pdf",

    params:
        results=",".join(
            expand(
                f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
                pole_model=POLE_MODELS,
            )
        ),

    log:
        f"{POLE_FOLDER}/rsw_prop_two_poles.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={POLE_FOLDER} \
            figure_config=conf/chapter_3/fig_rsw_prop_two_poles.yaml \
            +simulation_results=[{params.results}] \
            > {log} 2>&1
        """

rule keplerian_prop_two_poles:
    """Keplerian elements wrt Triton spice for every pole model, overlaid on one figure."""

    input:
        expand(
            f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_keplerian_prop_two_poles.yaml"),

    output:
        f"{POLE_FOLDER}/keplerian_prop_two_poles.pdf",

    params:
        results=",".join(
            expand(
                f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
                pole_model=POLE_MODELS,
            )
        ),

    log:
        f"{POLE_FOLDER}/keplerian_prop_two_poles.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={POLE_FOLDER} \
            figure_config=conf/chapter_3/fig_keplerian_prop_two_poles.yaml \
            +simulation_results=[{params.results}] \
            > {log} 2>&1
        """

rule equinoctial_prop_two_poles:
    """Modified equinoctial elements wrt Triton spice, one line per pole model."""

    input:
        expand(
            f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_equinoctial_prop_two_poles.yaml"),

    output:
        f"{POLE_FOLDER}/equinoctial_prop_two_poles.pdf",

    params:
        # Every pole model's result, in POLE_MODELS order.  The figure config's
        # ``labels`` must list them in this same order.
        results=",".join(
            expand(
                f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
                pole_model=POLE_MODELS,
            )
        ),

    log:
        f"{POLE_FOLDER}/equinoctial_prop_two_poles.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={POLE_FOLDER} \
            figure_config=conf/chapter_3/fig_equinoctial_prop_two_poles.yaml \
            +simulation_results=[{params.results}] \
            > {log} 2>&1
        """


rule rsw_between_pole_models:
    """RSW separation between the two pole models over time."""

    input:
        expand(
            f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_rsw_between_pole_models.yaml"),

    output:
        f"{POLE_FOLDER}/rsw_between_pole_models.pdf",

    params:
        reference=f"{POLE_FOLDER}/{POLE_MODELS[0]}/propagation_result.tudat",
        comparisons=",".join(
            f"{POLE_FOLDER}/{pole_model}/propagation_result.tudat"
            for pole_model in POLE_MODELS[1:]
        ),

    log:
        f"{POLE_FOLDER}/rsw_between_pole_models.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={POLE_FOLDER} \
            figure_config=conf/chapter_3/fig_rsw_between_pole_models.yaml \
            simulation_result={params.reference} \
            +simulation_results=[{params.comparisons}] \
            > {log} 2>&1
        """

######################################################
############## Initial state estimation ##############
######################################################
INITIAL_ESTIMATION_FOLDER = f"{STAGE}/initial_state_estimation"
rule generate_synthetic_observations:
    """Generate synthetic observations for the initial state estimation."""

    input:
        python_dependencies("scripts/generate_observation_collection.py"),
        hydra_dependencies("conf/chapter_3/coll_synthetic_3d_point.yaml"),

    output:
        f"{INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
        f"{INITIAL_ESTIMATION_FOLDER}/prefit_standard_frame.parquet",

    log:
        f"{INITIAL_ESTIMATION_FOLDER}/generate_synthetic_observations.log",

    shell:
        """
        {PYTHON} scripts/generate_observation_collection.py \
            --config-name chapter_3/coll_synthetic_3d_point \
            hydra.run.dir={INITIAL_ESTIMATION_FOLDER} \
            output.observations=prefit_observations \
            output.standard_frame=prefit_standard_frame.parquet \
            > {log} 2>&1
        """


rule estimate_initial_state_pole_models:
    """Fit the Triton initial state once per Neptune pole model.

    The pole model changes Neptune's orientation, so the two fits are separate
    experiments rather than repeats of the same one.
    """

    input:
        python_dependencies("scripts/solve_estimation.py"),
        hydra_dependencies("conf/chapter_3/estimation_pole_models.yaml"),
        f"{INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",

    output:
        f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimation_output.tudat",
        f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/final_iteration_simulation_results.tudat",
        f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimation_log.log",

    wildcard_constraints:
        pole_model="|".join(POLE_MODELS),

    log:
        f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimate.log",

    shell:
        """
        {PYTHON} scripts/solve_estimation.py \
            --config-name chapter_3/estimation_pole_models \
            hydra.run.dir={INITIAL_ESTIMATION_FOLDER}/{wildcards.pole_model} \
            observations={INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat \
            bodies_to_create.Neptune.rotation_model.type={wildcards.pole_model} \
            > {log} 2>&1
        """


rule rsw_initial_state_two_poles:
    """RSW separation for the two initial state estimates, overlaid on one figure."""

    input:
        expand(
            f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/final_iteration_simulation_results.tudat",
            pole_model=POLE_MODELS,
        ),
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_rsw_initial_state_two_poles.yaml"),

    output:
        f"{INITIAL_ESTIMATION_FOLDER}/rsw_initial_state_two_poles.pdf",

    params:
        results=",".join(
            expand(
                f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/final_iteration_simulation_results.tudat",
                pole_model=POLE_MODELS,
            )
        ),

    log:
        f"{INITIAL_ESTIMATION_FOLDER}/rsw_initial_state_two_poles.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={INITIAL_ESTIMATION_FOLDER} \
            figure_config=conf/chapter_3/fig_rsw_initial_state_two_poles.yaml \
            +simulation_results=[{params.results}] \
            > {log} 2>&1
        """

rule all_chapter_3:
    """Everything chapter 3 needs: the artifacts, every figure, and the sweep."""

    input:
        # Paragraph 1
        expand(
            f"{POLE_FOLDER}/{{pole_model}}/propagation_result.tudat",
            pole_model=POLE_MODELS,
        ),
        f"{POLE_FOLDER}/rsw_prop_two_poles.pdf",
        f"{POLE_FOLDER}/keplerian_prop_two_poles.pdf",
        f"{POLE_FOLDER}/equinoctial_prop_two_poles.pdf",
        f"{POLE_FOLDER}/rsw_between_pole_models.pdf",
        f"{INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
        f"{INITIAL_ESTIMATION_FOLDER}/prefit_standard_frame.parquet",
        # Paragraph 2: initial state estimation, one fit per pole model
        expand(
            f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimation_output.tudat",
            pole_model=POLE_MODELS,
        ),
        expand(
            f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimation_log.log",
            pole_model=POLE_MODELS,
        ),
        # Paragraph 2: initial state estimation, one figure comparing the two fits
        f"{INITIAL_ESTIMATION_FOLDER}/rsw_initial_state_two_poles.pdf",
        # Paragraph 2: one covariance/correlation corner plot per pole model
        expand(
            f"{INITIAL_ESTIMATION_FOLDER}/rsw_correlations_{{pole_model}}.pdf",
            pole_model=POLE_MODELS,
        ),
