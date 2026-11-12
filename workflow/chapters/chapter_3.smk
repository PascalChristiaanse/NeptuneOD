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

#: LaTeX styles the parameter-change table is written in.  Must match the
#: ``writer.styles`` list in conf/tables/parameter_change.yaml.
TABLE_STYLES = ["booktabs", "aa"]
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

rule rsw_correlations:
    """RSW covariance/correlation corner plot for each pole model's initial state estimate.

    The plotter rotates the covariance into the RSW frame, so it rebuilds the
    system of bodies.  The stage config composes the estimation config and the
    rule overrides the pole model, so the frame is the one the fit used.
    """

    input:
        python_dependencies("scripts/plot_estimation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_rsw_correlations.yaml"),
        # The figure config is named by a plain key rather than a defaults
        # entry, so hydra_dependencies() above does not see it.
        lambda wildcards: f"conf/chapter_3/fig_rsw_correlations_{wildcards.pole_model}.yaml",
        estimation_output=f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimation_output.tudat",

    output:
        f"{INITIAL_ESTIMATION_FOLDER}/rsw_correlations_{{pole_model}}.pdf",

    wildcard_constraints:
        pole_model="|".join(POLE_MODELS),

    log:
        f"{INITIAL_ESTIMATION_FOLDER}/rsw_correlations_{{pole_model}}.log",

    shell:
        """
        {PYTHON} scripts/plot_estimation_figure.py \
            --config-name chapter_3/fig_rsw_correlations \
            hydra.run.dir={INITIAL_ESTIMATION_FOLDER} \
            figure_config=conf/chapter_3/fig_rsw_correlations_{wildcards.pole_model}.yaml \
            estimation_output={input.estimation_output} \
            bodies_to_create.Neptune.rotation_model.type={wildcards.pole_model} \
            > {log} 2>&1
        """


rule parameter_change_table:
    """Initial vs final parameter table for each pole model's initial state estimate."""

    input:
        python_dependencies("scripts/write_estimation_table.py"),
        hydra_dependencies("conf/write_estimation_table.yaml"),
        # The table config is named by a plain key rather than a defaults entry,
        # so hydra_dependencies() above does not see it.
        "conf/tables/parameter_change.yaml",
        estimation_output=f"{INITIAL_ESTIMATION_FOLDER}/{{pole_model}}/estimation_output.tudat",

    output:
        # One LaTeX file per style the table config names (booktabs and aa).
        f"{INITIAL_ESTIMATION_FOLDER}/parameter_change_{{pole_model}}_booktabs.tex",
        f"{INITIAL_ESTIMATION_FOLDER}/parameter_change_{{pole_model}}_aa.tex",

    wildcard_constraints:
        pole_model="|".join(POLE_MODELS),

    log:
        f"{INITIAL_ESTIMATION_FOLDER}/parameter_change_{{pole_model}}.log",

    shell:
        """
        {PYTHON} scripts/write_estimation_table.py \
            hydra.run.dir={INITIAL_ESTIMATION_FOLDER} \
            table_config=conf/tables/parameter_change.yaml \
            estimation_output={input.estimation_output} \
            output_name=parameter_change_{wildcards.pole_model} \
            > {log} 2>&1
        """


######################################################
############## Gravitational parameter sweep #########
######################################################
GM_SWEEP_FOLDER = f"{STAGE}/gm_sweep"

#: Parameter sets the GM sweep estimates.  Each is a config group under
#: ``conf/parameters/``; the rule passes the wildcard value as ``parameters=``.
#: All three include the initial state, so the fits differ only in which
#: gravitational parameters are added to it.
GM_SETS = [
    "state_plus_neptune_GM",
    "state_plus_triton_GM",
    "state_gm_nep_gm_tri",
]





rule estimate_gm_sweep:
    """Fit the initial state with each swept set of gravitational parameters.

    The sweep asks how well the synthetic observations constrain the
    gravitational parameters, and whether adding Triton's GM to the fit changes
    Neptune's.  Each combination is a separate fit, so the enumeration lives
    here rather than in a Hydra multirun: every fit stays an ordinary rule with
    its own artifacts, and a single combination can be re-run on its own.

    The parameter set is the only thing that varies.  Dynamics, kernels, epochs
    and dependent variables are inherited from the initial state estimation
    config, so a GM fit differs from the state-only fit in exactly one thing.
    """

    input:
        python_dependencies("scripts/solve_estimation.py"),
        # The parameter set is selected by a group override, so the dependency
        # list has to be composed with that override: without it the base
        # config's own ``parameters`` default would be tracked instead, and
        # editing the swept parameter file would not re-run the fit.
        lambda wildcards: hydra_dependencies(
            "conf/chapter_3/estimation_gm_sweep.yaml",
            overrides=[f"parameters={wildcards.gm_set}"],
        ),
        f"{INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",

    output:
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimation_output.tudat",
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/final_iteration_simulation_results.tudat",
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimation_log.log",

    wildcard_constraints:
        gm_set="|".join(GM_SETS),

    log:
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimate.log",

    shell:
        """
        {PYTHON} scripts/solve_estimation.py \
            --config-name chapter_3/estimation_gm_sweep \
            hydra.run.dir={GM_SWEEP_FOLDER}/{wildcards.gm_set} \
            observations={INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat \
            parameters={wildcards.gm_set} \
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
        # Paragraph 2: one parameter-change table per pole model, in both styles
        expand(
            f"{INITIAL_ESTIMATION_FOLDER}/parameter_change_{{pole_model}}_{{style}}.tex",
            pole_model=POLE_MODELS,
            style=TABLE_STYLES,
        ),
        # Paragraph 3: one fit per swept set of gravitational parameters
        expand(
            f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimation_output.tudat",
            gm_set=GM_SETS,
        ),
