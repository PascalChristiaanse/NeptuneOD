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

#: Display name per estimated parameter, in parameter-vector order, for each GM
#: set.  TudatPy orders the blocks initial state first, then the scalar
#: parameters in the order the parameter config enables them, so the state names
#: come first and the gravitational parameters follow.  The parameter-change
#: table needs these because its vector is longer than the state-only fit's.
GM_PARAMETER_NAMES = {
    "state_plus_neptune_GM": ["x", "y", "z", "v_x", "v_y", "v_z", "mu_N"],
    "state_plus_triton_GM": ["x", "y", "z", "v_x", "v_y", "v_z", "mu_T"],
    "state_gm_nep_gm_tri": ["x", "y", "z", "v_x", "v_y", "v_z", "mu_N", "mu_T"],
}


def _gm_fit_paths() -> dict[str, str]:
    """Return ``slug -> estimation_output`` for the baseline and every GM fit.

    The baseline is the state-only fit from the initial state estimation
    section, so the comparison table shows what estimating a GM buys over not
    estimating one.
    """
    paths = {
        "initial_state": (
            f"{INITIAL_ESTIMATION_FOLDER}/{POLE_MODELS[0]}/estimation_output.tudat"
        )
    }
    for gm_set in GM_SETS:
        paths[gm_set] = f"{GM_SWEEP_FOLDER}/{gm_set}/estimation_output.tudat"
    return paths


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


rule gm_parameter_change_table:
    """Initial vs final parameter table for each GM-sweep fit.

    The parameter vector is longer than the state-only fit's, so the extra rows
    need naming.  The names come from :data:`GM_PARAMETER_NAMES` rather than the
    table config, because they depend on which parameters the set estimates and
    the rule is the only place that knows the set.
    """

    input:
        python_dependencies("scripts/write_estimation_table.py"),
        hydra_dependencies("conf/write_estimation_table.yaml"),
        "conf/tables/gm_parameter_change.yaml",
        estimation_output=f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimation_output.tudat",

    output:
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/parameter_change_booktabs.tex",
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/parameter_change_aa.tex",

    wildcard_constraints:
        gm_set="|".join(GM_SETS),

    params:
        params=lambda wildcards: ",".join(GM_PARAMETER_NAMES[wildcards.gm_set]),

    log:
        f"{GM_SWEEP_FOLDER}/{{gm_set}}/parameter_change.log",

    shell:
        """
        {PYTHON} scripts/write_estimation_table.py \
            hydra.run.dir={GM_SWEEP_FOLDER}/{wildcards.gm_set} \
            table_config=conf/tables/gm_parameter_change.yaml \
            estimation_output={input.estimation_output} \
            "+parameter_names=[{params}]" \
            > {log} 2>&1
        """


rule gm_goodness_of_fit_table:
    """Residual RMS and per-RSW-component maximum error for every GM-sweep fit.

    One table comparing all the fits, so the rows are the fits rather than the
    parameters.  The initial state fit from the previous section is included as
    the baseline the GM fits are measured against.
    """

    input:
        python_dependencies("scripts/write_estimation_table.py"),
        hydra_dependencies("conf/write_estimation_table.yaml"),
        "conf/tables/gm_goodness_of_fit.yaml",
        observations=f"{INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
        estimation_outputs=expand(
            f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimation_output.tudat",
            gm_set=GM_SETS,
        ),

    output:
        f"{GM_SWEEP_FOLDER}/goodness_of_fit_booktabs.tex",
        f"{GM_SWEEP_FOLDER}/goodness_of_fit_aa.tex",

    params:
        # One ``label=path`` override per fit, in the row order the table shows.
        # The baseline fit comes first so every GM fit is read against it.
        outputs=[
            f"+estimation_outputs.{slug}={path}"
            for slug, path in _gm_fit_paths().items()
        ],

    log:
        f"{GM_SWEEP_FOLDER}/goodness_of_fit.log",

    shell:
        """
        {PYTHON} scripts/write_estimation_table.py \
            hydra.run.dir={GM_SWEEP_FOLDER} \
            table_config=conf/tables/gm_goodness_of_fit.yaml \
            observations={input.observations} \
            {params.outputs} \
            > {log} 2>&1
        """


rule rsw_gm_sweep:
    """RSW separation of every GM-sweep fit, with the state-only fit as baseline.

    The reference is the SPICE ephemeris, so each curve is how far that fit's
    propagated Triton departs from SPICE.  Overlaying the state-only fit shows
    whether estimating a GM improves the trajectory or trades one error for
    another.
    """

    input:
        python_dependencies("scripts/plot_simulation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_rsw_gm_sweep.yaml"),
        simulation_result=f"{INITIAL_ESTIMATION_FOLDER}/{POLE_MODELS[0]}/final_iteration_simulation_results.tudat",
        simulation_results=expand(
            f"{GM_SWEEP_FOLDER}/{{gm_set}}/final_iteration_simulation_results.tudat",
            gm_set=GM_SETS,
        ),

    output:
        f"{GM_SWEEP_FOLDER}/rsw_gm_sweep.pdf",

    params:
        results=",".join(
            [f"{INITIAL_ESTIMATION_FOLDER}/{POLE_MODELS[0]}/final_iteration_simulation_results.tudat"]
            + [
                f"{GM_SWEEP_FOLDER}/{gm_set}/final_iteration_simulation_results.tudat"
                for gm_set in GM_SETS
            ]
        ),

    log:
        f"{GM_SWEEP_FOLDER}/rsw_gm_sweep.log",

    shell:
        """
        {PYTHON} scripts/plot_simulation_figure.py \
            hydra.run.dir={GM_SWEEP_FOLDER} \
            figure_config=conf/chapter_3/fig_rsw_gm_sweep.yaml \
            simulation_result={input.simulation_result} \
            +simulation_results=[{params.results}] \
            > {log} 2>&1
        """


rule rsw_correlations_gm_sweep:
    """RSW covariance/correlation corner plot for each GM-sweep fit.

    The plotter rotates the covariance into the RSW frame, so it rebuilds the
    system of bodies.  The stage config composes the GM-sweep estimation config
    and the rule overrides the parameter set, so the frame is the one the fit
    used.
    """

    input:
        python_dependencies("scripts/plot_estimation_figure.py"),
        hydra_dependencies("conf/chapter_3/fig_rsw_correlations_gm_sweep.yaml"),
        # The figure config is named by a plain key rather than a defaults
        # entry, so hydra_dependencies() above does not see it.
        lambda wildcards: f"conf/chapter_3/fig_rsw_correlations_gm_sweep_{wildcards.gm_set}.yaml",
        estimation_output=f"{GM_SWEEP_FOLDER}/{{gm_set}}/estimation_output.tudat",

    output:
        f"{GM_SWEEP_FOLDER}/rsw_correlations_{{gm_set}}.pdf",

    wildcard_constraints:
        gm_set="|".join(GM_SETS),

    log:
        f"{GM_SWEEP_FOLDER}/rsw_correlations_{{gm_set}}.log",

    shell:
        """
        {PYTHON} scripts/plot_estimation_figure.py \
            --config-name chapter_3/fig_rsw_correlations_gm_sweep \
            hydra.run.dir={GM_SWEEP_FOLDER} \
            figure_config=conf/chapter_3/fig_rsw_correlations_gm_sweep_{wildcards.gm_set}.yaml \
            estimation_output={input.estimation_output} \
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
        # Paragraph 3: one parameter-change table per GM fit, in both styles
        expand(
            f"{GM_SWEEP_FOLDER}/{{gm_set}}/parameter_change_{{style}}.tex",
            gm_set=GM_SETS,
            style=TABLE_STYLES,
        ),
        # Paragraph 3: one goodness-of-fit table comparing every fit
        expand(
            f"{GM_SWEEP_FOLDER}/goodness_of_fit_{{style}}.tex",
            style=TABLE_STYLES,
        ),
        # Paragraph 3: RSW separation of every fit, and one corner plot each
        f"{GM_SWEEP_FOLDER}/rsw_gm_sweep.pdf",
        expand(
            f"{GM_SWEEP_FOLDER}/rsw_correlations_{{gm_set}}.pdf",
            gm_set=GM_SETS,
        ),
