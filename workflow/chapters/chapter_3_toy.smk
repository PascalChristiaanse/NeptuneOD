"""Toy parameter sweep: the chapter 3 sweep rules on a seconds-long dataset.

This chapter exists to check the sweep wiring before committing to the real
run.  It reuses the chapter 3 rules verbatim through ``use rule ... as ... with:``
and only swaps the configs and the output folder, so a toy run exercises the
same shell commands, the same Hydra overrides and the same analysis and figure
stages as the real sweep.  Nothing here is a reimplementation.

What is reduced, and why it is still a valid check:

* the observations are four simulated bursts over one year instead of the
  synthetic 1963-2026 collection, so the collection stage takes seconds;
* the estimation is capped at two iterations, so a fit takes seconds;
* the dynamics, kernels and dependent variables are the real ones, so the code
  path a toy fit takes is the code path a real fit takes.

Build the whole toy sweep::

    snakemake --snakefile workflow/Snakefile all_chapter_3_toy --cores 4

Build one group only::

    snakemake --snakefile workflow/Snakefile \
        results/chapter_3_toy/parameter_sweeps/gravity/specification_curve_gravity.pdf \
        --cores 4

The toy writes under ``results/chapter_3_toy``, so it never touches the real
sweep's artifacts.
"""

STAGE_TOY = f"{RESULTS_ROOT}/chapter_3_toy"

TOY_INITIAL_ESTIMATION_FOLDER = f"{STAGE_TOY}/initial_state_estimation"
TOY_PARAMETER_SWEEP_FOLDER = f"{STAGE_TOY}/parameter_sweeps"

TOY_COLLECTION_CONFIG = "chapter_3/toy_coll_synthetic"
TOY_ESTIMATION_CONFIG = "chapter_3/toy_estimation_sweep"


rule all_chapter_3_toy:
    """The toy sweep's artifacts: every fit, both tables and both figures."""

    input:
        # The collection every fit consumes.
        f"{TOY_INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
        # Every specification of both groups, plus the per-group table and
        # figure.  Expanded per group because each group has its own indices.
        [
            f"{TOY_PARAMETER_SWEEP_FOLDER}/{group}/{index}/estimation_output.tudat"
            for group in SWEEP_GROUPS
            for index in SWEEP_PARAMSPACES[group].instance_patterns
        ],
        expand(
            f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_group}}_sweep_comparison_{{style}}.tex",
            sweep_group=SWEEP_GROUPS,
            style=TABLE_STYLES,
        ),
        expand(
            f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/specification_curve_{{sweep_group}}.pdf",
            sweep_group=SWEEP_GROUPS,
        ),


# -- Reused chapter 3 rules -------------------------------------------------
#
# Each ``use rule`` below overrides only what makes the rule toy-sized: the
# config name, the output folder and the observation path.  The shell command,
# the wildcard constraints and the Hydra override logic stay the chapter 3 ones,
# so a toy run cannot pass while the real rule is broken.

use rule generate_synthetic_observations as generate_toy_observations with:
    input:
        python_dependencies("scripts/generate_observation_collection.py"),
        hydra_dependencies(f"conf/{TOY_COLLECTION_CONFIG}.yaml"),
    output:
        f"{TOY_INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
        f"{TOY_INITIAL_ESTIMATION_FOLDER}/prefit_standard_frame.parquet",
    params:
        config_name=TOY_COLLECTION_CONFIG,
        folder=TOY_INITIAL_ESTIMATION_FOLDER,
        observations="prefit_observations",
        standard_frame="prefit_standard_frame.parquet",
    log:
        f"{TOY_INITIAL_ESTIMATION_FOLDER}/generate_synthetic_observations.log",


use rule estimate_parameter_sweep as estimate_toy_parameter_sweep with:
    input:
        python_dependencies("scripts/solve_estimation.py"),
        hydra_dependencies(f"conf/{TOY_ESTIMATION_CONFIG}.yaml"),
        f"{TOY_INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
    output:
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_spec}}/estimation_output.tudat",
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_spec}}/estimation_log.log",
    params:
        override=lambda wildcards: (
            "+estimation.parameters_to_estimate={"
            + ",".join(
                f"{key}:{str(value).lower()}"
                for key, value in {
                    "initial_state": True,
                    **SWEEP_PARAMSPACES[wildcards.sweep_group].instance(wildcards),
                }.items()
            )
            + "}"
        ),
        config_name=TOY_ESTIMATION_CONFIG,
        folder=TOY_PARAMETER_SWEEP_FOLDER,
        observations=f"{TOY_INITIAL_ESTIMATION_FOLDER}/prefit_observations.tudat",
    log:
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_spec}}/estimate.log",


use rule analyse_parameter_sweep as analyse_toy_parameter_sweep with:
    input:
        python_dependencies("scripts/analyse_parameter_sweep.py"),
        hydra_dependencies("conf/analyse_parameter_sweep.yaml"),
        "conf/tables/sweep_comparison.yaml",
        estimation_outputs=lambda wildcards: expand(
            f"{TOY_PARAMETER_SWEEP_FOLDER}/{wildcards.sweep_group}/{{sweep_spec}}/estimation_output.tudat",
            sweep_spec=list(SWEEP_PARAMSPACES[wildcards.sweep_group].instance_patterns),
        ),
        estimation_logs=lambda wildcards: expand(
            f"{TOY_PARAMETER_SWEEP_FOLDER}/{wildcards.sweep_group}/{{sweep_spec}}/estimation_log.log",
            sweep_spec=list(SWEEP_PARAMSPACES[wildcards.sweep_group].instance_patterns),
        ),
    output:
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_group}}_sweep_comparison.csv",
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_group}}_sweep_comparison.md",
        [
            f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_group}}_sweep_comparison_{style}.tex"
            for style in TABLE_STYLES
        ],
    params:
        folder=TOY_PARAMETER_SWEEP_FOLDER,
        table_config="conf/tables/sweep_comparison.yaml",
        sweep_csv=lambda wildcards: f"conf/sweeps/{wildcards.sweep_group}.csv",
    log:
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/analyse.log",


use rule plot_parameter_sweep as plot_toy_parameter_sweep with:
    input:
        python_dependencies("scripts/plot_sweep_figure.py"),
        hydra_dependencies("conf/plot_sweep_figure.yaml"),
        lambda wildcards: f"conf/figures/specification_curve_{wildcards.sweep_group}.yaml",
        sweep_table=f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/{{sweep_group}}_sweep_comparison.csv",
    output:
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/specification_curve_{{sweep_group}}.pdf",
    params:
        folder=TOY_PARAMETER_SWEEP_FOLDER,
        figure_config=lambda wildcards: (
            f"conf/figures/specification_curve_{wildcards.sweep_group}.yaml"
        ),
    log:
        f"{TOY_PARAMETER_SWEEP_FOLDER}/{{sweep_group}}/specification_curve.log",
