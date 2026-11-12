"""Toy pre-fit residual pipeline: collection -> residuals -> figures.

This is the smallest end-to-end path through the pipeline, and it exists to make
every stage debuggable in seconds: one NSDB dataset (nm0017, 54 photographic
observations of Triton, 1986-1993), the light pre-fit dynamical model, and the
classical kernel set.

The two stages are deliberately separate rules.  The collection stage is the
expensive one and depends only on the dataset selection; the residual stage
depends on the collection and on which ephemeris the residuals are taken
against.  Splitting them means a change of residual source - SPICE to a
propagated state, i.e. pre-fit to post-fit - re-runs only the cheap stage.

Build everything::

    snakemake --snakefile workflow/Snakefile all_prefit_residuals_toy --cores 4

Build up to one stage::

    snakemake --snakefile workflow/Snakefile --until prefit_residuals --cores 4
"""

from orbitdet.reproducibility import hydra_dependencies, python_dependencies

STAGE = f"{RESULTS_ROOT}/prefit_residuals_toy"

COLLECTION_FOLDER = f"{STAGE}/generate_observation_collection"
RESIDUALS_FOLDER = f"{STAGE}/prefit_residuals"

COLLECTION = f"{COLLECTION_FOLDER}/observations.tudat"
STANDARD_FRAME = f"{COLLECTION_FOLDER}/standard_frame.parquet"

RESIDUALS = f"{RESIDUALS_FOLDER}/observations.tudat"
RESIDUALS_FRAME = f"{RESIDUALS_FOLDER}/standard_frame.parquet"

# The figures this pipeline draws, as (figure config stem, output name) pairs.
FIGURES = [
    ("prefit_residuals_spice", "prefit_residuals_spice.pdf"),
    ("residual_scatter", "residual_scatter.pdf"),
    ("residual_histogram", "residual_histogram.pdf"),
]


rule generate_observation_collection:
    """Build the observation collection and its standard frame from a dataset config."""

    input:
        python_dependencies("scripts/generate_observation_collection.py"),
        hydra_dependencies("conf/generate_observation_collection.yaml"),

    output:
        COLLECTION,
        STANDARD_FRAME,

    log:
        f"{COLLECTION_FOLDER}/generate_observation_collection.log",

    shell:
        """
        {PYTHON} scripts/generate_observation_collection.py \
            hydra.run.dir={COLLECTION_FOLDER} \
            > {log} 2>&1
        """


rule prefit_residuals:
    """Compute the residuals of a collection against the SPICE ephemeris.

    The collection is an input rather than something this rule builds, so the
    same rule serves the post-fit case: point ``observation_source`` at a
    propagation result and the residuals become those of the propagated state.
    """

    input:
        python_dependencies("scripts/prefit_residuals.py"),
        hydra_dependencies("conf/prefit_residuals.yaml"),
        observations=COLLECTION,
        standard_frame=STANDARD_FRAME,

    output:
        RESIDUALS,
        RESIDUALS_FRAME,

    log:
        f"{RESIDUALS_FOLDER}/prefit_residuals.log",

    shell:
        """
        {PYTHON} scripts/prefit_residuals.py \
            observations={input.observations} \
            standard_frame={input.standard_frame} \
            hydra.run.dir={RESIDUALS_FOLDER} \
            > {log} 2>&1
        """


rule prefit_residuals_figure:
    """Draw one residual figure from the residual stage's artifacts."""

    input:
        python_dependencies("scripts/plot_residuals_figure.py"),
        hydra_dependencies("conf/plot_residuals_figure.yaml"),
        observations=RESIDUALS,
        standard_frame=RESIDUALS_FRAME,

    output:
        f"{RESIDUALS_FOLDER}/{{figure}}.pdf",

    wildcard_constraints:
        figure="|".join(stem for stem, _ in FIGURES),

    log:
        f"{RESIDUALS_FOLDER}/{{figure}}.log",

    shell:
        """
        {PYTHON} scripts/plot_residuals_figure.py \
            figure_config=conf/figures/{wildcards.figure}.yaml \
            observations={input.observations} \
            standard_frame={input.standard_frame} \
            hydra.run.dir={RESIDUALS_FOLDER} \
            > {log} 2>&1
        """


rule all_prefit_residuals_toy:
    """The toy pipeline's artifacts and every figure it draws."""

    input:
        RESIDUALS,
        RESIDUALS_FRAME,
        expand(
            f"{RESIDUALS_FOLDER}/{{figure}}",
            figure=[name for _, name in FIGURES],
        ),
