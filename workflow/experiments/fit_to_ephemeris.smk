# Snakemake workflow: fit_to_ephemeris
#
# Experiment: fit the Jacobsen (2009) Neptune model to the simulated 3D-point
# dataset rather than a real observation archive. This is the "can the pipeline
# recover a known truth" experiment, so it is cheap and self-contained.
#
# DAG:
#
#   resolve_config -> kernels -> observations -> ephemeris_residuals
#     -> outlier_rejection -> weighting -> prefit_propagation -> estimation
#     -> postfit_propagation -> covariance
#     -> figures / tables
#
# Run:
#
#   snakemake -s workflow/Snakefile --cores 4                  # both experiments
#   snakemake -s workflow/Snakefile fit_to_ephemeris --cores 1 # this one only
#   snakemake -s workflow/Snakefile --report report.html       # final report
#
# Paths are written out literally rather than with an ``{experiment}``
# wildcard: each experiment file targets exactly one experiment, and two rules
# matching the same wildcard pattern would be ambiguous.

EXPERIMENT = "fit_to_ephemeris"

# Figures this experiment draws. Each must have a styling block in
# conf/figures/, otherwise it renders with hardcoded defaults.
#
# The scan-frame variants are deliberately absent: the simulated dataset has no
# position_angle_scan, so a scan-frame residual would be meaningless.
FIGURES = [
    "prefit_residuals_spice",
    "prefit_normalized_residuals",
    "prefit_residuals_prop",
    "postfit_residuals",
    "postfit_normalized_residuals",
    "state_history_3d",
    "prefit_rsw",
    "postfit_rsw",
    "parameter_correlation_heatmap",
    "parameter_history_per_iteration",
    "propagated_formal_errors_cartesian",
    "propagated_formal_errors_rsw",
    "RSW_distance_with_uncertainty",
]

_R = f"results/{EXPERIMENT}"


rule fit_to_ephemeris:
    """Every output this experiment produces.

    The listing is explicit so the manifest -- and therefore the report -- is
    readable: it states exactly which artifacts this experiment is expected to
    produce.

    No ``name:`` directive: it would rename the rule, and the rule name is what
    makes ``snakemake fit_to_ephemeris`` work as a target.
    """
    default_target: True
    input:
        # Compute chain.
        f"{_R}/config/estimation.yaml",
        f"{_R}/prefit_ephemeris/residuals.csv",
        f"{_R}/preprocessing/outlier_rejection_metadata.json",
        f"{_R}/preprocessing/observation_weights.csv",
        f"{_R}/estimation/estimation_output.tudat",
        f"{_R}/estimation/estimation_summary.json",
        f"{_R}/propagation/postfit_residuals.csv",
        f"{_R}/covariance/propagated_formal_errors.npz",
        # Presentational outputs.
        f"{_R}/tables/weight_summary_source.csv",
        f"{_R}/report.md",
        expand(f"{_R}/figures/{{figure}}.pdf", figure=FIGURES),
    output:
        # A stamp so ``rule all`` can depend on "this experiment completed"
        # without duplicating the file list above.
        touch(f"{_R}/.complete")
