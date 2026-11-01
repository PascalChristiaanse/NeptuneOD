# Snakemake workflow: classic_triton_state
#
# Experiment: the full Triton state estimation, using the combined
# Jacobson (2009) + Voyager + Gaia observation archive, with outlier rejection
# and per-source weighting. This is the production experiment.
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
#   snakemake -s workflow/Snakefile --cores 4                    # both experiments
#   snakemake -s workflow/Snakefile classic_triton_state --cores 4
#   snakemake -s workflow/Snakefile --report report.html         # final report
#
# Paths are written out literally rather than with an ``{experiment}``
# wildcard: each experiment file targets exactly one experiment, and two rules
# matching the same wildcard pattern would be ambiguous.

EXPERIMENT = "classic_triton_state"

# Figures this experiment draws. The Gaia scan-frame variants are included
# because this archive carries position_angle_scan; the simulated dataset does
# not, and a scan-frame residual would be meaningless there.
FIGURES = [
    "prefit_residuals_spice",
    "prefit_residuals_spice_scan",
    "prefit_normalized_residuals",
    "prefit_normalized_residuals_scan",
    "prefit_residuals_prop",
    "prefit_residuals_prop_scan",
    "postfit_residuals",
    "postfit_residuals_scan",
    "postfit_normalized_residuals",
    "postfit_normalized_residuals_scan",
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


rule classic_triton_state:
    """Every output this experiment produces.

    The listing is explicit so the manifest -- and therefore the report -- is
    readable: it states exactly which artifacts this experiment is expected to
    produce.

    No ``name:`` directive: it would rename the rule, and the rule name is what
    makes ``snakemake classic_triton_state`` work as a target.
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
