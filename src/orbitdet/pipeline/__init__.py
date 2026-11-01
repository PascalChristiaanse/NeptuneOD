"""File-based pipeline stages for Snakemake-driven execution.

Each stage is a pure function of files: it loads its inputs from paths named by
the caller and returns the paths it wrote. Stages never read ``HydraConfig``,
never call ``plt.show()``, and never reach for a global runtime context -- the
caller (Snakemake, or a test) owns every path.

Why this module exists
----------------------
``scripts/solve_least_squares.py`` ran the whole orbit determination in one
process: build environment, build observations, reject outliers, weight,
propagate, estimate, propagate covariance, plot. Any change -- a figure title, a
weighting strategy, a convergence tolerance -- re-ran all of it.

The stage library is the same computation split at **file boundaries**. Between
two stages there is always an artifact on disk that tudatpy can serialize
natively, so stages can run in separate processes and be invalidated
independently. The artifacts are:

``kernels.done``
    Stamp recording that the SPICE kernel download completed.
``observations.tudat`` + ``dataset_metadata.json``
    The ``ObservationCollection`` plus the metadata that is not recoverable from
    it (``dataset_id`` per set, coordinate labels).
``outliers_applied.tudat``
    The collection after outlier rejection, with a rejection report.
``weighted.tudat``
    The collection with weights applied.
``prefit_residuals/*.csv``
    Residuals flattened to a tidy table, so figures are pure functions of a file.
``estimation_output.tudat``
    The full ``EstimationOutput``, including the covariance-propagation payload.

Design rules
------------
* **No stage imports another stage.** Stages compose by consuming artifacts, not
  by calling each other, so the DAG in the Snakefile is the single description
  of execution order.
* **Every stage re-furnishes SPICE kernels.** Snakemake runs each rule in its own
  process, so the SPICE pool is empty at the start of each one.
* **Config arrives as a file.** Each stage reads a resolved config slice that
  Snakemake tracked as an input, never the ``conf/`` tree directly.
"""
