Orbit determination for the Neptune system, as a Snakemake workflow.

Overview
========

The report is generated from the workflow's own provenance metadata, so every
figure, table and summary here is a file the pipeline actually produced -- the
run times, rule parameters and wildcards shown for each result come from the job
that wrote it, not from a separate logging call.

Structure
=========

Each experiment is a self-contained analysis sharing one set of stage rules. The
stages are file boundaries: ``observations.tudat``, ``weighted.tudat``,
``estimation_output.tudat`` and each propagation result are serialized tudatpy
objects, so a stage can be re-run without recomputing its upstream.
``docs/snakemake_design.md`` documents the boundaries and the serialization each
one relies on.

Categories
==========

The per-experiment results are ``residuals``, ``estimation``, ``trajectory``,
``rsw``, ``rsw_uncertainty`` and ``formal_errors``; ``Tables`` holds the weight
summary, ``Summary`` the assembled report, and ``Sweep comparison`` the
cross-combination telemetry and bar charts.

Reading a result
================

Clicking a result shows its caption, the rule that produced it, and its
wildcards and parameters. ``Statistics`` reports run times per rule, and
``Workflow`` shows the rule graph.
