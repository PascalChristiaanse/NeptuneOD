Per-source observation weights. One row per observatory, listing how many
observations it contributed and the weight applied to them. Rows are the
configured weight groups, so a source that is weighted differently from its
neighbours — or excluded — is visible here rather than only in the config.

When an experiment disables weighting the table still exists, with headers and
no rows: that states weighting was *off*, which is different from the stage
having been skipped.
