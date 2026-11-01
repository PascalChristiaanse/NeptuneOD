One metric per combination across the whole sweep. Each row is a Hydra
combination: the varied parameters are expanded into their own columns, so a
metric can be read against the settings that produced it. The ``overrides``
column carries the combination's raw override string and ``combo`` matches its
directory under ``sweeps/<sweep>/``, so any row can be traced back to the
configs, estimation output and covariance that generated it.

Produces: ``residuals_rms``, ``num_iterations``, ``estimation_duration_s``,
``num_parameters``, ``covariance_is_finite``, ``formal_errors_are_finite`` and
``max_position_sigma_m``. The two ``*_is_finite`` flags are what make a
degenerate combination identifiable without opening its covariance file.
