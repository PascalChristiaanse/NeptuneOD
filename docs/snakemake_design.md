# Snakemake and Orbitdet Project Architecture
## Technical Handover Document

**Status:** Implemented (see §14 for what was built and where it differs)  
**Purpose:** Establish a maintainable, reproducible framework for conducting multiple scientific analyses using Snakemake and the custom Python library `orbitdet`.

---

## 1. Project objectives

The project involves conducting several scientific analyses, including reproductions of published papers, for example:

- Jacobson (2009)
- Wang et al. (2025)

The analyses may share datasets, preprocessing steps, computational utilities, and visualisation functionality. However, each paper may require different methods, parameters, configurations, and outputs.

The architecture should therefore:

1. Avoid duplicating shared computational code.
2. Keep individual analyses logically independent.
3. Track Python implementation dependencies at a sufficiently granular level.
4. Track Hydra configuration dependencies, including composed configurations.
5. Ensure that changes only trigger the relevant Snakemake rules and downstream outputs.
6. Preserve the ability to reproduce each analysis independently.

---

## 2. Recommended high-level architecture

Use **one project-level Snakefile with separate analysis subworkflows**.

Avoid creating a completely independent Snakefile for every paper unless the analyses eventually require separate execution environments or independent project lifecycles.

Proposed structure:

```text
project/
│
├── Snakefile
│
├── config/
│   ├── common.yaml
│   ├── jacobson_2009.yaml
│   └── wang_2025.yaml
│
├── workflows/
│   ├── common/
│   │   ├── preprocessing.smk
│   │   └── data_checks.smk
│   │
│   ├── jacobson_2009/
│   │   ├── analysis.smk
│   │   └── config.yaml
│   │
│   ├── wang_2025/
│   │   ├── analysis.smk
│   │   └── config.yaml
│   │
│   └── dependency_tracking/
│       ├── python_graph.py
│       ├── hydra_graph.py
│       └── manifest.py
│
├── orbitdet/
│   ├── preprocessing/
│   ├── analysis/
│   │   ├── jacobson.py
│   │   └── wang.py
│   ├── visualization/
│   │   ├── common.py
│   │   ├── jacobson_plots.py
│   │   └── wang_plots.py
│   └── ...
│
├── scripts/
│   ├── run_jacobson.py
│   └── run_wang.py
│
├── data/
│
└── results/
    ├── jacobson_2009/
    └── wang_2025/
```

This is a proposed structure, not a requirement to create every directory immediately.

---

## 3. Main Snakefile

The project-level Snakefile assembles the different subworkflows.

```python
# Snakefile

configfile: "config/common.yaml"

include: "workflows/common/preprocessing.smk"

include: "workflows/jacobson_2009/analysis.smk"

include: "workflows/wang_2025/analysis.smk"
```

Snakemake treats included rule files as part of the same workflow.

This provides a single entry point while keeping individual analyses organised separately.

### Benefits

- Shared rules can be defined once.
- Each paper has its own analysis rules.
- Outputs can be separated by paper.
- Shared intermediate results can be reused.
- Analyses can be selected independently through their requested targets.

---

## 4. Analysis-specific workflows

Each paper should have its own `.smk` file.

For example:

```python
# workflows/jacobson_2009/analysis.smk

rule jacobson_analysis:
    input:
        "data/processed/{dataset}.parquet"
    output:
        "results/jacobson_2009/{dataset}/analysis.parquet"
    script:
        "scripts/run_jacobson.py"
```

And:

```python
# workflows/wang_2025/analysis.smk

rule wang_analysis:
    input:
        "data/processed/{dataset}.parquet"
    output:
        "results/wang_2025/{dataset}/analysis.parquet"
    script:
        "scripts/run_wang.py"
```

These are illustrative examples. Actual rules should explicitly declare their Python and configuration dependencies.

### Important principle

The analyses should remain methodologically independent.

If Jacobson (2009) and Wang et al. (2025) use different statistical methods or preprocessing assumptions, those differences should be preserved in their respective implementations and configurations.

Shared software infrastructure should not imply shared scientific assumptions.

---

## 5. Shared Python library: `orbitdet`

The `orbitdet` library should contain reusable computational functionality.

For example:

```text
orbitdet/
├── preprocessing/
│   ├── loading.py
│   └── filtering.py
│
├── analysis/
│   ├── jacobson.py
│   └── wang.py
│
└── visualization/
    ├── common.py
    ├── jacobson_plots.py
    └── wang_plots.py
```

Shared functions should be implemented once where appropriate.

Paper-specific functionality should remain in separate modules when the underlying methods differ.

For example:

- Common plotting utilities can live in `visualization/common.py`.
- Jacobson-specific plots can live in `visualization/jacobson_plots.py`.
- Wang-specific plots can live in `visualization/wang_plots.py`.

This organisation supports code reuse without unnecessarily coupling analyses.

---

## 6. Fine-grained Python dependency tracking

### Problem

Snakemake does not necessarily detect changes to imported Python modules in a custom library when those modules are not explicitly declared as rule inputs.

For example, a rule might execute:

```python
from orbitdet.visualization import plotting
```

If `plotting.py` changes, Snakemake needs a way to recognise that the corresponding output may be outdated.

A simple solution would be to declare the entire `orbitdet` package as an input dependency. However, this is unnecessarily coarse.

A change to an unrelated module could trigger many rules.

### Recommended approach

Use Python import dependency analysis to determine which modules are required by each rule.

For example:

```text
plotting rule
    ├── orbitdet.visualization.plotting
    ├── orbitdet.visualization.common
    └── orbitdet.utils.statistics
```

Only those source files should be declared as dependencies of the plotting rule.

A possible implementation uses the Python package `grimp` to construct an import graph.

Conceptual implementation:

```python
from pathlib import Path
from functools import lru_cache
import importlib.util
import grimp

PACKAGE = "orbitdet"


@lru_cache(maxsize=1)
def get_import_graph():
    return grimp.build_graph(PACKAGE)


def module_to_file(module):
    spec = importlib.util.find_spec(module)

    if spec is None or spec.origin is None:
        return None

    path = Path(spec.origin).resolve()

    if path.suffix != ".py" or not path.is_file():
        return None

    return path


@lru_cache(maxsize=None)
def implementation_files(entrypoint):
    graph = get_import_graph()

    modules = {entrypoint}
    modules.update(graph.find_upstream_modules(entrypoint))

    modules = {
        module
        for module in modules
        if module == PACKAGE or module.startswith(f"{PACKAGE}.")
    }

    files = {
        path
        for module in modules
        if (path := module_to_file(module)) is not None
    }

    return tuple(sorted(str(path) for path in files))
```

A Snakemake input function could then return the relevant implementation files.

This is a starting point, not a complete production implementation. In particular, the import graph must be checked against the installed `orbitdet` package structure.

### Limitations

Static import analysis may not detect:

- Dynamic imports.
- Runtime-loaded modules.
- Files accessed through arbitrary paths.
- Templates and other non-Python resources.
- Compiled or native dependencies.

These should be declared explicitly or handled through additional dependency tracking.

---

## 7. Hydra configuration dependency tracking

### Problem

`orbitdet` also uses Hydra YAML configuration files.

These configurations may be composed from multiple files using Hydra's `defaults` mechanism.

For example:

```yaml
defaults:
  - dataset: example
  - model: baseline
  - visualization: standard
  - _self_

experiment:
  seed: 42
```

A single analysis may therefore depend on several YAML files.

Tracking only the top-level configuration is insufficient.

### Recommended approach

Track the actual configuration files used during composition.

For example:

```text
Jacobson analysis
    ├── jacobson_2009.yaml
    ├── dataset/example.yaml
    ├── model/baseline.yaml
    └── visualization/standard.yaml
```

The dependency tracker should identify which configuration sources are selected for each analysis.

Hydra's Compose API can be used to compose configurations programmatically.

Documentation:

- [Hydra Compose API](https://hydra.cc/docs/advanced/compose_api/)
- [Hydra command-line flags](https://hydra.cc/docs/advanced/hydra-command-line-flags/)
- [Hydra defaults lists](https://hydra.cc/docs/1.3/tutorials/basic/your_first_app/defaults/)

Hydra diagnostic commands such as:

```bash
python your_pipeline.py --info defaults-tree
python your_pipeline.py --info defaults
```

can help inspect configuration composition.

However, parsing diagnostic CLI output should not be the primary production dependency-tracking mechanism.

### Additional considerations

Configuration dependencies can originate from:

1. Ordinary YAML files.
2. Hydra defaults lists.
3. Python `ConfigStore` registrations.
4. Custom OmegaConf resolvers.
5. Dynamically loaded configuration files.

Python `ConfigStore` registrations should also be tracked as Python source dependencies.

Custom resolvers that load external resources should explicitly expose or declare those resources as dependencies.

### File-level versus configuration-key-level tracking

There are two possible levels of granularity.

**File-level tracking**

Any change to a configuration file triggers the relevant analysis.

This is simpler and more conservative.

**Configuration-key-level tracking**

Only changes to configuration values actually consumed by a rule trigger that rule.

For example, changing a logging configuration should not necessarily rerun a plotting analysis.

This is potentially more efficient but requires reliable mapping between configuration fields and computational dependencies.

Start with file-level tracking unless there is a demonstrated performance problem.

---

## 8. Combining Python and Hydra dependencies

The intended dependency model is:

```text
Snakemake rule
    │
    ├── Data dependencies
    │
    ├── Python implementation dependencies
    │       ├── Entry-point module
    │       └── Transitive imported modules
    │
    ├── Hydra configuration dependencies
    │       ├── Main configuration
    │       ├── Selected defaults
    │       └── Additional configuration sources
    │
    └── Other runtime resources
            ├── Templates
            ├── Static files
            └── External resources
```

Each rule should declare only the dependencies relevant to its computation.

A conceptual rule might look like:

```python
rule plot_results:
    input:
        data="results/{sample}/statistics.parquet",
        code=python_dependencies(
            "orbitdet.visualization.plotting"
        ),
        config=hydra_dependencies("plot_results")
    output:
        "results/{sample}/plot.png"
    script:
        "scripts/plot.py"
```

`python_dependencies` and `hydra_dependencies` are proposed helper interfaces, not built-in Snakemake functions. They still need to be implemented.

### Dependency manifests

A useful intermediate representation would be a per-rule manifest:

```json
{
  "plot_results": {
    "python": [
      "orbitdet/visualization/plotting.py",
      "orbitdet/visualization/common.py"
    ],
    "config": [
      "config/plotting.yaml",
      "config/visualization/standard.yaml"
    ]
  }
}
```

The manifest should be generated from the actual dependencies.

Avoid making every rule depend on one global manifest: that would reintroduce unnecessary coupling.

Also ensure that changes to dependency discovery itself, including changes to configuration composition, invalidate or regenerate the relevant manifests. A stale manifest could otherwise cause Snakemake to miss dependencies.

---

## 9. Output organisation and reproducibility

Each paper should have a separate output directory:

```text
results/
├── jacobson_2009/
│   ├── dataset_a/
│   ├── dataset_b/
│   └── figures/
│
└── wang_2025/
    ├── dataset_a/
    ├── dataset_b/
    └── figures/
```

For reproducibility, preserve:

- Paper-specific Hydra configurations.
- Input dataset identifiers and versions.
- Preprocessing choices.
- Model and analysis parameters.
- Python package versions.
- Git commit or release identifier.
- Relevant implementation and configuration dependencies.

Consider recording the resolved Hydra configuration alongside each analysis output.

This makes it easier to determine exactly which settings were used for a particular result.

---

## 10. When to use separate Snakefiles

The current recommendation is one main Snakefile with included analysis workflows.

Separate Snakefiles become more attractive if:

- Each paper requires a different software environment.
- Analyses have substantially different execution requirements.
- Different teams maintain different analyses.
- Analyses need to be distributed as independent pipelines.
- The workflows have very little shared infrastructure.

A future arrangement could use separate entry points:

```text
Snakefile
Snakefile.jacobson
Snakefile.wang
```

Each entry point could include the relevant shared and analysis-specific rules.

This is not necessary initially.

---

## 11. Outstanding design decisions

The following questions should be resolved before implementing the dependency-tracking system.

### Hydra configuration architecture

- Are configurations composed entirely from ordinary YAML files?
- Are Hydra `defaults` lists used extensively?
- Are any configurations registered through Python `ConfigStore`?
- Are custom OmegaConf resolvers used?
- Do any resolvers load external files dynamically?

These details determine how Hydra dependencies should be discovered reliably.

### Analysis execution

- Should each paper be runnable independently?
- Should there be a single workflow that runs all analyses?
- Will multiple analyses share intermediate results?
- Will the same datasets be used across multiple papers?

### Python dependency tracking

- Is `orbitdet` installed as a package or imported directly from the project directory?
- Are there dynamic imports?
- Are non-Python resources loaded by the library?
- Should dependency tracking include third-party package versions?

### Configuration granularity

- Is file-level invalidation sufficient?
- Is there a need to track individual configuration keys?
- Should each analysis preserve a fully resolved Hydra configuration?

---

## 12. Recommended implementation sequence

Implement the architecture incrementally.

| Phase | Task | Expected result |
|---|---|---|
| 1 | Establish the project-level Snakefile and analysis subworkflows | Independent paper workflows |
| 2 | Organise shared functionality in `orbitdet` | Reduced code duplication |
| 3 | Implement Python import dependency discovery | Fine-grained source-code tracking |
| 4 | Implement Hydra configuration dependency discovery | Fine-grained configuration tracking |
| 5 | Combine dependencies into per-rule manifests | Unified dependency model |
| 6 | Test invalidation behaviour | Only relevant rules and downstream outputs rerun |
| 7 | Add reproducibility metadata | Traceable analysis outputs |

### Essential tests

At minimum, test the following scenarios:

1. Change a plotting module: only plotting and its downstream outputs rerun.
2. Change a shared preprocessing module: all dependent analyses rerun.
3. Change a Jacobson-specific module: Wang outputs remain untouched.
4. Change a Jacobson-specific Hydra configuration: only affected Jacobson rules rerun.
5. Change a shared Hydra configuration: all rules depending on it rerun.
6. Change an unrelated configuration: unaffected rules remain up to date.
7. Change the Hydra defaults composition: dependency manifests are regenerated.

---

## 13. Final recommendation

Adopt the following architecture:

- **One project-level Snakefile** as the main workflow entry point.
- **One `.smk` subworkflow per paper** to maintain analytical independence.
- **A shared `orbitdet` library** for reusable computational functionality.
- **Separate Hydra configurations** for each paper.
- **Python import-graph dependency tracking** for fine-grained source-code invalidation.
- **Hydra composition-aware dependency tracking** for configuration invalidation.
- **Paper-specific output directories** and reproducibility metadata.

The most important architectural principle is to make dependencies explicit and granular.

This will allow the project to grow from two paper reproductions into a larger collection of analyses without creating an increasingly monolithic workflow or rerunning unrelated computations whenever a small part of the codebase changes.
---

## 14. Implementation status

This section records what was actually built, and where it departs from the
proposal above. The proposal's reasoning is left intact; this is the delta.

### 14.1 What exists

```text
workflow/
├── Snakefile                  # configfile + includes + rule all
├── config.yaml                # which experiments exist
├── depgraph.py                # static Python import closure (§6)
├── rules/
│   ├── common.smk             # 9 resolve_*_config rules + 10 stage rules + report
│   └── figures.smk            # 21 figure rules, each with a report() caption
└── experiments/
    ├── fit_to_ephemeris.smk
    └── classic_triton_state.smk

src/orbitdet/pipeline/         # the stage library (§5)
├── config.py                  # Hydra compose API, no @hydra.main
├── context.py                 # RuntimeContext without HydraConfig
├── setup.py                   # kernel furnishing + environment preparation
├── artifacts.py               # .tudat read/write, metadata sidecar
├── resolve.py                 # config slicing (STAGE_KEYS)
├── kernels.py  observations.py  observation_models.py
├── residuals.py  outliers.py  weighting.py
├── dynamics.py  estimation.py  propagation.py  covariance.py
├── figures.py  estimation_figures.py  propagation_figures.py
├── tables.py  report.py
└── run_*.py                   # one CLI per stage

report/
├── workflow.rst               # landing-page description (report: directive)
├── report.rst                 # summary caption
├── figures/*.rst              # one caption per figure
└── sweeps/sweep_comparison.rst # caption for the comparison charts
```

### 14.2 The stage DAG

```text
resolve_<stage>_config   (one rule per config slice)
        │
        ▼
     kernels ──► observations ──► ephemeris_residuals
                                        │
                                        ▼
                                 outlier_rejection
                                        │
                                        ▼
                                    weighting
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            prefit_propagation      estimation ──► postfit_propagation
                                                            │
                                                            ▼
                                                       covariance
                                                            │
        ┌───────────────────────────────────────────────────┘
        ▼
   figures (21 rules) + weight_tables + report
```

### 14.3 File boundaries and the serialization they rely on

Every arrow above is a file. The boundaries were chosen so that each one is an
artifact tudatpy serializes natively:

| Boundary | Artifact | Serialization |
|---|---|---|
| observations → residuals | `observations.tudat` | `ObservationCollection.save_to_binary` |
| residuals → outliers | `observations_with_residuals.tudat` | same; **residuals persist** |
| outliers → weighting | `outliers_applied.tudat` | same; **weights persist** |
| weighting → estimation | `weighted.tudat` | same |
| estimation → covariance | `estimation_output.tudat` | `EstimationOutput.save_to_binary` |
| propagation → figures | `*_propagation_result.tudat` | `SimulationResults.save_to_binary` |

Two findings made this possible, both verified empirically:

1. **`SingleObservationSet` serializes residuals and weights.** Its `save`
   method writes `residuals_`, `weightMatrixType`, `diagonalWeights`,
   `blockWeights` and `fullWeights`. This is what lets outlier rejection and
   weighting be separate stages rather than one combined preprocessing step.

2. **`EstimationOutput` carries the covariance-propagation payload.** The
   state-transition interface needed for covariance propagation has no
   serialization of its own, so it would normally force the covariance
   propagation to run inside the estimation process. tudatpy stores the
   variational results and interface metadata inside `EstimationOutput` when the
   estimation runs with `save_state_history_per_iteration=True`, and
   `create_state_transition_interface_from_estimation_output` rebuilds the
   interface from that payload. The covariance stage therefore runs in a fresh
   process from the file alone.

   **This makes `save_state_history_per_iteration=True` a hard requirement** of
   the pipeline, not an optimisation. The estimation stage sets it and fails
   loudly if the payload is absent. Only single-arc estimation stores it.

### 14.4 Where this departs from the proposal

**Aim was removed, not wrapped.** The proposal did not mention Aim, but the
project used it for experiment tracking. It is gone: `orbitdet.reproducibility.aim`,
the `AimLogHandler` logging sink, the `aim:` config block, and the Aim-specific
scripts. Reporting is now Snakemake's `--report`, driven by `report()` directives
on the figure and report rules plus the caption files under `report/`; see §16.

**The demo pipeline was discarded.** An earlier `workflow/` demo covered only
pre-fit residuals. It was replaced wholesale. Its `depgraph.py` was kept: the
static import closure is the right answer to §6 and is reused unchanged.

**Config slicing is per stage, not per experiment.** §7 proposes file-level
tracking of the Hydra tree. That is implemented, but with one refinement: the
tree is composed once per stage into `results/<experiment>/config/<stage>.yaml`,
and each stage rule depends only on its own slice. A single combined config
would mean editing a figure title invalidates the observation build. Verified:

| Edit | Invalidates |
|---|---|
| `conf/figures/residuals.yaml` | `resolve_figures_config` only |
| `conf/kernels/jacobson2009.yaml` | all 9 resolve rules |

**`conf/experiment/` is excluded from the shared globs.** It holds one file per
experiment, so globbing it would make every experiment depend on every other
experiment's config. Each resolve rule adds its own experiment's file through an
input function instead.

**Stages use `sys.executable`, not `python`.** A bare `python` resolves through
`PATH` and can silently select a different environment, where the `orbitdet`
import fails. The Snakefile captures the launching interpreter.

**Rules are written out literally.** Snakemake's `rule` syntax is not valid
Python, so rules cannot be generated with `exec` or a comprehension. The
`resolve_*_config` and `figure_*` rules are therefore explicit, even though they
are highly repetitive.

### 14.5 Dependency tracking, verified

Python source dependencies use `depgraph.stage_sources`, which returns the
transitive import closure per stage entry module. Closures are graded rather
than flat:

| Stage | Files |
|---|---|
| `resolve` | 13 |
| `kernels` | 56 |
| `observations` | 64 |
| `estimation` | 68 |
| `figure` | 80 |

Isolation checks (asserted in `test/test_depgraph.py`):

- `figure` does **not** depend on `pipeline/estimation.py`
- `estimation` does **not** depend on `visualization/residuals.py`
- `kernels` depends on neither

So editing a plotter re-renders figures and nothing else, and editing the
estimator does not re-download kernels.

`depgraph` refuses to guess when a module uses a dynamic import
(`importlib`, `__import__`, `exec`, `eval`, or a module-level `__getattr__`),
because an incomplete closure means stale outputs rather than a visible failure.
`test_real_package_has_no_dynamic_imports` guards that assumption.

### 14.6 Running it

```bash
snakemake -s workflow/Snakefile -n                     # dry run, both experiments
snakemake -s workflow/Snakefile --cores 4              # execute everything
snakemake -s workflow/Snakefile fit_to_ephemeris       # one experiment
snakemake -s workflow/Snakefile --report report.html   # HTML report
snakemake -s workflow/Snakefile --dag | dot -Tpdf > dag.pdf
```

Job counts: 72 for both experiments, 33 for `fit_to_ephemeris`, 38 for
`classic_triton_state`.

### 14.7 Adding an experiment

1. Add the name to `workflow/config.yaml`.
2. Create `workflow/experiments/<name>.smk` declaring its `FIGURES` list and a
   target rule with a `.complete` stamp.
3. Ensure `conf/experiment/<name>.yaml` composes.

No stage rule needs to change: the shared rules are experiment-agnostic, and the
figure rules are selected by name from `workflow/rules/figures.smk`.

### 14.8 Known limitations

- **Single-arc estimation only.** The covariance-propagation payload is stored
  only for single-arc runs, so a multi-arc experiment cannot use the covariance
  stage as a separate process.
- **`save_state_history_per_iteration=True` increases output size**, because the
  variational results are serialized into `estimation_output.tudat`.
- **The figure rules are explicit and repetitive.** This is a Snakemake
  constraint, not a design choice; see §14.4.
- **`scripts/solve_least_squares.py` still exists** as the single-process
  interactive entry point. It duplicates the pipeline's logic and should be
  retired once the workflow has been validated on a full run.

### 14.9 What the first real run exposed

The first end-to-end execution of `fit_to_ephemeris` failed on three rules. All
three were implementation bugs, not design flaws, but two of them encoded wrong
assumptions worth recording.

**1. A stage's config keys are set by what the *environment build* reads, not by
the stage's own primary input.**

`STAGE_KEYS["prefit"]` originally omitted `datasets`, on the reasoning that the
pre-fit stage loads an already-built collection from disk and therefore does not
need dataset configuration. That is wrong: the environment build resolves a
*tabulated* ephemeris by looking up `cfg.datasets.<source_dataset>` in
`_configure_ephemeris_model`. Any stage calling `prepare_environment` needs
`datasets`, whatever else it does.

The failure was `ValueError: Voyager dataset configuration 'voyager' is required
to build a tabulated ephemeris for Voyager 2`, and it only appeared for
`classic_triton_state`, because that experiment uses
`tabulated_from_ancillary_file`. `fit_to_ephemeris` uses SPICE kernels, so it
had been passing all along — a reminder that a stage exercised by only one
experiment's config is not exercised.

`datasets` now lives in `_ENVIRONMENT_KEYS`, so every environment-building stage
receives it. `figures` is the only stage that never builds an environment, and
is the only one exempt.

**2. A figure name selects a plotter *and* a set filter, not just a plotter.**

The figure registry was keyed on plotter class names (`normalized_residuals`),
while the Snakefile and experiment files ask for config keys
(`prefit_normalized_residuals`), so every prefixed figure raised
`Unknown collection figure`.

Fixing the keys alone would have been wrong, though, because it would have
silently changed what the figures *showed*. The original script passed
**different subsets** of the collection to different plotters:

| Figure | Set filter | Why |
|---|---|---|
| `*_residuals*` (non-scan) | all sets | |
| `*_scan` | Gaia only | needs the archived `position_angle_scan` |
| `*_normalized_residuals` (non-scan) | **non-Gaia only** | a Gaia residual is meaningless without the scan rotation |

So the registry is now a table of `FigureSpec(plotter, select, styled)`, and the
name selects a combination. When a filter matches nothing — a dataset with no
Gaia observations has no scan-frame figure — the stage writes a placeholder PDF
naming the reason, because Snakemake requires every declared output to exist and
a silently missing file would be worse than an explicit "no figure produced".

**3. In an unconditional DAG, every stage must tolerate its upstream's
"disabled" state.**

`weight_tables` failed with `Empty weights table ...; nothing to summarise.`
because `fit_to_ephemeris` declares no `weighting:` block, so the weights CSV is
headers-only. The rule is unconditional by design (declaring outputs
conditionally on config makes the DAG fragile), so the stage now writes
schema-correct empty tables instead of raising. A reader can then tell that
weighting was *off* rather than that a step was *skipped*.

---

## 15. Sweep support

Sweeps are declared as Hydra multirun configs under `conf/sweep/` (see §15.1) and
executed by the workflow as a grid of runs under their own `sweeps/` root.

### 15.1 How a sweep is declared

A sweep file is an experiment config plus a `hydra.sweeper.params` block. Hydra
takes the Cartesian product of the value lists:

```yaml
defaults:
  - /experiment: fit_to_ephemeris
hydra:
  sweeper:
    params:
      estimation.parameters_to_estimate.neptune_GM: true,false
      estimation.parameters_to_estimate.triton_GM: true,false
```

Two override styles appear in this repository and both reduce to the same thing:

| Style | Example | Effect |
|---|---|---|
| config-group | `parameters: initial_state,state_plus_gravity` | swaps `conf/parameters/<name>.yaml` |
| dotted key | `estimation.parameters_to_estimate.neptune_GM: true,false` | patches one nested key |

Note that `true,false` is **not** a YAML list. It is a plain string that Hydra's
own sweeper splits on commas. `orbitdet.pipeline.sweeps` splits it too; treating
it as a single value silently collapses a 32-combination sweep to one job.

### 15.2 Enumeration

`orbitdet.pipeline.sweeps.enumerate_combos` reads `hydra.sweeper.params` from the
raw YAML (composing through Hydra would resolve the launcher, and these files pin
`submitit_local`, which the workflow does not need) and produces the product.

The product order matches Hydra's own `BasicSweeper` — last declared key varies
fastest — because the combination index becomes the directory name. If the order
changed, existing result directories would silently point at different parameter
combinations. `test_sweeps.py` asserts the order.

Each combination gets a zero-padded id (`00`, `01`, …) used as its directory
name, and the mapping is written to `sweeps/<sweep>/manifest.json` so a result
directory is self-describing.

### 15.3 Layout

```text
sweeps/<sweep>/
├── manifest.json              combo id -> overrides
├── config/figures.yaml        styling for the comparison charts
├── <combo>/
│   ├── config/{estimation,covariance}.yaml
│   ├── estimation/            estimation_output.tudat + summary
│   ├── propagation/           the combination's own post-fit trajectory
│   └── covariance/            formal errors + summary
├── sweep_telemetry.csv        every combination's metrics in one table
└── figures/sweep_comparison_*.pdf
```

Sweeps use their own root rather than nesting under `results/<experiment>/`
because **a Snakemake wildcard cannot span a `/`**. `results/{experiment}/x` and
`results/{experiment}/{combo}/x` are therefore distinct patterns, and two rules
would both claim the same file, producing an ambiguity error. A separate root
removes the overlap.

### 15.4 What is shared and what is not

Only `estimation.parameters_to_estimate.*` varies in the `fit_to_ephemeris`
sweep, and nothing before the estimation stage reads it. Re-running the
observation build and pre-fit propagation for 32 combinations would multiply the
most expensive work in the pipeline for no gain, so combinations depend on the
base experiment's `preprocessing/weighted.tudat` and resolve only their own
estimation-facing config.

Two consequences, both deliberate:

**The estimation and the post-fit propagation are per combination.** The
trajectory is not shared even though it is expensive, because `run_covariance`
defines its RSW frame from a trajectory and each combination estimates a
*different* state. Sharing the base experiment's trajectory rotated combination
31's covariance into a frame belonging to a different fit — a 16-parameter state
while the combination had estimated 6. `sweep_postfit_propagation` exists to
close that gap, and a test asserts `sweep_covariance` reads the combination's own
propagation output.

**Overrides that could reach a shared stage are refused.**
`run_resolve_sweep_config` raises if an override names a prefix the shared stages
read (`data`, `datasets`, `integrator`, `weighting`, …). The check is a
conservative over-approximation: refusing too much is cheap, whereas reusing an
observation collection built with different data silently fits something other
than the configuration describes. This is why `params_x_data` and the other
data-varying sweeps are not yet wired into the workflow — they need
per-combination preprocessing.

### 15.5 Telemetry and the comparison figures

`sweep_telemetry.csv` collects each combination's estimation and covariance
summaries into one table: residual RMS, iteration counts, duration, parameter
count, and whether the covariance is finite. The telemetry rule depends on *all*
combinations, so it is the sweep's barrier and re-runs if any one changes.

`SweepComparison` renders one bar chart per metric. The design choices:

- **One figure per metric**, because the metrics differ by orders of magnitude
  and a shared axis would flatten the informative one.
- **Bars sorted by the metric**, because with 32 combinations the ordering *is*
  the result; the x tick labels carry the combination id so the ordering stays
  traceable to the manifest.
- **Colour encodes the estimated-parameter count**, so the chart shows *why* the
  metric changes rather than restating the bar height.
- **Non-finite values are drawn grey and counted in a corner annotation**, so a
  degenerate run is visible rather than absent.

For `fit_to_ephemeris` this is the interesting result: the base experiment
estimates a parameter set the observations cannot constrain (normal matrix rank 3
of 16, so the covariance inverts to NaN), while combinations estimating fewer
parameters produce a finite covariance. The sweep turns "this fit is degenerate"
into "these parameter subsets are estimable", which is a result the single-run
report cannot express.

### 15.6 Running a sweep

```bash
# Enumerate, estimate all combinations, aggregate, and chart:
snakemake -s workflow/Snakefile sweeps/fit_to_ephemeris/.complete --cores 4

# Just the manifest, to see the grid without running it:
snakemake -s workflow/Snakefile sweeps/fit_to_ephemeris/manifest.json --cores 1

# One combination, for a smoke test:
snakemake -s workflow/Snakefile \
  sweeps/fit_to_ephemeris/00/estimation/estimation_summary.json --cores 1
```

A sweep target cannot be named `sweep_<name>`: the `sweep` rule uses a
`{sweep}` wildcard, so there is no rule by that literal name. Target the
completion stamp (`sweeps/<sweep>/.complete`) instead.

Cost: 32 combinations × ~5 min estimation ≈ 2.7 h at `--cores 4`, plus the
per-combination propagation.

### 15.7 Adding a sweep

1. Add an entry under `sweeps:` in `workflow/config.yaml` naming the experiment
   whose preprocessing it reuses.
2. Ensure `conf/sweep/<name>.yaml` declares `hydra.sweeper.params`.
3. Check the overrides only touch estimation-facing settings; otherwise give the
   combinations their own preprocessing, as the shared-stage guard requires.

No rule needs editing: `_experiment_for` reads the mapping from the workflow
config.

---

## 16. Reports

Reporting is Snakemake's own `--report`, not a bespoke template. The workflow
only has to say *what* goes in the report; Snakemake owns the layout, the
navigation and the provenance.

The `.html` form **embeds every result** as a base64 data URI, so the file is
self-contained and portable but grows with the figure count — with 13 PDF
figures it is already a few megabytes. Pass a `.zip` path instead when results
are large or when a directory with an `htmlindex` is included: the suffix
selects the mode, and the results are then stored alongside the HTML rather than
inlined.

### 16.1 Generating one

```bash
# Everything: every experiment, every sweep.
snakemake -s workflow/Snakefile --report report.html

# One experiment's figures and summary only.
snakemake -s workflow/Snakefile --report report.html fit_to_ephemeris

# Just the sweep comparison charts.
snakemake -s workflow/Snakefile --report report.html sweeps/fit_to_ephemeris/.complete

# Self-contained results/ directory instead of one embedded file:
snakemake -s workflow/Snakefile --report report.zip
```

`--report` does not run the workflow. It evaluates the DAG and renders whatever
is already on disk, so **the target you name decides the report's contents**.
Name a single experiment and the report contains that experiment's results; name
nothing and it contains all of them. This is the same target selection the
workflow already uses, so the report needs no separate configuration.

Two consequences worth knowing before the first run:

- **Anything missing is a hard error, not an empty section.** `--report` raises
  `File marked for report but does not exist` if a target's outputs have not
  been built. Reports are generated *after* a successful run, not instead of
  one.
- **`--report` reads the provenance metadata in `.snakemake/`.** Deleting that
  directory (or running with `--drop-metadata`) means the run times, rule
  parameters and wildcards on each result page are missing, though the results
  themselves still render. It is gitignored, so a fresh clone must run before it
  can report.

### 16.2 How the content is declared

Three mechanisms, all in the rule files:

| Mechanism | Where | Effect |
|---|---|---|
| `report(value, caption=..., category=...)` | on a rule's `output:` | makes that file a report result, filed under `category` |
| `report: <path>` | `workflow/Snakefile` | the landing-page description, rendered from RST |
| caption `.rst` files | `report/` | per-result prose, rendered and shown with the result |

Captions and the workflow description go through `_caption(...)` in
`common.smk`, which builds an **absolute** path under `report/`. That is not
cosmetic: Snakemake resolves a caption relative to the *including* file, so a
literal `caption="report/figures/x.rst"` written inside `workflow/rules/`
resolves to `workflow/rules/report/figures/x.rst` and fails with a
`FileNotFoundError` naming a path that does not look like it should exist. The
helper removes the question. `test_every_caption_file_exists` and
`test_caption_paths_are_built_from_the_helper` guard both halves of that — the
files' existence and the call convention.

Categories in use: `residuals`, `estimation`, `trajectory`, `rsw`,
`formal_errors`, `rsw_uncertainty`, `Tables`, `Summary`, and
`Sweep comparison`. They are declared per rule, so moving a figure between
categories is a one-line edit and does not re-render anything.

### 16.3 What the report contains

Beyond the declared results, `--report` derives three things from the workflow
itself: the **rule graph**, a **run-time** summary per rule, and the **packages**
and versions in the environment. None of it is written by the pipeline — it
comes from the DAG and the persisted job records — which is why it cannot drift
out of sync with what actually ran.

Result formatting follows the file type: PDF figures are embedded in an iframe,
images inline, and anything else (CSV, Markdown, `.npz`) is offered as a
download rather than previewed. The per-experiment summary is a `.md` file, so
it appears as a download link; its numbers are also repeated in the figure
captions where they matter, so the report is readable without opening it.

### 16.4 A stale `report.html` in the repository root

`report.html` at the repository root is tracked by git and is a leftover from
the discarded demo pipeline (§14.4). It was generated before the current
workflow existed and embeds an empty result set. Regenerating it overwrites the
file in place.


