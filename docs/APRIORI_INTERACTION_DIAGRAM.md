# `apriori.py` interaction diagram

UML **sequence (interaction) diagram** for
[`src/orbitdet/estimation/apriori.py`](../src/orbitdet/estimation/apriori.py),
which builds the diagonal inverse a priori covariance matrix that regularises the
Gauss–Newton normal matrix during estimation.

## High-level flow

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Caller<br/>(scripts/solve_least_squares.py)
    participant Apriori as apriori<br/>get_apriori_covariance_matrix
    participant Estimators as estimators
    participant Cfg as cfg<br/>(DictConfig)
    participant ParamSet as parameter_set<br/>(EstimatableParameterSet)

    Caller->>Apriori: get_apriori_covariance_matrix(cfg, parameter_set)

    Note over Apriori: 1. Read configured sigmas
    Apriori->>Cfg: estimation.get("apriori", {}) → global_apriori
    Apriori->>Apriori: _extract_per_parameter_apriori(cfg)
    Apriori->>Estimators: iter_parameter_entries(cfg)
    Estimators->>Cfg: estimation.parameters_to_estimate.items()
    Cfg-->>Estimators: [(key, enabled), ...]
    Estimators-->>Apriori: [(key, enabled), ...]
    loop each enabled mapping-form entry
        Apriori->>Cfg: entry.apriori
        Cfg-->>Apriori: [sigma, ...]
    end
    Apriori->>Apriori: per_param_apriori

    Note over Apriori: 2. Nothing configured?
    Apriori->>Apriori: _has_global_apriori(global_apriori)
    alt no per-parameter and no global apriori
        Apriori-->>Caller: None
    else at least one constraint
        Note over Apriori: 3. Resolve block layout from TudatPy
        Apriori->>Estimators: get_parameter_block_layout(cfg, parameter_set)
        Estimators->>ParamSet: get_parameter_identifiers()
        ParamSet-->>Estimators: [identifier, ...]
        loop each identifier
            Estimators->>ParamSet: indices_for_parameter_type(identifier)
            ParamSet-->>Estimators: [(start, size), ...]
            Note right of Estimators: map identifier → config key<br/>(or "initial_state")
        end
        Estimators-->>Apriori: [ParameterBlock(key, start, size), ...]

        Note over Apriori: 4. Build inverse variances (set order)
        loop each block
            alt block.key == "initial_state"
                Apriori->>Apriori: _state_inverse_variances(size, ...)
            else additional parameter
                Apriori->>Apriori: _block_inverse_variances(key, size, ...)
            end
        end

        Note over Apriori: 5. Assemble & validate
        Apriori->>Apriori: np.diag(inv_var_list)
        Apriori->>ParamSet: parameter_set_size
        ParamSet-->>Apriori: expected size
        alt shape mismatch
            Apriori-->>Caller: raise ValueError
        else sizes match
            Apriori-->>Caller: inverse_apriori_covariance (N×N)
        end
    end
```

## Sigma resolution priority

Within `_block_inverse_variances` / `_state_inverse_variances` each block's sigma
is resolved in priority order:

```mermaid
flowchart TD
    Start(["block: key, size"]) --> IsState{"key is initial_state?"}

    IsState -- yes --> StatePerParam{"initial_state:<br/>{apriori: [pos, vel]}?"}
    StatePerParam -- yes --> SPos["sigma_pos, sigma_vel"]
    StatePerParam -- no --> StateGlobal{"global apriori<br/>has position and velocity?"}
    StateGlobal -- yes --> SPos
    StateGlobal -- no --> SUnconstrained["both = 1e30 (unconstrained)"]

    IsState -- no --> PerParam{"per-param<br/>apriori list?"}
    PerParam -- yes --> PLen{"len == size?"}
    PLen -- "len == 1" --> Broadcast["broadcast single sigma"]
    PLen -- "len == size" --> PerScalar["one sigma per scalar"]
    PLen -- "otherwise" --> Raise(["raise ValueError"])
    PerParam -- no --> Global{"global apriori<br/>has key?"}
    Global -- yes --> Broadcast
    Global -- no --> SUnconstrained

    SPos --> Layout
    SUnconstrained --> Layout
    Broadcast --> Layout
    PerScalar --> Layout
    Layout["inverse variance = 1 / sigma^2<br/>• state: 3 pos + 3 vel per body<br/>• size not %6 → pos sigma for whole block"]
    Layout --> Emit(["append entries in parameter-set order"])
```

## Key collaborators

| Participant | Role |
| --- | --- |
| `estimators.iter_parameter_entries` | Normalises plain (`key: true`) and mapping (`key: {value, apriori}`) config forms into `(key, enabled)` pairs. |
| `estimators.get_parameter_block_layout` | Maps each block of the consolidated `parameter_set` to its config key and its real `(start, size)`, in **parameter-vector order** (TudatPy reorders blocks: state → scalars → vectors). |
| `parameter_set` (`EstimatableParameterSet`) | Authoritative source of block identifiers, indices and total size — the reason no per-parameter size table lives in `apriori.py`. |
| `cfg` (`DictConfig`) | Supplies the sigmas from `estimation.apriori` (global) and `estimation.parameters_to_estimate.<key>.apriori` (per-parameter). |

## Notes

- The returned matrix is **diagonal and in parameter-set order**, so its `i`-th
  diagonal entry is the inverse variance of the `i`-th scalar in
  `parameter_set.parameter_vector`.
- If **nothing** is configured anywhere, the function returns `None` (no prior),
  and the caller builds an `EstimationInput` without `inverse_apriori_covariance`.
- The `ParameterBlock` objects themselves are not mutated here; they are only read.
