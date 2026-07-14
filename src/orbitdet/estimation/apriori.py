import logging

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_apriori_covariance_matrix(cfg: DictConfig) -> np.ndarray:
    """
    Constructs the inverse a priori covariance matrix based on the configuration.

    The parameter order is:
      1. Initial state (always first, from ``bodies_to_propagate``):
         [x, y, z, vx, vy, vz] per propagated body.
      2. Additional parameters in ``cfg.estimation.parameters_to_estimate`` order
         (excluding the initial_state entries).

    For each parameter block the a priori sigma is taken from (in priority order):
      - Per-parameter config, e.g.
        ``initial_state: {apriori: [sigma_pos, sigma_vel]}``
      - Global config ``cfg.estimation.apriori``
      - A huge default (1e30) if nothing is specified, meaning *no constraint*.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra / OmegaConf configuration object.

    Returns
    -------
    np.ndarray
        Diagonal **inverse** a priori covariance matrix (shape ``N x N``),
        ready to be passed to
        ``est_an.EstimationInput(…, inverse_apriori_covariance=…)``.
    """
    # ------------------------------------------------------------------
    # 1. Count propagated bodies – each contributes 6 state parameters
    # ------------------------------------------------------------------
    n_propagated = len(cfg.bodies_to_propagate)

    # ------------------------------------------------------------------
    # 2. Extract global apriori defaults
    # ------------------------------------------------------------------
    global_apriori = cfg.estimation.get("apriori", {})

    # ------------------------------------------------------------------
    # 3. Extract per-parameter apriori values from dict-form entries
    #    e.g.  - initial_state: {apriori: [1.0e7, 1.0e2]}
    # ------------------------------------------------------------------
    per_param_apriori: dict[str, list[float]] = {}
    for param_entry in cfg.estimation.parameters_to_estimate:
        if isinstance(param_entry, DictConfig):
            param_name = next(iter(param_entry.keys()))
            param_config = param_entry[param_name]
            if "apriori" in param_config:
                per_param_apriori[param_name] = list(param_config.apriori)

    # ------------------------------------------------------------------
    # 4. Helper: get sigma for a parameter block
    # ------------------------------------------------------------------
    def _get_sigma(param_name: str, default: float = 1e30) -> float:
        """Return a single sigma for *param_name*."""
        # 1st priority – per-parameter a priori
        if param_name in per_param_apriori and len(per_param_apriori[param_name]) > 0:
            return per_param_apriori[param_name][0]
        # 2nd priority – global a priori
        if hasattr(global_apriori, param_name):
            return global_apriori[param_name]
        # 3rd priority – default (no constraint)
        return default

    # ------------------------------------------------------------------
    # 5. Build inverse-variance list in parameter order
    # ------------------------------------------------------------------
    inv_var_list: list[float] = []

    # ---- 5a. Initial state(s) -----------------------------------------
    initial_apriori = per_param_apriori.get("initial_state", None)
    if initial_apriori is not None:
        sigma_pos = float(initial_apriori[0])
        sigma_vel = float(initial_apriori[1])
    elif hasattr(global_apriori, "position") and hasattr(global_apriori, "velocity"):
        sigma_pos = float(global_apriori.position)
        sigma_vel = float(global_apriori.velocity)
    else:
        sigma_pos = sigma_vel = 1e30

    for _ in range(n_propagated):
        inv_var_list.extend([1.0 / sigma_pos**2] * 3)  # x, y, z
        inv_var_list.extend([1.0 / sigma_vel**2] * 3)  # vx, vy, vz

    # ---- 5b. Additional parameters (in config order) ------------------
    # Known parameter sizes (number of estimatable parameters per type)
    #   - iau_rotation_model_pole : 2  (RA, Dec)
    #   - neptune_GM             : 1  (gravitational parameter)
    #   - neptune_j2_j4          : 2  (C20, C40)
    # Unknown types default to 1.
    size_map: dict[str, int] = {
        "iau_rotation_model_pole": 2,
        "neptune_GM": 1,
        "neptune_j2_j4": 2,
    }

    for param_entry in cfg.estimation.parameters_to_estimate:
        # Unwrap dict entries
        if isinstance(param_entry, DictConfig):
            param_name = next(iter(param_entry.keys()))
        else:
            param_name = param_entry

        if param_name == "initial_state":
            continue  # already handled above

        n_params = size_map.get(param_name, 1)
        sigma = _get_sigma(param_name)
        inv_var_list.extend([1.0 / sigma**2] * n_params)

    # ------------------------------------------------------------------
    # 6. Assemble and return the diagonal inverse covariance matrix
    # ------------------------------------------------------------------
    inverse_apriori_covariance = np.diag(inv_var_list)

    logger.info(
        "Constructed inverse a priori covariance matrix "
        f"(size {inverse_apriori_covariance.shape[0]}×"
        f"{inverse_apriori_covariance.shape[1]})."
    )
    logger.debug(f"Inverse a priori covariance diagonal:\n{np.diag(inverse_apriori_covariance)}")

    return inverse_apriori_covariance
