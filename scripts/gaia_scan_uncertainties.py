"""Export the per-transit Gaia uncertainties rotated into the scan frame.

For every Gaia transit this writes a CSV with the archived random/systematic
RA/Dec errors and their along-scan (AL) / across-scan (AC) counterparts, obtained
by rotating the full ``(RA, Dec)`` covariance into the scan frame
(``C_scan = T C T^T``, see :func:`orbitdet.visualization.residuals._scan_frame_sigmas`).

The Gaia transit covariance is essentially rank-1 (the along-scan direction is
well measured, the across-scan direction is nearly unconstrained), so the RA/Dec
errors are large while the AL error is small.  This table makes that explicit.

Usage:
    python scripts/gaia_scan_uncertainties.py --config-name=experiment/gaia_triton_state
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from orbitdet.data.gaia import GaiaQuery
from orbitdet.reproducibility import enforce_initialization, initialize
from orbitdet.visualization.residuals import _scan_frame_sigmas

logger = logging.getLogger(__name__)

#: radians -> milliarcseconds
_RAD_TO_MAS = np.rad2deg(1.0) * 3600.0 * 1e3


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiment/gaia_triton_state",
)
@enforce_initialization
def main(cfg: DictConfig):
    initialize(cfg)

    # Locate the Gaia dataset config (source_ids / filter_outcomes).
    gaia_cfg = None
    for _, dataset_cfg in cfg.datasets.items():
        if dataset_cfg.get("type", "") == "gaia":
            gaia_cfg = dataset_cfg
            break
    if gaia_cfg is None:
        raise ValueError("No Gaia dataset found in the experiment configuration.")

    source_ids = list(gaia_cfg.get("source_ids", []))
    filter_outcomes = bool(gaia_cfg.get("filter_outcomes", True))

    query = GaiaQuery()
    query.retrieve_data(source_ids=source_ids, filter_outcomes=filter_outcomes)
    table = query.observation_table

    rows = []
    for transit_id, transit in table.groupby("transit_id", sort=False):
        ra_r = transit["ra_error_random"].to_numpy()
        dec_r = transit["dec_error_random"].to_numpy()
        corr_r = transit["ra_dec_correlation_random"].to_numpy()
        ra_s = transit["ra_error_systematic"].to_numpy()
        dec_s = transit["dec_error_systematic"].to_numpy()
        corr_s = transit["ra_dec_correlation_systematic"].to_numpy()
        psi = transit["position_angle_scan"].to_numpy()
        dec = transit["dec"].to_numpy()

        al_r, ac_r = _scan_frame_sigmas(ra_r, dec_r, corr_r, psi, dec)
        al_s, ac_s = _scan_frame_sigmas(ra_s, dec_s, corr_s, psi, dec)

        rows.append(
            {
                "transit_id": transit_id,
                "n_obs": len(transit),
                "epoch_first": float(transit["epoch"].iloc[0]),
                "ra_error_random_mas": ra_r.mean() * _RAD_TO_MAS,
                "dec_error_random_mas": dec_r.mean() * _RAD_TO_MAS,
                "ra_dec_correlation_random": corr_r.mean(),
                "al_error_random_mas": al_r.mean() * _RAD_TO_MAS,
                "ac_error_random_mas": ac_r.mean() * _RAD_TO_MAS,
                "ra_error_systematic_mas": ra_s.mean() * _RAD_TO_MAS,
                "dec_error_systematic_mas": dec_s.mean() * _RAD_TO_MAS,
                "ra_dec_correlation_systematic": corr_s.mean(),
                "al_error_systematic_mas": al_s.mean() * _RAD_TO_MAS,
                "ac_error_systematic_mas": ac_s.mean() * _RAD_TO_MAS,
            }
        )

    df = pd.DataFrame(rows)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gaia_scan_uncertainties.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %d transit(s) to %s", len(df), csv_path)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSummary (mas):")
    print(
        df[
            [
                "ra_error_random_mas",
                "dec_error_random_mas",
                "al_error_random_mas",
                "ac_error_random_mas",
                "ra_error_systematic_mas",
                "dec_error_systematic_mas",
                "al_error_systematic_mas",
                "ac_error_systematic_mas",
            ]
        ]
        .describe()
        .to_string()
    )


if __name__ == "__main__":
    main()
