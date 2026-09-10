"""Summary tables of observation sources and their RMSE statistics.

Produces two table variants from a weights DataFrame:

1. **Per-source** (``build_source``): one row per academic source, with RA and
   DEC statistics side by side, matching the format used in the literature::

    +----------------------------+------+------+-------------+-----------+------+-------------+-----------+
    | Source                     | Obs  | N    | RA type     | RA RMSE   | N    | DEC type    | DEC RMSE  |
    +----------------------------+------+------+-------------+-----------+------+-------------+-----------+
    | Veiga et al. (1996)        | 874  | 51   | Δα cos δ    | 0.′′080   | 51   | Δδ          | 0.′′046   |

2. **Per-group** (``build_group``): one row per time-based group (e.g. per
   observing night), with the same columns plus the group identifier.

The input is the per-observation weights CSV written by
:class:`~orbitdet.observations.weighting.WeightEngine`, which now includes
``source_name``, ``observatory_code``, ``ra_type``, and ``dec_type`` columns
from the dataset metadata.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class WeightSummaryTable:
    """Build summary tables from a weights DataFrame.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment configuration.
    weights_df : pd.DataFrame
        Per-observation weights DataFrame as produced by
        :class:`~orbitdet.observations.weighting.WeightEngine` and saved to
        ``observation_weights.csv``.
    """

    def __init__(self, cfg: DictConfig, weights_df: pd.DataFrame):
        self.cfg = cfg
        self.weights_df = weights_df

    def build_source(self) -> pd.DataFrame:
        """Build the per-source (top-level) summary table.

        Returns
        -------
        pd.DataFrame
            One row per source with columns:
            ``source``, ``dataset_id``, ``observatory_code``, ``n_obs``,
            ``ra_type``, ``ra_rms_arcsec``, ``dec_type``, ``dec_rms_arcsec``.
        """
        df = self.weights_df
        if df.empty:
            return pd.DataFrame()

        source_col = "source_name" if "source_name" in df.columns else "set_id"

        records = []
        for source_name, grp in df.groupby(source_col, sort=False):
            n_obs = len(grp)
            ra_rms = float(np.sqrt(np.mean(np.square(grp["ra_residual_arcsec"].values))))
            dec_rms = float(np.sqrt(np.mean(np.square(grp["dec_residual_arcsec"].values))))
            ra_type = grp["ra_type"].iloc[0] if "ra_type" in grp.columns else ""
            dec_type = grp["dec_type"].iloc[0] if "dec_type" in grp.columns else ""
            dataset_id = grp["dataset_id"].iloc[0] if "dataset_id" in grp.columns else ""
            obs_code = grp["observatory_code"].iloc[0] if "observatory_code" in grp.columns else ""

            records.append(
                {
                    "source": source_name,
                    "dataset_id": dataset_id,
                    "observatory_code": obs_code,
                    "n_obs": n_obs,
                    "ra_type": ra_type,
                    "ra_rms_arcsec": ra_rms,
                    "dec_type": dec_type,
                    "dec_rms_arcsec": dec_rms,
                }
            )

        result = pd.DataFrame(records)
        result = result.sort_values("source")
        result = result.reset_index(drop=True)
        return result

    def build_group(self) -> pd.DataFrame:
        """Build the per-group (time-based) summary table.

        Returns
        -------
        pd.DataFrame
            One row per group with columns:
            ``source``, ``dataset_id``, ``observatory_code``, ``group_id``,
            ``n_obs``, ``ra_type``, ``ra_rms_arcsec``, ``dec_type``,
            ``dec_rms_arcsec``.
        """
        df = self.weights_df
        if df.empty:
            return pd.DataFrame()

        source_col = "source_name" if "source_name" in df.columns else "set_id"

        records = []
        for (source_name, group_id), grp in df.groupby([source_col, "group_id"], sort=False):
            n_obs = len(grp)
            ra_rms = float(np.sqrt(np.mean(np.square(grp["ra_residual_arcsec"].values))))
            dec_rms = float(np.sqrt(np.mean(np.square(grp["dec_residual_arcsec"].values))))
            ra_type = grp["ra_type"].iloc[0] if "ra_type" in grp.columns else ""
            dec_type = grp["dec_type"].iloc[0] if "dec_type" in grp.columns else ""
            dataset_id = grp["dataset_id"].iloc[0] if "dataset_id" in grp.columns else ""
            obs_code = grp["observatory_code"].iloc[0] if "observatory_code" in grp.columns else ""

            records.append(
                {
                    "source": source_name,
                    "dataset_id": dataset_id,
                    "observatory_code": obs_code,
                    "group_id": group_id,
                    "n_obs": n_obs,
                    "ra_type": ra_type,
                    "ra_rms_arcsec": ra_rms,
                    "dec_type": dec_type,
                    "dec_rms_arcsec": dec_rms,
                }
            )

        result = pd.DataFrame(records)
        result = result.sort_values(["source", "group_id"])
        result = result.reset_index(drop=True)
        return result

    def to_string_source(self, table: pd.DataFrame | None = None) -> str:
        """Format the per-source table as a human-readable string."""
        if table is None:
            table = self.build_source()
        if table.empty:
            return "No weight data."

        lines = []
        lines.append("=" * 140)
        lines.append(
            f"{'Source':<40}  {'DS':>7}  {'Obs':>5}  {'N':>5}  {'RA type':<20}  "
            f"{'RA RMSE':>10}  {'N':>5}  {'DEC type':<20}  {'DEC RMSE':>10}"
        )
        lines.append("=" * 140)

        for _, row in table.iterrows():
            lines.append(
                f"{row['source']:<40}  {row['dataset_id']:>7}  {row['observatory_code']:>5}  {row['n_obs']:>5}  "
                f"{row['ra_type']:<20}  {row['ra_rms_arcsec']:>8.4f}  {row['n_obs']:>5}  "
                f"{row['dec_type']:<20}  {row['dec_rms_arcsec']:>8.4f}"
            )

        lines.append("=" * 140)
        total_obs = table["n_obs"].sum()
        lines.append(f"{'Total':<40}  {'':>7}  {'':>5}  {total_obs:>5}")
        return "\n".join(lines)

    def to_string_group(self, table: pd.DataFrame | None = None) -> str:
        """Format the per-group table as a human-readable string."""
        if table is None:
            table = self.build_group()
        if table.empty:
            return "No weight data."

        lines = []
        lines.append("=" * 160)
        lines.append(
            f"{'Source':<40}  {'DS':>7}  {'Obs':>5}  {'Group':<18}  {'N':>5}  "
            f"{'RA type':<20}  {'RA RMSE':>10}  {'DEC type':<20}  {'DEC RMSE':>10}"
        )
        lines.append("=" * 160)

        for _, row in table.iterrows():
            lines.append(
                f"{row['source']:<40}  {row['dataset_id']:>7}  {row['observatory_code']:>5}  {row['group_id']:<18}  "
                f"{row['n_obs']:>5}  {row['ra_type']:<20}  "
                f"{row['ra_rms_arcsec']:>8.4f}  {row['dec_type']:<20}  "
                f"{row['dec_rms_arcsec']:>8.4f}"
            )

        lines.append("=" * 160)
        total_obs = table["n_obs"].sum()
        lines.append(f"{'Total':<40}  {'':>7}  {'':>5}  {'':>18}  {total_obs:>5}")
        return "\n".join(lines)

    def to_latex_source(self, table: pd.DataFrame | None = None) -> str:
        """Format the per-source table as a LaTeX ``booktabs`` table."""
        if table is None:
            table = self.build_source()
        if table.empty:
            return "% No weight data."

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Observation source summary with per-source RMSE statistics.}",
            r"\label{tab:weight_summary_source}",
            r"\begin{tabular}{l r r r l r r l r}",
            r"\toprule",
            r"Source & DS & Obs. & \multicolumn{2}{c}{RA} & \multicolumn{2}{c}{DEC} \\",
            r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
            r"& & & N & Type & RMSE [arcsec] & N & Type & RMSE [arcsec] \\",
            r"\midrule",
        ]

        for _, row in table.iterrows():
            source_escaped = str(row["source"]).replace("&", r"\&")
            lines.append(
                f"{source_escaped} & {row['dataset_id']} & {row['observatory_code']} & {row['n_obs']} & "
                f"{row['ra_type']} & {row['ra_rms_arcsec']:.4f} & {row['n_obs']} & "
                f"{row['dec_type']} & {row['dec_rms_arcsec']:.4f} \\\\"
            )

        total_obs = table["n_obs"].sum()
        lines.extend(
            [
                r"\midrule",
                f"Total & & & {total_obs} & & & {total_obs} & & \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def to_latex_group(self, table: pd.DataFrame | None = None) -> str:
        """Format the per-group table as a LaTeX ``booktabs`` table."""
        if table is None:
            table = self.build_group()
        if table.empty:
            return "% No weight data."

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Observation group summary with per-group RMSE statistics.}",
            r"\label{tab:weight_summary_group}",
            r"\begin{tabular}{l r r l r l r r l r}",
            r"\toprule",
            r"Source & DS & Obs. & Group & \multicolumn{2}{c}{RA} & \multicolumn{2}{c}{DEC} \\",
            r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}",
            r"& & & & N & Type & RMSE [arcsec] & N & Type & RMSE [arcsec] \\",
            r"\midrule",
        ]

        for _, row in table.iterrows():
            source_escaped = str(row["source"]).replace("&", r"\&")
            lines.append(
                f"{source_escaped} & {row['dataset_id']} & {row['observatory_code']} & {row['group_id']} & "
                f"{row['n_obs']} & {row['ra_type']} & {row['ra_rms_arcsec']:.4f} & "
                f"{row['n_obs']} & {row['dec_type']} & {row['dec_rms_arcsec']:.4f} \\\\"
            )

        total_obs = table["n_obs"].sum()
        lines.extend(
            [
                r"\midrule",
                f"Total & & & & {total_obs} & & & {total_obs} & & \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)